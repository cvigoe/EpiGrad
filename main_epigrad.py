# Author: Conor Igoe
# Date: September 27 2021
# Using pretrained networks from PyTorch-Playground: 
# https://github.com/aaron-xichen/pytorch-playground

import matplotlib as mpl
mpl.rcParams['figure.dpi']= 300
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.autograd import Variable
from utee import selector

import tqdm
import copy
import torch.nn as nn
import numpy as np
import random
import math
import torch.optim as optim
import torch.nn.functional as F

from orthonormal_certs import Ortho
from roc import ROC_test
from variant import *

import mlflow
from mlflow.tracking import MlflowClient
import sys
import collections
import pudb

def epigrad_experiment(variant, run_id):
    ID_model_name = variant['ID_model_name']    # ['mnist', 'cifar10', 'svhn']
    OOD_model_name = variant['OOD_model_name']  # ['mnist', 'cifar10', 'svhn']
    batch_size_epi = variant['batch_size_epi']  # [1, 64]
    num_tests = variant['num_tests']            # [1000]
    max_ent = variant['max_ent']                # [False]            
    num_ROC = variant['num_ROC']                # [20]
    MC_iters = variant['MC_iters']              # [1000]

    ID_model, ID_fetcher, _ = selector.select(ID_model_name, cuda=True)
    OOD_model, OOD_fetcher, _ = selector.select(OOD_model_name, cuda=True)

    ID_testing_loader = ID_fetcher(batch_size=1, train=False, 
        val=True)
    OOD_testing_loader = OOD_fetcher(batch_size=1, train=False, 
        val=True)

    optimiser = torch.optim.SGD(ID_model.parameters(), lr=1)
    
    # Generate Histogram
    var_lb_id, var_lb_ood = epistemic_test(id_loader=ID_testing_loader, 
        ood_loader=OOD_testing_loader, max_ent=False, 
        id_dataset_name=ID_model_name, ood_dataset_name=OOD_model_name, 
        network=ID_model, optimizer=optimiser, 
        batch_size_epi=batch_size_epi,num_tests=num_tests, 
        MC_iters=MC_iters)

    # Generate ROC Curves
    ROC_lower = np.log10(min(np.percentile(var_lb_id,1) , np.percentile(var_lb_ood,1)))
    ROC_upper = np.log10(max(np.percentile(var_lb_id,99) , np.percentile(var_lb_ood,99)))

    FPRs, TPRs = [], []
    for threshold in tqdm.tqdm(np.logspace(ROC_lower, ROC_upper, num_ROC)):
        ID_results, count_id, OOD_results, count_ood = ROC_test(
            num_tests=num_tests, threshold=threshold, 
            id_loader=ID_testing_loader, 
            ood_loader=OOD_testing_loader, max_ent=False, 
            id_dataset_name=ID_model_name, 
            ood_dataset_name=OOD_model_name,
            network=ID_model, optimizer=optimiser, 
            batch_size_epi=batch_size_epi,
            certs=None, MC_iters=MC_iters)

        FP, TP = np.sum(ID_results), np.sum(OOD_results)
        N, P = count_id, count_ood
        FPR, TPR = FP/N, TP/P
        FPRs.append(FPR)
        TPRs.append(TPR)

    np.savetxt("data/epi_FPRs_" + run_id + ".csv", FPRs, delimiter=",")
    np.savetxt("data/epi_TPRs_" + run_id + ".csv", TPRs, delimiter=",")

    # Calculate AUC
    auc = calculate_auc(FPRs, TPRs)
    mlflow.log_metric('AUC', auc)
    with open('data/epi_AUCs.csv','a') as fd:
        fd.write(run_id + ', ' + str(auc) + '\n')

    plt.figure(dpi=300)
    plt.plot(FPRs, TPRs, '-o', label='EpiGrad')
    plt.legend(loc='lower right')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.title('ROC Curve; AUC = ' + str(auc) + '\n ID = ' + ID_model_name + '; OOD = ' + OOD_model_name)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('figures/epi_roc_' + run_id + '.pdf')

def epistemic_test(id_loader, ood_loader, max_ent, 
    id_dataset_name, ood_dataset_name, network, 
    optimizer, batch_size_epi, num_tests, MC_iters):
    
    network.eval() 
    var_lb_id = []    
    count_id = 0
    id_shape = None
    MC_grad = None

    for i, (data, target) in enumerate(tqdm.tqdm(id_loader)):
        
        if id_shape is None:
            id_shape = list(data.shape)
        if count_id >= num_tests:
            break

        for iteration in range(MC_iters):
            optimizer.zero_grad()
            logits = network(data.cuda())
            log_probs = F.log_softmax(logits, dim=1)
            labels = []
            for i in range(data.shape[0]):
                labels.append(np.random.choice(10, p=torch.exp(log_probs[i]).cpu().detach().numpy()))
            
            # F.nll_loss gives the negative log-likelihood, and expects a vector of log-probabilities
            loss = -1*F.nll_loss(log_probs, torch.tensor(labels).cuda())
            loss.backward()
            grad = []
            for param in network.parameters():
                grad.append(param.grad.view(-1))
            grad = torch.cat(grad)
            if MC_grad == None:
                MC_grad = grad/MC_iters
            else:
                MC_grad += grad/MC_iters
        
        f = (torch.linalg.norm(MC_grad)**2).cpu().detach().item()
        var_lb_id.append(1/(f+1e-5))                
        count_id += 1
  
    var_lb_ood = []
    count_ood = 0
    MC_grad = None    

    for i, (unshaped_data, target) in enumerate(tqdm.tqdm(ood_loader)):

        x = unshaped_data.cuda()
        id_shape[0] = x.size(0)
        data = torch.zeros(tuple(id_shape), device='cuda:0')
        if data.size(1) == 3:
            if np.array_equal(id_shape, np.array(x.shape)):
                data = copy.deepcopy(x)
            else:
                n = x.size(2)
                data[:,[0],:n,:n] = x
                data[:,[1],:n,:n] = x
                data[:,[2],:n,:n] = x
        else:
            data[:,:,:,:] = x[:,[0],:28,:28]
            
        if count_ood >= num_tests:
            break

        for iteration in range(MC_iters):
            optimizer.zero_grad()
            logits = network(data.cuda())
            log_probs = F.log_softmax(logits, dim=1)
            labels = []            
            for i in range(data.shape[0]): 
                labels.append(np.random.choice(10, p=torch.exp(log_probs[i]).cpu().detach().numpy()))
            
            # F.nll_loss gives the negative log-likelihood, and expects a vector of log-probabilities            
            loss = -1*F.nll_loss(log_probs, torch.tensor(labels).cuda())
            loss.backward()
            grad = []
            for param in network.parameters():
                grad.append(param.grad.view(-1))
            grad = torch.cat(grad)
            if MC_grad == None:
                MC_grad = grad/MC_iters
            else:
                MC_grad += grad/MC_iters
                
        f = (torch.linalg.norm(MC_grad)**2).cpu().detach().item()
        var_lb_ood.append(1/(f+1e-5))
        count_ood += 1            
    
    plt.figure(dpi=300)
    sns.kdeplot( var_lb_id,
        fill=True, common_norm=False, palette="crest",
        alpha=.5, linewidth=0, label='In Distribution', bw_adjust=0.2, log_scale=[True,False]
    )
    sns.kdeplot( var_lb_ood,
        fill=True, common_norm=False, palette="crest",
        alpha=.5, linewidth=0, label='Out of Distribution', bw_adjust=0.2, log_scale=[True,False]
    )
    plt.legend() 
    plt.xlabel(r'$\mathrm{trace}(I(\theta; X^\star))^{-1}$')
    plt.title('MC Iters: ' + str(MC_iters) + '; Num Epi Tests: '   \
        + str(num_tests) + ';\nID: ' + id_dataset_name + '; OOD: ' \
        + ood_dataset_name + '; Epi. Batch Size: ' + str(batch_size_epi))            
    plt.savefig('figures/epi_hist_' + run_id + '.pdf')

    return var_lb_id, var_lb_ood

def calculate_auc(FPRs, TPRs):
    auc = 0
    for index, fpr in enumerate(FPRs):
        if index == 0:
            continue
        auc += (fpr - FPRs[index-1])*TPRs[index-1]
    return abs(auc)

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

if __name__ == "__main__":
    experiment_name = sys.argv[1]
    run_name = sys.argv[2]
    note = sys.argv[3]

    mlflow.set_tracking_uri(variant['mlflow_uri'])
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()              
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        mlflow.log_params(flatten_dict(variant))
        client.set_tag(run.info.run_id, "mlflow.note.content", note)
        epigrad_experiment(variant, run_id)