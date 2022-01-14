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

    epistemic_test_functions = [epistemic_test0, epistemic_test1, epistemic_test2, epistemic_test3]

    for index, epistemic_test in enumerate(epistemic_test_functions):
        ID_model_name = variant['ID_model_name']    # ['mnist', 'cifar10', 'svhn']
        OOD_model_name = variant['OOD_model_name']  # ['mnist', 'cifar10', 'svhn']
        num_tests = variant['num_tests']            # [1000]

        ID_model, ID_fetcher, _ = selector.select(ID_model_name, cuda=True)
        OOD_model, OOD_fetcher, _ = selector.select(OOD_model_name, cuda=True)

        ID_testing_loader = ID_fetcher(batch_size=1, train=False, 
            val=True)
        OOD_testing_loader = OOD_fetcher(batch_size=1, train=False, 
            val=True)

        optimiser = torch.optim.SGD(ID_model.parameters(), lr=1)
        
        # Generate Histogram
        epigrad_id, epigrad_ood = epistemic_test(id_loader=ID_testing_loader, 
            ood_loader=OOD_testing_loader, id_dataset_name=ID_model_name, 
            ood_dataset_name=OOD_model_name, network=ID_model, 
            optimizer=optimiser, num_tests=num_tests)

        FPRs, TPRs = [], []
        all_points = sorted(np.concatenate((epigrad_ood,epigrad_id)))

        for threshold in tqdm.tqdm(all_points):
            ID_results, count_id, OOD_results, count_ood = ROC_test(
                threshold=threshold,epigrad_id=epigrad_id,
                epigrad_ood=epigrad_ood, flip=False)   

            FP, TP = np.sum(ID_results), np.sum(OOD_results)
            N, P = count_id, count_ood
            FPR, TPR = FP/N, TP/P
            FPRs.append(FPR)
            TPRs.append(TPR)

        # Calculate AUC
        auc = calculate_auc(FPRs, TPRs)
        if auc < 0.5:
            FPRs, TPRs = [], []            
            for threshold in tqdm.tqdm(all_points):
                ID_results, count_id, OOD_results, count_ood = ROC_test(
                    threshold=threshold,epigrad_id=epigrad_id,
                    epigrad_ood=epigrad_ood, flip=True)   

                FP, TP = np.sum(ID_results), np.sum(OOD_results)
                N, P = count_id, count_ood
                FPR, TPR = FP/N, TP/P
                FPRs.append(FPR)
                TPRs.append(TPR)

            # Calculate AUC
            auc = calculate_auc(FPRs, TPRs)


        mlflow.log_metric('AUC-' + str(index) + ID_model_name + OOD_model_name, auc)

        plt.figure(dpi=300)
        plt.plot(FPRs, TPRs, label='EpiGrad' + str(index))
        plt.legend(loc='lower right')
        plt.xlim([-0.05,1.05])
        plt.ylim([-0.05,1.05])
        plt.title('ROC Curve; AUC = ' + str(auc) + '\n ID = ' + ID_model_name + '; OOD = ' + OOD_model_name)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.savefig('figures/epi_roc_' + str(index) + '_' + run_id + '.pdf')

        mlflow.log_artifact('figures/epi_roc_' + str(index) + '_' + run_id + '.pdf')

def epistemic_test0(id_loader, ood_loader, id_dataset_name, 
    ood_dataset_name, network, optimizer, num_tests):
    # Exp, Norm, Grad
    network.eval() 
    id_shape = None

    epigrad_id = []    
    depth = 1

    for count_id, (data, target) in enumerate(tqdm.tqdm(id_loader)):

        score = 0
        
        if id_shape is None:
            id_shape = list(data.shape)
        if count_id >= num_tests:
            break

        for synthetic_label in range(10):
            optimizer.zero_grad()
            logits = network(data.cuda())
            log_probs = F.log_softmax(logits, dim=1)
            # F.nll_loss gives the negative log-likelihood, and expects a vector of log-probabilities
            loss = -1*F.nll_loss(log_probs, torch.tensor([synthetic_label]).cuda())
            loss.backward()
            grad = []
            for param in list(network.parameters())[-depth*2:]:                
                grad.append(param.grad.view(-1))
            grad = torch.cat(grad)
            norm2 = (torch.norm(grad)**2)
            score += norm2*(torch.exp(log_probs[0][synthetic_label]))
        
        epigrad_id.append(score.cpu().detach().item())
  
    epigrad_ood = []

    for count_ood, (unshaped_data, target) in enumerate(tqdm.tqdm(ood_loader)):

        score = 0

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

        for synthetic_label in range(10):
            optimizer.zero_grad()
            logits = network(data.cuda())
            log_probs = F.log_softmax(logits, dim=1)
            # F.nll_loss gives the negative log-likelihood, and expects a vector of log-probabilities
            loss = -1*F.nll_loss(log_probs, torch.tensor([synthetic_label]).cuda())
            loss.backward()
            grad = []
            for param in list(network.parameters())[-depth*2:]:                
                grad.append(param.grad.view(-1))
            grad = torch.cat(grad)
            norm2 = (torch.norm(grad)**2)
            score += norm2*(torch.exp(log_probs[0][synthetic_label]))
        
        epigrad_ood.append(score.cpu().detach().item())
    
    epigrad_id = np.array(epigrad_id)
    epigrad_ood = np.array(epigrad_ood)

    plt.figure(dpi=300)
    sns.kdeplot( epigrad_id,
        fill=True, common_norm=True, palette="crest",
        alpha=.5, linewidth=0, label='In Distribution'
    )
    sns.kdeplot( epigrad_ood,
        fill=True, common_norm=True, palette="crest",
        alpha=.5, linewidth=0, label='Out of Distribution'
    )
    plt.legend() 
    plt.xlabel('EpiGrad0 (no-inverse)')
    plt.title('Num Epi Tests: ' + str(num_tests) + \
        ';\nID: ' + id_dataset_name + '; OOD: ' \
        + ood_dataset_name)            
    plt.savefig('figures/epi_hist_0_' + run_id + '.pdf')

    mlflow.log_artifact('figures/epi_hist_0_' + run_id + '.pdf')

    return epigrad_id, epigrad_ood

def epistemic_test1(id_loader, ood_loader, id_dataset_name, 
    ood_dataset_name, network, optimizer, num_tests):
    # Norm, Exp, Grad
    network.eval() 
    id_shape = None

    epigrad_id = []    
    depth = 1

    for count_id, (data, target) in enumerate(tqdm.tqdm(id_loader)):

        score = 0
        exp_grad = torch.zeros( sum(p.numel() for p in list(network.parameters())[-depth*2:] if p.requires_grad) ).cuda()        
        if id_shape is None:
            id_shape = list(data.shape)
        if count_id >= num_tests:
            break

        for synthetic_label in range(10):
            optimizer.zero_grad()
            logits = network(data.cuda())
            log_probs = F.log_softmax(logits, dim=1)
            # F.nll_loss gives the negative log-likelihood, and expects a vector of log-probabilities
            loss = -1*F.nll_loss(log_probs, torch.tensor([synthetic_label]).cuda())
            loss.backward()
            grad = []
            for param in list(network.parameters())[-depth*2:]:                
                grad.append(param.grad.view(-1))
            grad = torch.cat(grad)
            exp_grad += grad*(torch.exp(log_probs[0][synthetic_label]))

        score += torch.norm(exp_grad)**2
        epigrad_id.append(score.cpu().detach().item())
  
    epigrad_ood = []

    for count_ood, (unshaped_data, target) in enumerate(tqdm.tqdm(ood_loader)):

        score = 0
        exp_grad = torch.zeros( sum(p.numel() for p in list(network.parameters())[-depth*2:] if p.requires_grad) ).cuda()

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

        for synthetic_label in range(10):
            optimizer.zero_grad()
            logits = network(data.cuda())
            log_probs = F.log_softmax(logits, dim=1)
            # F.nll_loss gives the negative log-likelihood, and expects a vector of log-probabilities
            loss = -1*F.nll_loss(log_probs, torch.tensor([synthetic_label]).cuda())
            loss.backward()
            grad = []
            for param in list(network.parameters())[-depth*2:]:                
                grad.append(param.grad.view(-1))
            grad = torch.cat(grad)
            exp_grad += grad*(torch.exp(log_probs[0][synthetic_label]))
        score += torch.norm(exp_grad)**2
        epigrad_ood.append(score.cpu().detach().item())
    
    epigrad_id = np.array(epigrad_id)
    epigrad_ood = np.array(epigrad_ood)

    plt.figure(dpi=300)
    sns.kdeplot( epigrad_id,
        fill=True, common_norm=True, palette="crest",
        alpha=.5, linewidth=0, label='In Distribution'
    )
    sns.kdeplot( epigrad_ood,
        fill=True, common_norm=True, palette="crest",
        alpha=.5, linewidth=0, label='Out of Distribution'
    )
    plt.legend() 
    plt.xlabel('EpiGrad1 (no-inverse)')
    plt.title('Num Epi Tests: ' + str(num_tests) + \
        ';\nID: ' + id_dataset_name + '; OOD: ' \
        + ood_dataset_name)            
    plt.savefig('figures/epi_hist_1_' + run_id + '.pdf')

    mlflow.log_artifact('figures/epi_hist_1_' + run_id + '.pdf')

    return epigrad_id, epigrad_ood

def epistemic_test2(id_loader, ood_loader, id_dataset_name, 
    ood_dataset_name, network, optimizer, num_tests):
    # Log EpiGrad
    network.eval() 
    id_shape = None

    epigrad_id = []    

    depth = 1

    for count_id, (data, target) in enumerate(tqdm.tqdm(id_loader)):

        score = 0
        
        if id_shape is None:
            id_shape = list(data.shape)
        if count_id >= num_tests:
            break

        for synthetic_label in range(10):
            optimizer.zero_grad()
            logits = network(data.cuda())
            log_probs = F.log_softmax(logits, dim=1)
            # F.nll_loss gives the negative log-likelihood, and expects a vector of log-probabilities
            loss = -1*F.nll_loss(log_probs, torch.tensor([synthetic_label]).cuda())
            loss.backward()
            grad = []
            for param in list(network.parameters())[-depth*2:]:                
                grad.append(param.grad.view(-1))
            grad = torch.cat(grad)
            norm2 = (torch.norm(grad)**2)
            score += norm2*(log_probs[0][synthetic_label])
        
        epigrad_id.append(score.cpu().detach().item())
  
    epigrad_ood = []

    for count_ood, (unshaped_data, target) in enumerate(tqdm.tqdm(ood_loader)):

        score = 0

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

        for synthetic_label in range(10):
            optimizer.zero_grad()
            logits = network(data.cuda())
            log_probs = F.log_softmax(logits, dim=1)
            # F.nll_loss gives the negative log-likelihood, and expects a vector of log-probabilities
            loss = -1*F.nll_loss(log_probs, torch.tensor([synthetic_label]).cuda())
            loss.backward()
            grad = []
            for param in list(network.parameters())[-depth*2:]:                
                grad.append(param.grad.view(-1))
            grad = torch.cat(grad)
            norm2 = (torch.norm(grad)**2)
            score += norm2*(log_probs[0][synthetic_label])
        
        epigrad_ood.append(score.cpu().detach().item())
    
    epigrad_id = np.array(epigrad_id)
    epigrad_ood = np.array(epigrad_ood)

    plt.figure(dpi=300)
    sns.kdeplot( epigrad_id,
        fill=True, common_norm=True, palette="crest",
        alpha=.5, linewidth=0, label='In Distribution'
    )
    sns.kdeplot( epigrad_ood,
        fill=True, common_norm=True, palette="crest",
        alpha=.5, linewidth=0, label='Out of Distribution'
    )
    plt.legend() 
    plt.xlabel('EpiGrad2 (no-inverse)')
    plt.title('Num Epi Tests: ' + str(num_tests) + \
        ';\nID: ' + id_dataset_name + '; OOD: ' \
        + ood_dataset_name)            
    plt.savefig('figures/epi_hist_2_' + run_id + '.pdf')

    mlflow.log_artifact('figures/epi_hist_2_' + run_id + '.pdf')

    return epigrad_id, epigrad_ood   


def epistemic_test3(id_loader, ood_loader, id_dataset_name, 
    ood_dataset_name, network, optimizer, num_tests):
    # GradNorm
    network.eval() 
    id_shape = None
    depth = 1

    epigrad_id = []    

    for count_id, (data, target) in enumerate(tqdm.tqdm(id_loader)):

        score = 0
        exp_grad = torch.zeros( sum(p.numel() for p in list(network.parameters())[-depth*2:] if p.requires_grad) ).cuda()
        
        if id_shape is None:
            id_shape = list(data.shape)
        if count_id >= num_tests:
            break

        for synthetic_label in range(10):
            optimizer.zero_grad()
            logits = network(data.cuda())
            log_probs = F.log_softmax(logits, dim=1)
            # F.nll_loss gives the negative log-likelihood, and expects a vector of log-probabilities
            loss = -1*F.nll_loss(log_probs, torch.tensor([synthetic_label]).cuda())
            loss.backward()
            grad = []
            for param in list(network.parameters())[-depth*2:]:
                grad.append(param.grad.view(-1))
            grad = torch.cat(grad)
            exp_grad += grad*(1/10)

        score += torch.norm(exp_grad,p=1)
        epigrad_id.append(score.cpu().detach().item())
  
    epigrad_ood = []

    for count_ood, (unshaped_data, target) in enumerate(tqdm.tqdm(ood_loader)):

        score = 0
        exp_grad = torch.zeros( sum(p.numel() for p in list(network.parameters())[-depth*2:] if p.requires_grad) ).cuda()

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

        for synthetic_label in range(10):
            optimizer.zero_grad()
            logits = network(data.cuda())
            log_probs = F.log_softmax(logits, dim=1)
            # F.nll_loss gives the negative log-likelihood, and expects a vector of log-probabilities
            loss = -1*F.nll_loss(log_probs, torch.tensor([synthetic_label]).cuda())
            loss.backward()
            grad = []
            for param in list(network.parameters())[-depth*2:]:
                grad.append(param.grad.view(-1))
            grad = torch.cat(grad)
            exp_grad += grad*(1/10)
        score += torch.norm(exp_grad,p=1)
        epigrad_ood.append(score.cpu().detach().item())
    
    epigrad_id = np.array(epigrad_id)
    epigrad_ood = np.array(epigrad_ood)

    plt.figure(dpi=300)
    sns.kdeplot( epigrad_id,
        fill=True, common_norm=True, palette="crest",
        alpha=.5, linewidth=0, label='In Distribution'
    )
    sns.kdeplot( epigrad_ood,
        fill=True, common_norm=True, palette="crest",
        alpha=.5, linewidth=0, label='Out of Distribution'
    )
    plt.legend() 
    plt.xlabel('GradNorm')
    plt.title('Num Epi Tests: ' + str(num_tests) + \
        ';\nID: ' + id_dataset_name + '; OOD: ' \
        + ood_dataset_name)            
    plt.savefig('figures/epi_hist_3_' + run_id + '.pdf')

    mlflow.log_artifact('figures/epi_hist_3_' + run_id + '.pdf')

    return epigrad_id, epigrad_ood


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

    model_names = ['cifar10', 'mnist', 'svhn']

    for ID_model_name in model_names:
        for OOD_model_name in model_names:
            if OOD_model_name == ID_model_name:
                continue
            variant['ID_model_name'] = ID_model_name
            variant['OOD_model_name'] = OOD_model_name

            mlflow.set_tracking_uri(variant['mlflow_uri'])
            mlflow.set_experiment(experiment_name)
            client = MlflowClient()              
            with mlflow.start_run(run_name=run_name + '_' + ID_model_name + '_' + OOD_model_name) as run:
                run_id = run.info.run_id
                mlflow.log_params(flatten_dict(variant))
                client.set_tag(run.info.run_id, "mlflow.note.content", note)
                epigrad_experiment(variant, run_id)
