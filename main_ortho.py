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

def ortho_experiment(variant, run_id):
    ID_model_name = variant['ID_model_name']    # ['mnist', 'cifar10', 'svhn']
    OOD_model_name = variant['OOD_model_name']  # ['mnist', 'cifar10', 'svhn']
    k = variant['k']                            # [1e0, 1e1, 1e2, 1e3, 1e4]
    lr = variant['lr']                          # [1e-4]
    lmbda = variant['lmbda']                    # [1e-6, 1e-4, 1e-2, 1e0, 1e2]
    epochs = variant['epochs']                  # [10]
    batch_size = variant['batch_size']          # [64]
    batch_size_epi = variant['batch_size_epi']  # [1, 64]
    num_tests = variant['num_tests']            # [1000]
    max_ent = variant['max_ent']                # [False]            
    num_ROC = variant['num_ROC']                # [20]
    MC_iters = variant['MC_iters']              # [1000]

    ID_model, ID_fetcher, _ = selector.select(ID_model_name, cuda=True)
    OOD_model, OOD_fetcher, _ = selector.select(OOD_model_name, cuda=True)

    cert_training_loader, _ = ID_fetcher(batch_size=batch_size, train=True, 
        val=True)
    ID_testing_loader = ID_fetcher(batch_size=1, train=False, 
        val=True)
    OOD_testing_loader = OOD_fetcher(batch_size=1, train=False, 
        val=True)

    # Train Certs
    ortho = Ortho(k, epochs, lr, lmbda, ID_model)
    _ = ortho.train(cert_training_loader)

    # Generate Histogram
    ID_certificates = []
    for x, y in tqdm.tqdm(ID_testing_loader):
        f = ortho.featurise(x)
        certs = ortho(f)
        ID_certificates.append(torch.linalg.norm(certs.detach().cpu()).pow(2).item())

    OOD_certificates = []
    for x, y in tqdm.tqdm(OOD_testing_loader):
        f = ortho.featurise_OOD(x)
        certs = ortho(f)
        OOD_certificates.append(torch.linalg.norm(certs.detach().cpu()).pow(2).item())

    plt.figure(dpi=300)
    sns.kdeplot( ID_certificates,
        fill=True, common_norm=False, palette="crest",
        alpha=.5, linewidth=0, label='In Distribution', bw_adjust=0.2, log_scale=[True,False]
    )
    sns.kdeplot( OOD_certificates,
        fill=True, common_norm=False, palette="crest",
        alpha=.5, linewidth=0, 
                label='Out of Distribution', bw_adjust=0.2, log_scale=[True,False]
    )
    plt.legend() 
    plt.xlabel(r'$ || C^T \phi(X) || _2^2 $')
    plt.title('OC, k = ' + str(k) + r'; $\lambda = $' + str(lmbda))
    plt.savefig('figures/hist_' + run_id + '.pdf')

    # Generate ROC Curves
    ROC_lower = np.log10(min(np.percentile(ID_certificates,1) , np.percentile(OOD_certificates,1)))
    ROC_upper = np.log10(max(np.percentile(ID_certificates,99) , np.percentile(OOD_certificates,99)))

    FPRs, TPRs = [], []
    for threshold in tqdm.tqdm(np.logspace(ROC_lower, ROC_upper, num_ROC)):
        ID_results, count_id, OOD_results, count_ood = ROC_test(
            num_tests=num_tests, threshold=threshold, 
            id_loader=ID_testing_loader, 
            ood_loader=OOD_testing_loader, max_ent=False, 
            id_dataset_name=ID_model_name, 
            ood_dataset_name=OOD_model_name,
            network=ID_model, optimizer=None, 
            batch_size_epi=batch_size_epi,
            certs=ortho, MC_iters=MC_iters)

        FP, TP = np.sum(ID_results), np.sum(OOD_results)
        N, P = count_id, count_ood
        FPR, TPR = FP/N, TP/P
        FPRs.append(FPR)
        TPRs.append(TPR)

    np.savetxt("data/FPRs_" + run_id + ".csv", FPRs, delimiter=",")
    np.savetxt("data/TPRs_" + run_id + ".csv", TPRs, delimiter=",")

    # Calculate AUC
    auc = calculate_auc(FPRs, TPRs)
    with open('data/AUCs.csv','a') as fd:
        fd.write(run_id + ', ' + str(auc) + '\n')

    plt.figure(dpi=300)
    plt.plot(FPRs, TPRs, '-o', label='OC, k = ' + str(k) + r'; $\lambda = $' + str(lmbda))
    plt.legend(loc='lower right')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.title('ROC Curve; AUC = ' + str(auc) + '\n ID = ' + ID_model_name + '; OOD = ' + OOD_model_name)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('figures/roc_' + run_id + '.pdf')

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

    lmbdas = [1e-4, 1e-2, 1e0, 1e2, 1e4]
    ks = [1e0, 1e1, 1e2, 1e3, 1e4]

    for k in ks:
        for lmbda in lmbdas:            
            variant['k'] = k
            variant['lmbda'] = lmbda
            mlflow.set_tracking_uri(variant['mlflow_uri'])
            mlflow.set_experiment(experiment_name)
            client = MlflowClient()              
            with mlflow.start_run(run_name=run_name) as run:
                run_id = run.info.run_id
                mlflow.log_params(flatten_dict(variant))
                client.set_tag(run.info.run_id, "mlflow.note.content", note)
                ortho_experiment(variant, run_id)
