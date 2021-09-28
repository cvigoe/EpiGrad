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

def experiment(variant):
    # Train Certs
    ID_model_name = variant['ID_model_name']    # ['mnist', 'cifar10', 'svhn', 'fashion']
    OOD_model_name = variant['OOD_model_name']  # ['mnist', 'cifar10', 'svhn', 'fashion']
    k = variant['k']                            # [1e0, 1e1, 1e2, 1e3, 1e4]
    lr = variant['lr']                          # [1e-4]
    lmbda = variant['lmbda']                    # [1e-6, 1e-4, 1e-2, 1e0, 1e2]
    epochs = variant['epochs']                  # [10]
    batch_size = variant['batch_size']          # [64]
    batch_size_epi = variant['batch_size_epi']  # [1, 64]
    num_tests = variant['num_tests']            # [1000]
    max_ent = variant['max_ent']                # [False]
    ROC_lower = variant['ROC_lower']            
    ROC_upper = variant['ROC_upper']
    num_ROC = variant['num_ROC']

    ID_model, ID_fetcher, _ = selector.select(ID_model_name, cuda=True)
    OOD_model, OOD_fetcher, _ = selector.select(OOD_model_name, cuda=True)

    cert_training_loader, _ = ID_fetcher(batch_size=batch_size, train=True, 
        val=True)
    ID_testing_loader = ID_fetcher(batch_size=1, train=False, 
        val=True)
    OOD_testing_loader = OOD_fetcher(batch_size=1, train=False, 
        val=True)

    ortho = Ortho(k, epochs, lr, lmbda, ID_model)
    losses = ortho.train(cert_training_loader)

    # Generate Loss Curve
    plt.figure(dpi=300)
    plt.plot(losses)
    plt.xlabel('Training Iteration')
    plt.ylabel('Orthonormal Certificates Loss')
    plt.title('Orthonormal Certificates Training Loss')
    plt.savefig('training_loss.pdf')

    # Generate Histogram
    ID_certificates = []
    for x, _ in tqdm.tqdm(ID_testing_loader):
        id_shape = np.array(x.shape)
        f = ortho.featurise(x, ID_model)
        certs = ortho(f)
        for i in certs.pow(2).mean(axis=1).detach():
            ID_certificates.append(i.item()) 

    OOD_certificates = []
    for x, _ in tqdm.tqdm(OOD_testing_loader):
        f = ortho.featurise_OOD(x, OOD_model)
        certs = ortho(f)
        for i in certs.pow(2).mean(axis=1).detach():
            OOD_certificates.append(i.item())     

    pudb.set_trace()

    # Generate ROC Curves
    FPRs = []
    TPRs = []
    for threshold in tqdm.tqdm(np.linspace(ROC_lower, ROC_upper, num_ROC)):
        ID_results, count_id, OOD_results, count_ood = ROC_test(
            num_tests=num_tests, threshold=threshold, 
            id_loader=ID_testing_loader, 
            ood_loader=OOD_testing_loader, max_ent=False, 
            id_dataset_name=ID_model_name, 
            ood_dataset_name=OOD_model_name,
            network=ID_model, optimizer=None, 
            batch_size_epi=batch_size_epi,
            certs=ortho)

        FP = np.sum(ID_results)
        N = count_id
        TP = np.sum(OOD_results)
        P = count_ood
        FPR = FP/N
        TPR = TP/P
        FPRs.append(FPR)
        TPRs.append(TPR)

    plt.figure(dpi=300)
    plt.plot(FPRs, TPRs, '-o')
    plt.savefig('ROC.pdf')

    # Calculate AUC

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
        mlflow.log_params(flatten_dict(variant))
        client.set_tag(run.info.run_id, "mlflow.note.content", note)
        experiment(variant)


