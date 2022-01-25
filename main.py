# Author: Conor Igoe
# Date: January 23 2022
# Using pretrained networks from PyTorch-Playground: 
# https://github.com/aaron-xichen/pytorch-playground

import sys
import collections
import copy
import random
import math

import matplotlib as mpl
mpl.rcParams['figure.dpi']= 300
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from mlflow.tracking import MlflowClient
import pudb
import imageio
import torch
from torch.autograd import Variable
from utee import selector
import tqdm
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from scipy.stats import entropy

from roc import ROC_test
from epistemic_tests import (epistemic_test_epigrad_originalL1, 
                            epistemic_test_grad_norm,
                            epistemic_test_batch_grad,
                            epistemic_test_GN_term,
                            epistemic_test_EG_term)
from helpers import (calculate_auc, flatten_dict)
from variant import *

def experiment(variant, run_id):

    epistemic_test_functions = [epistemic_test_EG_term]
    epistemic_test_names = ['EpiGradTerm']

    for epistemic_test_name, epistemic_test in zip(
        epistemic_test_names, epistemic_test_functions):
        ID_model_name = variant['ID_model_name']    
        OOD_model_name = variant['OOD_model_name']  
        num_tests = variant['num_tests']            
        deep = variant['deep']                      
        
        ID_model, ID_fetcher, _ = selector.select(
            ID_model_name, cuda=True)
        OOD_model, OOD_fetcher, _ = selector.select(
            OOD_model_name, cuda=True)

        ID_model.eval()

        ID_testing_loader = ID_fetcher(batch_size=1, train=False, 
            val=True)
        OOD_testing_loader = OOD_fetcher(batch_size=1, train=False, 
            val=True)

        optimiser = torch.optim.SGD(ID_model.parameters(), lr=1)
        
        # Generate Histogram
        (epigrad_id, epigrad_ood, 
        entropies_id, entropies_ood) = epistemic_test(
            id_loader=ID_testing_loader, 
            ood_loader=OOD_testing_loader, 
            id_dataset_name=ID_model_name, 
            ood_dataset_name=OOD_model_name, network=ID_model, 
            optimizer=optimiser, num_tests=num_tests, deep=deep,
            run_id=run_id)

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
                (ID_results, count_id, 
                OOD_results, count_ood) = ROC_test(
                    threshold=threshold,epigrad_id=epigrad_id,
                    epigrad_ood=epigrad_ood, flip=True)   

                FP, TP = np.sum(ID_results), np.sum(OOD_results)
                N, P = count_id, count_ood
                FPR, TPR = FP/N, TP/P
                FPRs.append(FPR)
                TPRs.append(TPR)

            # Calculate AUC
            auc = calculate_auc(FPRs, TPRs)

        mlflow.log_metric('AUC-' + epistemic_test_name + \
            ID_model_name + OOD_model_name, auc)

        plt.figure(dpi=300)
        plt.plot(FPRs, TPRs, label=epistemic_test_name)
        plt.legend(loc='lower right')
        plt.xlim([-0.05,1.05])
        plt.ylim([-0.05,1.05])
        plt.title('ROC Curve; AUC = ' + str(auc) + '\n ID = ' + \
            ID_model_name + '; OOD = ' + OOD_model_name)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.savefig('figures/epi_roc_' + epistemic_test_name + \
            '_' + run_id + '.pdf')

        mlflow.log_artifact('figures/epi_roc_' + \
            epistemic_test_name + '_' + run_id + '.pdf')

        plt.close()
        plt.figure(dpi=300)
        plt.scatter(entropies_id, epigrad_id, label='ID', s=1)
        plt.scatter(entropies_ood, epigrad_ood, label='OOD', s=1)
        plt.xlabel('Entropy')
        plt.title('ID={}; OOD={}'.format(ID_model_name, 
            OOD_model_name))
        fname = 'figures/scatter_{}_{}_{}.pdf'.format(ID_model_name, 
            OOD_model_name, run_id)
        plt.savefig(fname) 
        mlflow.log_artifact(fname)

if __name__ == "__main__":
    experiment_name = sys.argv[1]
    run_name = sys.argv[2]
    note = sys.argv[3]

    model_names = variant['model_names']

    for ID_model_name in model_names:
        for OOD_model_name in model_names:
            if OOD_model_name == ID_model_name:
                continue
            variant['ID_model_name'] = ID_model_name
            variant['OOD_model_name'] = OOD_model_name

            mlflow.set_tracking_uri(variant['mlflow_uri'])
            mlflow.set_experiment(experiment_name)
            client = MlflowClient()              
            with mlflow.start_run(run_name=run_name + '_' + \
                ID_model_name + '_' + OOD_model_name) as run:
                run_id = run.info.run_id
                mlflow.log_params(flatten_dict(variant))
                client.set_tag(
                    run.info.run_id, "mlflow.note.content", note)
                experiment(variant, run_id)
