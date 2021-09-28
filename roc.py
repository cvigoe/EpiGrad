# Author: Conor Igoe
# Date: September 27 2021
# Using pretrained networks from PyTorch-Playground: 
# https://github.com/aaron-xichen/pytorch-playground

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

# Note: we are testing for OOD. Test returns True if 
# we are guessing OOD, returns False if guessing ID.
def ROC_test(num_tests, threshold, 
    id_loader, ood_loader, max_ent, 
    id_dataset_name, ood_dataset_name, 
    network, optimizer, batch_size_epi,
    certs):
    network.eval()
    count_id = 0
    ID_results = []
    id_shape = None
    MC_grad = None



    if optimizer is None:
        for data, target in id_loader:
            if count_id >= num_tests:
                break
            cert = certs(certs.featurise(data))
            ID_results.append(torch.linalg.norm(cert.cpu()) > threshold)
            count_id += 1

            count_ood = 0
            OOD_results = []    
        
        for unshaped_data, target in ood_loader: 
            if count_ood >= num_tests:
                break
            cert = certs(certs.featurise_OOD(data))
            OOD_results.append(torch.linalg.norm(cert.cpu()) > threshold)
            count_ood += 1

        return ID_results, count_id, OOD_results, count_ood

    for data, target in id_loader:
        if id_shape is None:
            id_shape = list(data.shape)
        if count_id >= num_tests:
            break
        fisher_samples = []
        for iteration in range(MC_ITERS):
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
                MC_grad = grad/MC_ITERS
            else:
                MC_grad += grad/MC_ITERS
                
        epi = (torch.linalg.norm(MC_grad)**2).cpu().detach().item()
        ID_results.append(epi > threshold)
        count_id += 1

    count_ood = 0
    OOD_results = []    
    MC_grad = None    
    for unshaped_data, target in ood_loader: 
        id_shape[0] = unshaped_data.size(0)
        data = torch.zeros(tuple(id_shape))
        if data.size(1) == 3:
            data[:,[0],:28,:28] = unshaped_data
            data[:,[1],:28,:28] = unshaped_data
            data[:,[2],:28,:28] = unshaped_data
        else:
            data[:,:,:] = unshaped_data[:,[0],:28,:28]
            
        if count_ood >= num_tests:
            break
        fisher_samples = []
        for iteration in range(MC_ITERS):
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
                MC_grad = grad/MC_ITERS
            else:
                MC_grad += grad/MC_ITERS
                
        epi = (torch.linalg.norm(MC_grad)**2).cpu().detach().item()     
        OOD_results.append(epi > threshold)
        count_ood += 1
    return ID_results, count_id, OOD_results, count_ood