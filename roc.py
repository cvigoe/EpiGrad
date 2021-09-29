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
    certs, MC_iters):
    network.eval()
    count_id = 0
    ID_results = []
    count_ood = 0
    OOD_results = []        
    id_shape = None
    MC_grad = None

    if optimizer is None:
        for data, target in id_loader:
            if count_id >= num_tests:
                break
            cert = certs(certs.featurise(data))
            ID_results.append(torch.linalg.norm(cert.cpu()).pow(2) > threshold)
            count_id += 1
        
        for unshaped_data, target in ood_loader: 
            if count_ood >= num_tests:
                break
            cert = certs(certs.featurise_OOD(unshaped_data))
            OOD_results.append(torch.linalg.norm(cert.cpu()).pow(2) > threshold)
            count_ood += 1

        return ID_results, count_id, OOD_results, count_ood

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

        epi = (torch.linalg.norm(MC_grad)**2).cpu().detach().item()
        ID_results.append(epi > threshold)
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
        
        epi = (torch.linalg.norm(MC_grad)**2).cpu().detach().item()     
        OOD_results.append(epi > threshold)
        count_ood += 1 

    return ID_results, count_id, OOD_results, count_ood
