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

from helpers import (calculate_auc, flatten_dict)

def epistemic_test_batch_grad(id_loader, ood_loader, id_dataset_name, 
    ood_dataset_name, network, optimizer, num_tests, deep, run_id):
    # Original EpiGrad with L1 nrom and gradients additionally taken 
    # w.r.t. ID datapoints
    network.eval() 
    id_shape = None

    epigrad_id = []    
    entropies_id = []
    depth = 1

    prototype_features = []
    prototype_labels = []

    for synthetic_label in range(10):
        for data, target in id_loader:
            if target.item() == synthetic_label:
                prototype_features.append(data.cuda())
                prototype_labels.append(synthetic_label)
                break

    for count_id, (data, target) in enumerate(tqdm.tqdm(id_loader)):

        score = 0
        
        if id_shape is None:
            id_shape = list(data.shape)
        if count_id >= num_tests:
            break

        for synthetic_label in range(10):
            optimizer.zero_grad()
            loss = 0
            for prototype_feature, prototype_label in zip(
                prototype_features, prototype_labels):
                logits = network(prototype_feature)
                log_probs = F.log_softmax(logits, dim=1)
                # F.nll_loss gives the negative log-likelihood, and 
                # expects a vector of log-probabilities
                loss += -1*F.nll_loss(
                    log_probs, torch.tensor([prototype_label]).cuda())

            logits = network(data.cuda())
            log_probs = F.log_softmax(logits, dim=1)
            # F.nll_loss gives the negative log-likelihood, and 
            # expects a vector of log-probabilities
            loss += -1*F.nll_loss(
                log_probs, torch.tensor([synthetic_label]).cuda())
            loss.backward()
            grad = []
            if deep:
                for param in network.parameters():
                    grad.append(param.grad.view(-1))
            else:
                for param in list(network.parameters())[-depth*2:]:                
                    grad.append(param.grad.view(-1))
            grad = torch.cat(grad)
            norm2 = (torch.norm(grad,p=1))
            score += norm2*(torch.exp(log_probs[0][synthetic_label]))

        entropy = -1 * log_probs @ torch.exp(log_probs).T
        entropies_id.append(entropy.cpu().detach().item()) 
        epigrad_id.append(score.cpu().detach().item())
  
    epigrad_ood = []
    entropies_ood = []

    for count_ood, (unshaped_data, target) in enumerate(
        tqdm.tqdm(ood_loader)):

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
            loss = 0
            for prototype_feature, prototype_label in zip(
                prototype_features, prototype_labels):
                logits = network(prototype_feature)
                log_probs = F.log_softmax(logits, dim=1)
                # F.nll_loss gives the negative log-likelihood, and 
                # expects a vector of log-probabilities
                loss += -1*F.nll_loss(
                    log_probs, torch.tensor([prototype_label]).cuda())

            logits = network(data.cuda())
            log_probs = F.log_softmax(logits, dim=1)
            # F.nll_loss gives the negative log-likelihood, and 
            # expects a vector of log-probabilities
            loss += -1*F.nll_loss(
                log_probs, torch.tensor([synthetic_label]).cuda())
            loss.backward()
            grad = []
            if deep:
                for param in network.parameters():
                    grad.append(param.grad.view(-1))
            else:
                for param in list(network.parameters())[-depth*2:]:                
                    grad.append(param.grad.view(-1))
            grad = torch.cat(grad)
            norm2 = (torch.norm(grad,p=1))
            score += norm2*(torch.exp(log_probs[0][synthetic_label]))
        
        entropy = -1 * log_probs @ torch.exp(log_probs).T
        entropies_ood.append(entropy.cpu().detach().item())
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

    return epigrad_id, epigrad_ood, entropies_id, entropies_ood

def epistemic_test_epigrad_originalL1(id_loader, 
    ood_loader, id_dataset_name, ood_dataset_name, network, optimizer, 
    num_tests, deep, run_id):
    # Original EpiGrad, with L1 norm
    network.eval() 
    id_shape = None

    epigrad_id = []    
    entropies_id = []
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
            # F.nll_loss gives the negative log-likelihood, and 
            # expects a vector of log-probabilities
            loss = -1*F.nll_loss(
                log_probs, torch.tensor([synthetic_label]).cuda())
            loss.backward()
            grad = []
            if deep:
                for param in network.parameters():
                    grad.append(param.grad.view(-1))
            else:
                for param in list(network.parameters())[-depth*2:]:                
                    grad.append(param.grad.view(-1))
            grad = torch.cat(grad)
            norm2 = (torch.norm(grad,p=1))
            score += norm2*(torch.exp(log_probs[0][synthetic_label]))

        entropy = -1 * log_probs @ torch.exp(log_probs).T
        entropies_id.append(entropy.cpu().detach().item()) 
        epigrad_id.append(score.cpu().detach().item())
  
    epigrad_ood = []
    entropies_ood = []

    for count_ood, (unshaped_data, target) in enumerate(
        tqdm.tqdm(ood_loader)):

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
            # F.nll_loss gives the negative log-likelihood, and 
            # expects a vector of log-probabilities
            loss = -1*F.nll_loss(
                log_probs, torch.tensor([synthetic_label]).cuda())
            loss.backward()
            grad = []
            if deep:
                for param in network.parameters():
                    grad.append(param.grad.view(-1))
            else:
                for param in list(network.parameters())[-depth*2:]:                
                    grad.append(param.grad.view(-1))
            grad = torch.cat(grad)
            norm2 = (torch.norm(grad,p=1))
            score += norm2*(torch.exp(log_probs[0][synthetic_label]))
        
        entropy = -1 * log_probs @ torch.exp(log_probs).T
        entropies_ood.append(entropy.cpu().detach().item())
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

    return epigrad_id, epigrad_ood, entropies_id, entropies_ood

def epistemic_test1(id_loader, ood_loader, id_dataset_name, 
    ood_dataset_name, network, optimizer, num_tests, deep, run_id):
    # Norm, Exp, Grad
    network.eval() 
    id_shape = None

    epigrad_id = []    
    depth = 1

    for count_id, (data, target) in enumerate(tqdm.tqdm(id_loader)):

        score = 0
        if deep:
            exp_grad = torch.zeros( 
                sum(p.numel() for p in \
                    network.parameters() if p.requires_grad) ).cuda()        
        else:
            exp_grad = torch.zeros( sum(
                p.numel() for p in \
                list(network.parameters())[-depth*2:] 
                if p.requires_grad) ).cuda()
        if id_shape is None:
            id_shape = list(data.shape)
        if count_id >= num_tests:
            break

        for synthetic_label in range(10):
            optimizer.zero_grad()
            logits = network(data.cuda())
            log_probs = F.log_softmax(logits, dim=1)
            # F.nll_loss gives the negative log-likelihood, and 
            # expects a vector of log-probabilities
            loss = -1*F.nll_loss(
                log_probs, torch.tensor([synthetic_label]).cuda())
            loss.backward()
            grad = []
            if deep:
                for param in network.parameters():
                    grad.append(param.grad.view(-1))
            else:
                for param in list(network.parameters())[-depth*2:]:                
                    grad.append(param.grad.view(-1))
            grad = torch.cat(grad)
            exp_grad += grad*(
                torch.exp(log_probs[0][synthetic_label]))

        score += torch.norm(exp_grad)**2
        epigrad_id.append(score.cpu().detach().item())
  
    epigrad_ood = []

    for count_ood, (unshaped_data, target) in enumerate(
        tqdm.tqdm(ood_loader)):

        score = 0
        if deep:
            exp_grad = torch.zeros( sum(p.numel() for p in \
                network.parameters() if p.requires_grad) ).cuda()        
        else:
            exp_grad = torch.zeros( sum(p.numel() for p in \
                list(network.parameters())[-depth*2:] 
                    if p.requires_grad) ).cuda()
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
            # F.nll_loss gives the negative log-likelihood, and 
            # expects a vector of log-probabilities
            loss = -1*F.nll_loss(
                log_probs, torch.tensor([synthetic_label]).cuda())
            loss.backward()
            grad = []
            if deep:
                for param in network.parameters():
                    grad.append(param.grad.view(-1))
            else:
                for param in list(network.parameters())[-depth*2:]:                
                    grad.append(param.grad.view(-1))
            grad = torch.cat(grad)
            exp_grad += grad*(
                torch.exp(log_probs[0][synthetic_label]))
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
    ood_dataset_name, network, optimizer, num_tests, deep, run_id):
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
            # F.nll_loss gives the negative log-likelihood, and 
            # expects a vector of log-probabilities
            loss = -1*F.nll_loss(
                log_probs, torch.tensor([synthetic_label]).cuda())
            loss.backward()
            grad = []
            if deep:
                for param in network.parameters():
                    grad.append(param.grad.view(-1))
            else:
                for param in list(network.parameters())[-depth*2:]:                
                    grad.append(param.grad.view(-1))
            grad = torch.cat(grad)
            norm2 = (torch.norm(grad)**2)
            score += norm2*(log_probs[0][synthetic_label])
        
        epigrad_id.append(score.cpu().detach().item())
  
    epigrad_ood = []

    for count_ood, (unshaped_data, target) in \
        enumerate(tqdm.tqdm(ood_loader)):

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
            # F.nll_loss gives the negative log-likelihood, and 
            # expects a vector of log-probabilities
            loss = -1*F.nll_loss(
                log_probs, torch.tensor([synthetic_label]).cuda())
            loss.backward()
            grad = []
            if deep:
                for param in network.parameters():
                    grad.append(param.grad.view(-1))
            else:
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

def epistemic_test_grad_norm(id_loader, ood_loader, id_dataset_name, 
    ood_dataset_name, network, optimizer, num_tests, deep, run_id):
    # GradNorm
    network.eval() 
    id_shape = None
    depth = 1

    epigrad_id = []    
    entropies_id = []

    for count_id, (data, target) in enumerate(tqdm.tqdm(id_loader)):

        score = 0
        if deep:
            exp_grad = torch.zeros( sum(p.numel() for p in \
                network.parameters() if p.requires_grad) ).cuda()        
        else:
            exp_grad = torch.zeros( sum(p.numel() for p in \
                list(network.parameters())[-depth*2:] 
                    if p.requires_grad) ).cuda()
        
        if id_shape is None:
            id_shape = list(data.shape)
        if count_id >= num_tests:
            break

        for synthetic_label in range(10):
            optimizer.zero_grad()
            logits = network(data.cuda())
            log_probs = F.log_softmax(logits, dim=1)
            # F.nll_loss gives the negative log-likelihood, and 
            # expects a vector of log-probabilities
            loss = -1*F.nll_loss(
                log_probs, torch.tensor([synthetic_label]).cuda())
            loss.backward()
            grad = []
            if deep:
                for param in network.parameters():
                    grad.append(param.grad.view(-1))
            else:
                for param in list(network.parameters())[-depth*2:]:                
                    grad.append(param.grad.view(-1))
            grad = torch.cat(grad)
            exp_grad += grad*(1/10)

        score += torch.norm(exp_grad,p=1)
        epigrad_id.append(score.cpu().detach().item())
        entropy = -1 * log_probs @ torch.exp(log_probs).T
        entropies_id.append(entropy.cpu().detach().item()) 
  
    epigrad_ood = []
    entropies_ood = []

    for count_ood, (unshaped_data, target) in enumerate(
        tqdm.tqdm(ood_loader)):

        score = 0
        if deep:
            exp_grad = torch.zeros( sum(p.numel() for p in \
                network.parameters() if p.requires_grad) ).cuda()        
        else:
            exp_grad = torch.zeros( sum(p.numel() for p in \
                list(network.parameters())[-depth*2:] 
                    if p.requires_grad) ).cuda()
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
            # F.nll_loss gives the negative log-likelihood, and 
            # expects a vector of log-probabilities
            loss = -1*F.nll_loss(
                log_probs, torch.tensor([synthetic_label]).cuda())
            loss.backward()
            grad = []
            if deep:
                for param in network.parameters():
                    grad.append(param.grad.view(-1))
            else:
                for param in list(network.parameters())[-depth*2:]:                
                    grad.append(param.grad.view(-1))
            grad = torch.cat(grad)
            exp_grad += grad*(1/10)
        score += torch.norm(exp_grad,p=1)
        epigrad_ood.append(score.cpu().detach().item())
        entropy = -1 * log_probs @ torch.exp(log_probs).T
        entropies_ood.append(entropy.cpu().detach().item())         
    
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

    return epigrad_id, epigrad_ood, entropies_id, entropies_ood