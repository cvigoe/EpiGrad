# Author: Conor Igoe
# Date: September 27 2021
# Using pretrained networks from PyTorch-Playground: 
# https://github.com/aaron-xichen/pytorch-playground

import torch
from torch.autograd import Variable
from utee import selector

import tqdm
import copy
import pudb

import torch.nn as nn
import numpy as np
import random
import math
import torch.optim as optim
import torch.nn.functional as F

import mlflow

class Ortho():
    def __init__(self, k, epochs, lr, lmbda, model):
        self.k = int(k)
        self.epochs = epochs
        self.lr = lr
        self.lmbda = lmbda
        self.model = model
        self.model.eval()
        self.feature_dim = None
        self.id_shape = None

    def __call__(self, x):
        return self.certs(x.cuda())

    def train(self, train_data_loader):
        self.id_shape = np.array(next(enumerate(train_data_loader))[1][0].shape)
        self.feature_dim = self.get_feature_dim(train_data_loader)
        self.certs = torch.nn.Linear(self.feature_dim, self.k, 
            bias=False, device='cuda:0')
        self.opt = torch.optim.Adam(self.certs.parameters(), 
            lr=self.lr)
        self.loss = torch.nn.MSELoss()
        losses = []
        for epoch in tqdm.tqdm(range(self.epochs)):
            for x, y in train_data_loader:
                self.opt.zero_grad()
                f = self.featurise(x)
                error = self.loss(self.certs(f), self.target(f))
                penalty = self.lmbda * \
                    (self.certs.weight @ self.certs.weight.t() - 
                     torch.eye(self.k, device='cuda:0')).pow(2).mean()
                ell = error + penalty
                ell.backward()
                self.opt.step()
                mlflow.log_metric('loss', ell.detach().cpu().item())
                losses.append(ell.detach().cpu().item())

        return losses

    def featurise(self, x):
        batch_size = x.shape[0]
        if 'features' in dir(self.model):
            f = self.model.features(x.cuda())
            return torch.reshape(f, [batch_size,-1])
        # Apparently need to squeeze input tensor to
        # remove redundant channel dimension when 
        # manually passing through layers...
        x = torch.reshape(x.cuda(), [batch_size,-1])
        num_layers = len(self.model.model)
        for layer_idx in range(num_layers-1):
            x = self.model.model[layer_idx](x)
        return x

    def featurise_OOD(self, x):
        x = x.cuda()
        self.id_shape[0] = x.size(0)
        # if np.array_equal(self.id_shape, np.array(x.shape)):
        #     return x
        data = torch.zeros(tuple(self.id_shape), device='cuda:0')
        if data.size(1) == 3:
            n = x.size(2)
            data[:,[0],:n,:n] = x
            data[:,[1],:n,:n] = x
            data[:,[2],:n,:n] = x
        else:
            data[:,:,:,:] = x[:,[0],:28,:28]
        return self.featurise(data)

    def target(self, x):
        return torch.zeros(x.size(0), self.k, device='cuda:0')

    def get_feature_dim(self, train_data_loader):
        x = next(enumerate(train_data_loader))[1][0].cuda()
        batch_size = x.shape[0]
        if 'features' in dir(self.model):
            f = self.model.features(x.cuda())
            return torch.reshape(f, [batch_size,-1]).shape[1]
        else:
            x = torch.reshape(x, [batch_size,-1])
            num_layers = len(self.model.model)
            for layer_idx in range(num_layers-1):
                x = self.model.model[layer_idx](x)             
            return torch.reshape(x, [batch_size,-1]).shape[1]
