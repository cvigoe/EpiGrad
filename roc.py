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
def ROC_test(threshold, epigrad_id, epigrad_ood, flip=False):
    count_id = 0
    ID_results = []
    count_ood = 0
    OOD_results = []        

    for point in epigrad_id:
        if flip:
            ID_results.append(point < threshold)
        else:
            ID_results.append(point > threshold)
        count_id += 1

    for point in epigrad_ood:
        if flip:
            OOD_results.append(point < threshold)
        else:
            OOD_results.append(point > threshold)        
        count_ood += 1

    return ID_results, count_id, OOD_results, count_ood
