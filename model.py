# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 22:20:08 2020

@author: Prudhvinath
"""

import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    
    def __init__(self, inputDim, outputDim):
        super(Model, self).__init__()
        
        self.layer1 = nn.Linear(inputDim, 128)
        self.layer2 = nn.Linear(128, 128)
        
        self.actorHead = nn.Linear(128, outputDim)
        
        self.criticHead = nn.Linear(128, 1)
        
        self.savedActions = []
        self.rewards = []
        
    def forward(self, state):
        
        state = F.relu(self.layer1(state))
        state = F.relu(self.layer2(state))
        
        actionProbs = F.softmax(self.actorHead(state), dim = -1)
        
        stateValues = self.criticHead(state)
        
        return actionProbs, stateValues
    
    