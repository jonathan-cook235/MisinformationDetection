#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 11:56:59 2020

@author: jonnycook
"""

import numpy as np
import torch

import graphs
import temporal_graph_sum

dynamic_graph = graphs.generate_dynamic_graph.dynamic_graph

class veracity(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(veracity, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim  = hidden_dim
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.weight = torch.nn.Parameter(data=torch.randn(len(dynamic_graph[-1])), requires_grad=True)
        self.bias = torch.nn.Parameter(data=torch.Tensor(0,0), requires_grad=False)
        
    def forward(self,x):
        hidden = x*self.weight + self.bias
        relu = self.relu(hidden)
        soft = np.exp(relu - np.max(relu))
        output = soft/soft.sum()
        return output
    
    def compute_loss(self,output,veracity_label):
        y_1 = output[0]
        y_2 = output[1]
        loss = -veracity_label*np.log(y_2) - (1-veracity_label)*np.log(1 - y_1)
        return loss
