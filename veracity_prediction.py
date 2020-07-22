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
        
    def forward(self,x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        soft = np.exp(output - np.max(output))
        output = soft/soft.sum()
        return output
    
    def compute_loss(self,output,veracity_label):
        y_1 = output[0]
        y_2 = output[1]
        loss = -veracity_label*np.log(y_2) - (1-veracity_label)*np.log(1 - y_1)
        return loss
