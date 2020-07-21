#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:31:31 2020

@author: jonnycook
"""

import graphs

import numpy as np
import torch

dynamic_graph = graphs.generate_dynamic_graph.dynamic_graph
edges = graphs.generate_dynamic_graph.edges
datum = graphs.generate_datum.datum

nodes = []

for i in dynamic_graph:
    for j in dynamic_graph[i]:
        for k in dynamic_graph[i][j]:
            if dynamic_graph[i][j][k] not in nodes:
                nodes.append(dynamic_graph[i][j][k])
                
num_nodes = len(nodes)

class TGS_stack(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TGS_stack, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim  = hidden_dim
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
    
    
    def node_embeddings(dynamic_graph):
        h = []
        for i in dynamic_graph:
            h.append(np.zeros_like(dynamic_graph[i]))
        h_tilde = np.zeros_like(h)
        del h_tilde[0]
        h[0] = dynamic_graph[0]
        for i in h_tilde:
            for j in h_tilde[i]:
                h_tilde[i][j] = np.sum(np.concatenate(h[i][j],edges[j]))
            
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            output = self.sigmoid(output)
            return output
            
        for i in h:
            for j in h[i]:
                h[i+1][j] = forward(np.concatenate(h[i][j],h_tilde[i][j]))
                
        return h
    
    def generate_hidden(h):
        hidden = []
        for i in h[-1]:
            hidden.append(h[-1][i])
            
        return hidden
        
        

    
    