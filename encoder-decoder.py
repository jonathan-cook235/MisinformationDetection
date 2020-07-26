#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:21:24 2020

@author: jonnycook
"""

import torch
import torch.nn as nn
import numpy as np

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, TGS, veracity_prediction, dynamic_graph, hidden):
        super(EncoderDecoder, self).__init__()
        self.TGS = TGS
        self.veracity_prediction = veracity_prediction
        self.dynamic_graph = dynamic_graph
        self.hidden = hidden
        
    def forward(self, input_dim, hidden_dim, output_dim, tgt_mask, tgt_dim):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(input_dim, hidden_dim, ouput_dim), tgt_mask, tgt_dim)
    
    def encode(self, input_dim, hidden_dim, output_dim):
        return self.TGS(input_dim, hidden_dim, output_dim)
    
    def decode(self, output_dim, tgt_mask, tgt_dim):
        return self.veracity_prediction(output_dim, tgt_mask, tgt_dim)
    
class TGS(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TGS, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim  = hidden_dim
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    
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
            return relu
            
        for i in h:
            for j in h[i]:
                h[i+1][j] = forward(np.concatenate(h[i][j],h_tilde[i][j]))
                
        return h
    
    def generate_hidden(h):
        hidden = []
        for i in h[-1]:
            hidden.append(h[-1][i])
            
        return hidden
    
class veracity_prediction(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(veracity_prediction, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim  = hidden_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, 2)
        
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


def make_model(): #hyperparameters to be defined
    "Helper: Construct a model from hyperparameters."
    model = EncoderDecoder(
        TGS(),
        veracity_prediction()
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model