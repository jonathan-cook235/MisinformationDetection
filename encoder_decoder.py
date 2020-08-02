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
    def __init__(self, TGS, veracity_prediction):
        super(EncoderDecoder, self).__init__()
        self.TGS = TGS
        self.veracity_prediction = veracity_prediction
        
    def forward(self, data):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(data))
    
    def encode(self, data):
        return self.TGS(data)
    
    def decode(self, hidden):
        return self.veracity_prediction(hidden)
    
class TGS(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, args):
        super(TGS, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim  = hidden_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.relu = nn.ReLU()
    
    
    def forward(self, data):
        
        h = []
        for i in data:
            h.append(np.zeros_like(data[i]))
        h_tilde = np.zeros_like(h)
        del h_tilde[0]
        h[0] = data[0]
        for i in h_tilde:
            for j in h_tilde[i]:
                h_tilde[i][j] = np.sum(np.concatenate(h[i][j],data[-1][j]))
  
        for i in h:
            for j in h[i]:
                x = np.concatenate(h[i][j],h_tilde[i][j])
                hidden = self.fc1(x)
                relu = self.relu(hidden)
                h[i+1][j] = relu
    
        hidden = []
        for i in h[-1]:
            hidden.append(h[-1][i])
            
        return hidden
    
class veracity_prediction(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(veracity_prediction, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim  = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, output_dim)
        
    def forward(self, hidden):
        embed = self.fc1(hidden)
        relu = self.relu(embed)
        output = self.fc2(relu)
        soft = np.exp(output - np.max(output))
        output = soft/soft.sum()
        return output


def make_model(input_dim, hidden_dim, embed_dim, output_dim, args): #hyperparameters to be defined
    "Helper: Construct a model from hyperparameters."
    model = EncoderDecoder(
        TGS(input_dim, hidden_dim, args),
        veracity_prediction(hidden_dim, embed_dim, output_dim, args)
        )
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model