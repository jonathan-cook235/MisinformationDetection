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
        
    def forward(self, input_dim, hidden_dim,
                datum, embed_dim, output_dim):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(
                input_dim, hidden_dim), embed_dim, output_dim)
    
    def encode(self, input_dim, hidden_dim):
        return self.TGS(input_dim, hidden_dim)
    
    def decode(self, hidden_dim, embed_dim, output_dim):
        return self.veracity_prediction(hidden_dim, embed_dim, output_dim)
    
class TGS(nn.Module):
    
    def __init__(self, input_dim, hidden_dim):
        super(TGS, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim  = hidden_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.relu = nn.ReLU()
    
    
    def node_embeddings(self, datum):
        h = []
        for i in self.datum[2]:
            h.append(np.zeros_like(self.datum[2][i]))
        h_tilde = np.zeros_like(h)
        del h_tilde[0]
        h[0] = self.datum[2][0]
        for i in h_tilde:
            for j in h_tilde[i]:
                h_tilde[i][j] = np.sum(np.concatenate(h[i][j],self.datum[1][j]))
            
        def forward(self):
            hidden = self.fc1(self.datum[2])
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
    
    def compute_loss(self,output, label):
        y_1 = output[0]
        y_2 = output[1]
        loss = -label*np.log(y_2) - (1-label)*np.log(1 - y_1)
        return loss


def make_model(input_dim, hidden_dim, embed_dim, output_dim, args): #hyperparameters to be defined
    "Helper: Construct a model from hyperparameters."
    model = EncoderDecoder(
        TGS(input_dim, hidden_dim),
        veracity_prediction(hidden_dim, embed_dim, output_dim)
        )
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model