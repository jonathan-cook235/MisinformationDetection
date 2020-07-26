#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:21:24 2020

@author: jonnycook
"""

import torch
import torch.nn as nn
import numpy as np

import graphs

labels = graphs.get_labels(dumps_dir="rumor_detection_acl2017", dataset="twitter15")
datum = graphs.generate_datum
edges = datum[1]
dynamic_graph = datum[2]

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, TGS, veracity_prediction, dynamic_graph):
        super(EncoderDecoder, self).__init__()
        self.TGS = TGS
        self.veracity_prediction = veracity_prediction
        self.dynamic_graph = dynamic_graph
        
    def forward(self, input_dim, hidden_dim, output_dim, dynamic_graph, edges,
                tgt_mask, tgt_dim, hidden):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(
                input_dim, hidden_dim, output_dim, dynamic_graph, edges), tgt_mask, tgt_dim, hidden)
    
    def encode(self, input_dim, hidden_dim, output_dim, dynamic_graph, edges):
        return self.TGS(input_dim, hidden_dim, output_dim, dynamic_graph, edges)
    
    def decode(self, output_dim, tgt_mask, tgt_dim, hidden):
        return self.veracity_prediction(output_dim, tgt_mask, tgt_dim, hidden)
    
class TGS(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, dynamic_graph, edges):
        super(TGS, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim  = hidden_dim
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.dynamic_graph = dynamic_graph
        self.edges = edges
    
    
    def node_embeddings(self):
        h = []
        for i in self.dynamic_graph:
            h.append(np.zeros_like(self.dynamic_graph[i]))
        h_tilde = np.zeros_like(h)
        del h_tilde[0]
        h[0] = self.dynamic_graph[0]
        for i in h_tilde:
            for j in h_tilde[i]:
                h_tilde[i][j] = np.sum(np.concatenate(h[i][j],self.edges[j]))
            
        def forward(self):
            hidden = self.fc1(self.dynamic_graph)
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
    
    def __init__(self, input_dim, hidden_dim, output_dim, hidden):
        super(veracity_prediction, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim  = hidden_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, 2)
        self.hidden = hidden
        
    def forward(self):
        embed = self.fc1(self.hidden)
        relu = self.relu(embed)
        output = self.fc2(relu)
        soft = np.exp(output - np.max(output))
        output = soft/soft.sum()
        return output
    
    def compute_loss(self,output,labels):
        veracity_label = #?
        y_1 = output[0]
        y_2 = output[1]
        loss = -veracity_label*np.log(y_2) - (1-veracity_label)*np.log(1 - y_1)
        return loss


def make_model(): #hyperparameters to be defined
    "Helper: Construct a model from hyperparameters."
    model = EncoderDecoder(
        TGS(input_dim, hidden_dim, output_dim),
        veracity_prediction(output_dim, tgt_mask, tgt_dim),)
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model