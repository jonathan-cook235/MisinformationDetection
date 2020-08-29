#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:21:24 2020

@author: jonnycook
"""

import torch
import torch.nn as nn
import numpy as np

from TGS_utils import get_neighbor_finder, NeighborFinder
from embedding_module import get_embedding_module
from time_encoding import TimeEncode

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. 
    
    """
    def __init__(self, TGS, veracity_prediction):
        super(EncoderDecoder, self).__init__()
        self.TGS = TGS
        self.veracity_prediction = veracity_prediction
        
    def forward(self, data):
        return self.decode(self.encode(data))
    
    def encode(self, data):
        return self.TGS(data)
    
    def decode(self, source_embedding):
        return self.veracity_prediction(source_embedding)
    
class TGS(nn.Module):
    
    def __init__(self, input_dim, embedding_dimension):
        """
        Temporal Graph Sum encoder

        """
        super(TGS, self).__init__()
        self.memory = None
        self.embedding_module = get_embedding_module
        
    def forward(self, data):
        node_features = data.x
        timestamps = data.t
        # print('encoder-decoder', type(timestamps))
        edge_features = data.edge_index
        neighbor_finder = get_neighbor_finder(data)
        n_node_features = node_features.shape[1]
        n_nodes = node_features.shape[0]
        n_edge_features = edge_features.shape[1]
        time_encoder = TimeEncode(n_node_features)
        embedding_dimension = n_node_features
        graph_encoder = self.embedding_module(module_type="graph_sum", 
                                               node_features=node_features, 
                                               edge_features=edge_features, 
                                               memory=None, 
                                               neighbor_finder=neighbor_finder,
                                               time_encoder=time_encoder, 
                                               n_layers=1, 
                                               n_node_features=n_node_features, 
                                               n_edge_features=n_edge_features, 
                                               n_time_features=n_node_features,
                                               embedding_dimension=embedding_dimension, 
                                               device="cuda",
                                               n_heads=2, dropout=0.1, n_neighbors=None,
                                               use_memory=True)
        node_embeddings = graph_encoder.compute_embedding(memory=None, source_nodes=node_features,
                            timestamps=timestamps,
                            n_layers=1,
                            n_neighbors=20,
                            time_diffs=None)
        return node_embeddings
    
    
# class TGS(nn.Module):
    
#     def __init__(self, input_dim, hidden_dim, args):
#         """
#         Temporal Graph Sum encoder

#         """
#         super(TGS, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim  = hidden_dim
#         self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
#         self.relu = nn.ReLU()
    
    
#     def forward(self, data):
#         """
        
#         Parameters
#         ----------
#         data : dynamic graph, storing a sequence of timestamped graph snapshots.

#         Returns
#         -------
#         hidden : hidden representation of nodes.

#         """
#         x_final, edge_index_final, batch_final = data[-1].x, data[-1].edge_index, data[-1].batch
#         batch_final = batch_final.to(x_final.device)
        
#         h = np.zeros_like(data)
#         h_tilde = np.zeros_like(h)
#         np.delete(h_tilde, 0)
#         h[0] = x_final
#         for i in h_tilde:
#             for j in h_tilde[i]:
#                 edge_index_current = data[i].edge_index
#                 h_tilde[i][j] = np.sum(np.concatenate(h[i][j],edge_index_current[j]))
#                 x = np.concatenate(h[i][j],h_tilde[i][j])
#                 hidden = self.fc1(x)
#                 relu = self.relu(hidden)
#                 h[i+1][j] = relu
    
#         hidden = np.zeros_like(h[-1])
#         for i in h[-1]:
#             hidden[i] = h[-1][i]
            
#         return hidden
    
class veracity_prediction(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        """
        Decoder for veracity prediction

        """
        super(veracity_prediction, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim  = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, output_dim)
        
    def forward(self, hidden):
        """

        Parameters
        ----------
        hidden : hidden representation of nodes from TGS encoder.

        Returns
        -------
        output : veracity prediction.

        """
        embed = self.fc1(hidden)
        relu = self.relu(embed)
        output = self.fc2(relu)
        soft = np.exp(output - np.max(output))
        output = soft/soft.sum()
        return output


def make_model(input_dim, embedding_dimension, output_dim, args): #hyperparameters to be defined
    "Construct a model from hyperparameters."
    model = EncoderDecoder(
        TGS(input_dim, embedding_dimension),
        veracity_prediction(input_dim=embedding_dimension, hidden_dim=embedding_dimension, 
                            output_dim=output_dim, args=args))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model