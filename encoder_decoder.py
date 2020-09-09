#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:21:24 2020

@author: jonnycook
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dynamic_graph.TGS_utils import get_neighbor_finder, NeighborFinder
from dynamic_graph.embedding_module import get_embedding_module
from dynamic_graph.time_encoding import TimeEncoder

from point_process.sahp import SAHP
from point_process.train_sahp import MaskBatch,make_sahp_model


import torch_geometric.nn as pyg_nn

def make_model(n_node_features, output_dim, args, device):  # hyperparameters to be defined
    "Construct a model from hyperparameters."
    tgs_encoder  = TGS(
            n_node_features,
            # neighbor_finder=train_ngh_finder,
            # node_features=node_features,
            # edge_features=edge_features,
            device=device, n_layers=args.n_layer, n_heads=args.n_head, dropout=args.dropout,
            # use_memory=USE_MEMORY,
            # message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            # memory_update_at_start=not args.memory_update_at_end,
            embedding_module_type=args.embedding_module,
            # message_function=args.message_function,
            # aggregator_type=args.aggregator,
            n_neighbors=args.n_degree,
            # mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            # mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst
        )
    veracity_predictor =  Veracity_Pred(input_dim=n_node_features, hidden_dim=args.hidden_dim,
                        output_dim=output_dim, args=args)
    sahp_model = make_sahp_model(nLayers=6, d_model=128, atten_heads=8, dropout=0.1, process_dim=10,
                       device='cpu', pe='concat', max_sequence_length=4096)
    timestamp_predictor = Timestamp_Pred(tpp=sahp_model,input_dim=n_node_features, hidden_dim=args.hidden_dim,
                            output_dim=output_dim, args=args)
    model = EncoderDecoder(encoder = tgs_encoder,
                           decoder1 = veracity_predictor,
                           decoder2 = timestamp_predictor
                           )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. 
    
    """
    def __init__(self, encoder, decoder1, decoder2):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        
    def forward(self, data):
        node_embeddings = self.encode(data)
        veracity_pred_loss  = self.decode1(node_embeddings, data.batch)
        # time_pred_loss = self.decode2(node_embeddings, data.batch)

        total_loss = veracity_pred_loss #+ time_pred_loss

        return total_loss
    
    def encode(self, data):
        return self.encoder(data)
    
    def decode1(self, source_embedding, batch):
        return self.decoder1(source_embedding, batch)

    def decode2(self, source_embedding, batch):
        return self.decoder2(source_embedding, batch)
    
class TGS(nn.Module):
    
    def __init__(self,
                 n_node_features,
                 # neighbor_finder, node_features, edge_features,
                 device='cpu', n_layers=2, n_heads=2, dropout=0.1,
                 use_memory=False,
                 # memory_update_at_start=True, message_dimension=100,
                 # memory_dimension=500,
                 embedding_module_type="graph_sum",
                 # message_function="mlp",
                 # mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
                 # std_time_shift_dst=1,
                 # aggregator_type="last",
                 n_neighbors=None
                 ):
        """
        Temporal Graph Sum encoder

        """
        super(TGS, self).__init__()
        # self.memory = None
        self.embedding_module_type = embedding_module_type
        self.embedding_module = get_embedding_module

        self.n_layers = n_layers
        # self.neighbor_finder = neighbor_finder
        self.device = device
        self.n_neighbors = n_neighbors

        self.n_node_features =  n_node_features
        # self.n_node_features = node_features.shape[1]
        # self.n_nodes = node_features.shape[0]
        self.n_edge_features = 20#???
        self.n_time_features = 5
        self.embedding_dimension = self.n_node_features

        # self.time_encoder = TimeEncode(dimension=self.n_node_features)
        self.time_encoder = TimeEncoder(dimension=self.n_time_features)

        self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                     # node_features=self.node_features,
                                                     # edge_features=self.edge_features,
                                                     # memory=self.memory,
                                                     # neighbor_finder=self.neighbor_finder,
                                                     time_encoder=self.time_encoder,
                                                     n_layers=self.n_layers,
                                                     n_node_features=self.n_node_features,
                                                     n_edge_features=self.n_edge_features,
                                                     n_time_features=self.n_time_features,
                                                     embedding_dimension=self.embedding_dimension,
                                                     device=self.device,
                                                     n_heads=n_heads, dropout=dropout,
                                                     use_memory=use_memory,
                                                     n_neighbors=self.n_neighbors)

    def forward(self, data):

        # define temporal neighbor search
        neighbor_finder = get_neighbor_finder(data)#???

        # load datapoint features from batch
        node_features = data.x #???
        timestamps = data.t #???
        # edge_features = data.edge_index #???

        # n_nodes = node_features.shape[0]
        # n_node_features = node_features.shape[1]#???
        # n_edge_features = edge_features.shape[1]#???


        # time_encoder = TimeEncode(n_node_features)
        # embedding_dimension = n_node_features

        # define graph sum encoder
        # graph_encoder = self.embedding_module(module_type="graph_sum",
        #                                        node_features=node_features,
        #                                        edge_features=edge_features,
        #                                        memory=None,
        #                                        neighbor_finder=neighbor_finder,
        #                                        time_encoder=self.time_encoder,
        #                                        n_layers=1,
        #                                        n_node_features=n_node_features,
        #                                        n_edge_features=n_edge_features,
        #                                        n_time_features=n_time_features,
        #                                        embedding_dimension=embedding_dimension,
        #                                        device="cuda",
        #                                        n_heads=2, dropout=0.1, n_neighbors=None,
        #                                        use_memory=True)

        # aggregate node embeddings
        node_embeddings = \
            self.embedding_module.compute_embedding(
                raw_node_features=node_features,
                source_nodes=node_features,
                timestamps=timestamps,
                neighbor_finder=neighbor_finder,
                n_layers=self.n_layers,
                n_neighbors=self.n_neighbors,
                time_diffs=None,
                memory = None)
        return node_embeddings
    
class Veracity_Pred(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        """
        Decoder for veracity prediction

        """
        super(Veracity_Pred, self).__init__()
        self.dropout = float(args.dropout)

        self.input_dim = input_dim
        self.hidden_dim  = hidden_dim
        # self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(self.hidden_dim, output_dim)
        # self.sft =  nn.Softmax()

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(3 * hidden_dim, 3 * hidden_dim), nn.Dropout(self.dropout),
            nn.Linear(3 * hidden_dim, output_dim))
        
    def forward(self, x, batch):
        """

        Parameters
        ----------
        hidden : hidden representation of nodes from dynamic_graph encoder.

        Returns
        -------
        output : veracity prediction.

        """
        x = self.relu(self.fc1(x))
        # output = self.sft(self.fc2(hidden))
        # return output

        # concatenate max_pool, mean_pool and embedding of first node (i.e. the news root)
        x1 = pyg_nn.global_max_pool(x, batch)  # shape batch_size * embedding size
        x2 = pyg_nn.global_mean_pool(x, batch)

        batch_size = x1.size(0)
        indices_first_nodes = [(batch == i).nonzero()[0] for i in range(batch_size)]
        x3 = x[indices_first_nodes, :]

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.post_mp(x)

        return F.log_softmax(x, dim=1)

class Timestamp_Pred(nn.Module):

    def __init__(self, tpp, input_dim, hidden_dim, output_dim, args):
        """
        Decoder for timestamp prediction

        """
        super(Timestamp_Pred, self).__init__()
        self.dropout = float(args.dropout)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.tpp = tpp

    def forward(self, x, batch):
        """

        Parameters
        ----------
        hidden : hidden representation of nodes from dynamic_graph encoder.

        Returns
        -------
        output : timestamp prediction.

        """

        src_mask = MaskBatch(x,pad=self.tpp.process_dim, device='cpu')

        self.tpp.forward(batch.t, x, src_mask)