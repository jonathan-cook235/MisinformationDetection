from __future__ import division

import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import torch.nn.functional as F
import numpy as np
import sys
import os
import argparse
from process_data import create_dataset, to_label

from torch import autograd
import pickle
import random
from multiprocessing import Pool
import torch.multiprocessing as mp
FType = torch.FloatTensor
LType = torch.LongTensor


class MMDNE(nn.Module):
    def __init__(self, file_path, save_graph_path,model_name,save_model_path,
                 emb_size=32, neg_size=10, hist_len=2, directed=False,
                 batch_size=1000, epoch_num=100,
                 only_binary=True,seed=64, backprop_every = 1,
                 tlp_flag=False, trend_prediction=False, device='cpu',gpu=-1,
                 epsilon=1.0, epsilon1=1.0,epsilon2=1.0, epsilon3=1.0, dropout=0.5
    ):
        super(MMDNE, self).__init__()
        self.emb_size = emb_size
        self.neg_size = neg_size
        self.hist_len = hist_len

        self.only_binary = only_binary

        self.epochs = epoch_num
        self.batch_size = batch_size
        self.model_name = model_name
        self.save_model_path = save_model_path
        self.backprop_every = backprop_every

        self.seed = seed
        self.device = device
        self.gpu = gpu
        self.dropout = dropout

        self.epsilon = epsilon
        self.epsilon1=epsilon1
        self.epsilon2=epsilon2
        self.epsilon3 = epsilon3

        if self.only_binary:
            self.output_dim = 2
        else:
            self.output_dim = 4

        ## obtain data-related model parameters
        self.graph_data_dict, self.node_dim_dict, self.max_timestamp_dict, \
        self.node_dim, self.max_timestamp, self.train_ids, self.val_ids, self.test_ids, \
        self.labels, self.news_ids_to_consider, \
        self.preprocessed_tweet_fts, self.preprocessed_user_fts, \
        self.num_tweet_features, self.num_user_features = \
            create_dataset(directed, file_path, hist_len, neg_size, save_graph_path,
                           only_binary, seed, tlp_flag, trend_prediction)

        ## initialize model trainable parameters
        if 'cpu' not in self.device.type:
        # ??? single delta value for all nodes
            self.delta_s = Variable((torch.ones(1)).type(FType).cuda(self.gpu), requires_grad=True)
            self.delta_t = Variable((torch.ones(1)).type(FType).cuda(self.gpu), requires_grad=True)

            self.zeta = Variable((torch.ones(1)).type(FType).cuda(self.gpu), requires_grad=True)
            self.gamma = Variable((torch.ones(1)).type(FType).cuda(self.gpu), requires_grad=True)
            self.theta = Variable((torch.ones(1)).type(FType).cuda(self.gpu), requires_grad=True)
            self.time_W = Variable((torch.ones(1)).type(FType).cuda(self.gpu), requires_grad=True)
        else:
            self.delta_s = Variable((torch.ones(1)).type(FType), requires_grad=True)
            self.delta_t = Variable((torch.ones(1)).type(FType), requires_grad=True)

            self.zeta = Variable((torch.ones(1)).type(FType), requires_grad=True)
            self.gamma = Variable((torch.ones(1)).type(FType), requires_grad=True)
            self.theta = Variable((torch.ones(1)).type(FType), requires_grad=True)
            self.time_W = Variable((torch.ones(1)).type(FType), requires_grad=True)

        self.a = torch.nn.Parameter(torch.zeros(size=(2 * self.emb_size, 1)),requires_grad=True)
        # self.a = torch.nn.Parameter(torch.zeros(size=(2 * self.gat_hidden_size, 1)))
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # self.fts2emb = nn.Linear(self.num_user_features, self.emb_size) #usage: y = self.fts2emb(x)
        self.fts2emb = nn.Linear(self.num_user_features+self.num_tweet_features, self.emb_size, bias=True) #usage: y = self.fts2emb(x)
        self.bilinear = nn.Bilinear(self.emb_size, self.emb_size, 1, bias=True)
        self.aggre_emb = nn.Linear(3*self.emb_size, self.emb_size, bias=True)
        self.node_emb_output = nn.Linear(self.emb_size, self.output_dim, bias=True)
        # p – probability of an element to be zeroed
        self.dropout_layer = nn.Dropout(self.dropout)

        self.leakyrelu = torch.nn.LeakyReLU(0.2)  # alpha =0.2 for leakyrelu
        self.softplus = torch.nn.Softplus(beta=10.)
        self.MSE_criterion = torch.nn.MSELoss()
        self.CE_criterion = torch.nn.CrossEntropyLoss()

        self.para_to_opt = [self.delta_s, self.delta_t, self.zeta,self.gamma,self.theta, self.a, self.time_W] \
                           + list(self.fts2emb.parameters()) \
                           + list(self.bilinear.parameters()) \
                           + list(self.aggre_emb.parameters()) \
                           + list(self.node_emb_output.parameters())

    def get_emb_from_id(self, user_id, news_id, dim_num=1, dim_size=None):

        graph_data = self.graph_data_dict[news_id]
        user_tweet_dict = graph_data.user_tweet_dict

        if dim_num==1:
            ##  only one node-id
            tweet_id = user_tweet_dict[user_id]
            user_fts = torch.FloatTensor(self.preprocessed_user_fts[user_id])
            tweet_fts = torch.FloatTensor(self.preprocessed_tweet_fts[tweet_id])

        else:
            if dim_num == 2:
                # a two-dimensional vector of node-id; return (bach, emb_dim)
                tweet_id = [user_tweet_dict[user_] for user_ in user_id]
                user_fts = torch.FloatTensor([self.preprocessed_user_fts[user_] for user_ in user_id])
                tweet_fts = torch.FloatTensor([self.preprocessed_tweet_fts[tweet_] for tweet_ in tweet_id])

            elif dim_num==3 and dim_size:
                # a three-dimensional vector of node-id; return (bach-size, dim_size, emb_dim)
                batch_size = len(user_id[0])#str
                tweet_id = [user_tweet_dict[user_] for hist in user_id for user_ in hist]
                user_fts = torch.FloatTensor([self.preprocessed_user_fts[user_] for hist in user_id for user_ in hist])#
                tweet_fts = torch.FloatTensor([self.preprocessed_tweet_fts[tweet_] for tweet_ in tweet_id])#.view(self.batch_size, dim_size, -1)
                # print('user_fts.size-',user_fts.size(), 'tweet_fts.size-',tweet_fts.size())
                user_fts = user_fts.view(batch_size, dim_size, -1)
                tweet_fts=tweet_fts.view(batch_size, dim_size, -1)

        # node_fts = user_fts
        node_fts = torch.cat([user_fts, tweet_fts], dim=-1).to(self.device)
        node_emb = self.fts2emb(node_fts)  # (bach, emb_dim)
        return node_emb

    def compute_intensity(self, s_nodes, t_nodes, event_time,
                      s_h_nodes, s_h_times, s_h_time_mask,
                      t_h_nodes, t_h_times, t_h_time_mask,
                      s_neg_node,t_neg_node,news_id):

        s_node_emb = self.get_emb_from_id(s_nodes,news_id, dim_num=2)
        t_node_emb = self.get_emb_from_id(t_nodes,news_id, dim_num=2)
        s_h_node_emb = self.get_emb_from_id(s_h_nodes,news_id, dim_num=3, dim_size=self.hist_len)
        t_h_node_emb = self.get_emb_from_id(t_h_nodes,news_id, dim_num=3, dim_size=self.hist_len)

        delta_s = self.delta_s#.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)  # (b,1)
        d_time_s = torch.abs(event_time.unsqueeze(1) - s_h_times)  # (batch, hist_len)
        delta_t = self.delta_t#.index_select(0, Variable(t_nodes.view(-1))).unsqueeze(1)  # TODO: delta_t ???
        d_time_t = torch.abs(event_time.unsqueeze(1) - t_h_times)  # (batch, hist_len)

        ###############################################################
        # GAT attention_rewrite
        ## delta_s: exponential time-decaying
        for i in range(self.hist_len):
            s_h_node_emb_i = torch.transpose(s_h_node_emb[:, i:(i + 1), :], dim0=1, dim1=2).squeeze()  # (b, dim)
            s_node_emb_i = s_node_emb  # (b, dim)
            d_time_s_i = Variable(d_time_s)[:,i:(i+1)].to(self.device)
            if i == 0:
                a_input = torch.cat([s_node_emb_i, s_h_node_emb_i],dim=1)  # (b, 2*dim)
                sim_s_s_his = self.leakyrelu(torch.exp(-delta_s * d_time_s_i) * torch.mm(a_input, self.a))  # (b.dim),leakyrelu
            else:
                a_input = torch.cat([s_node_emb_i, s_h_node_emb_i], dim=1)
                sim_s_s_his = torch.cat([sim_s_s_his,
                                         self.leakyrelu(torch.exp(-delta_s * d_time_s_i) * torch.mm(a_input, self.a))], dim=1)

        for i in range(self.hist_len):
            t_h_node_emb_i = torch.transpose(t_h_node_emb[:, i:(i + 1), :], dim0=1, dim1=2).squeeze()
            t_node_emb_i = t_node_emb
            d_time_t_i = Variable(d_time_t)[:, i:(i + 1)].to(self.device)  # (b,1)
            if i == 0:
                # a_input = torch.cat([torch.mm(t_node_emb_i, self.W), torch.mm(t_h_node_emb_i, self.W)], dim=1)  # (b, 2*dim)
                a_input = torch.cat([t_node_emb_i, t_h_node_emb_i], dim=1)
                sim_t_t_his = self.leakyrelu(torch.exp(-delta_t * d_time_t_i) * torch.mm(a_input, self.a))
            else:
                # a_input = torch.cat([torch.mm(t_node_emb_i, self.W), torch.mm(t_h_node_emb_i, self.W)], dim=1)
                a_input = torch.cat([t_node_emb_i, t_h_node_emb_i], dim=1)
                sim_t_t_his = torch.cat([sim_t_t_his,
                                         self.leakyrelu(torch.exp(-delta_t * d_time_t_i) * torch.mm(a_input, self.a))], dim=1)

        att_s_his_s = softmax(sim_s_s_his, dim=1)  # (batch, h)
        att_t_his_t = softmax(sim_t_t_his, dim=1)  # (batch, h)

        ## Compute the intensity lambda of positive (truely occurred) events
        # p_mu = self.softplus(self.bilinear(s_node_emb, t_node_emb).squeeze(-1))
        # p_alpha_s = self.softplus(self.bilinear(s_h_node_emb, t_node_emb.unsqueeze(1).repeat(1, self.hist_len, 1)).squeeze(-1))# batch-size, hist-len, emb-size
        p_mu = ((s_node_emb - t_node_emb) ** 2).sum(dim=1).neg()  # (batch, 1)
        p_alpha_s = ((s_h_node_emb - t_node_emb.unsqueeze(1)) ** 2).sum(dim=2).neg()  # (batch, h_len)

        p_lambda = p_mu + (att_s_his_s * p_alpha_s * torch.exp(delta_s * Variable(d_time_s).to(self.device))
                           * Variable(s_h_time_mask).to(self.device)).sum(dim=1)

        ## Compute the intensity lambda of negative (did not occur) events
        s_n_node_emb = self.get_emb_from_id(s_neg_node,news_id, dim_num=3, dim_size=self.neg_size)
        t_n_node_emb = self.get_emb_from_id(t_neg_node,news_id, dim_num=3, dim_size=self.neg_size)

        # n_mu_s = self.bilinear(s_node_emb.unsqueeze(1).repeat(1, self.neg_size, 1), t_n_node_emb).squeeze(-1)  # (batch, neg_len)
        # n_mu_t = self.softplus(self.bilinear(t_node_emb.unsqueeze(1).repeat(1, self.neg_size, 1), s_n_node_emb).squeeze(-1))
        # n_alpha_s = self.bilinear(s_h_node_emb.unsqueeze(2).repeat(1, 1, self.neg_size, 1),
        #                           t_n_node_emb.unsqueeze(1).repeat(1, self.hist_len, 1, 1)).squeeze(-1)
        # n_alpha_t = self.softplus(self.bilinear(t_h_node_emb.unsqueeze(2).repeat(1, 1, self.neg_size, 1),
        #                           s_n_node_emb.unsqueeze(1).repeat(1, self.hist_len, 1, 1)).squeeze(-1))

        n_mu_s = ((s_node_emb.unsqueeze(1) - t_n_node_emb) ** 2).sum(dim=2).neg()  # (batch, neg_len)
        n_mu_t = ((t_node_emb.unsqueeze(1) - s_n_node_emb) ** 2).sum(dim=2).neg()
        n_alpha_s = ((s_h_node_emb.unsqueeze(2) - t_n_node_emb.unsqueeze(1)) ** 2).sum(dim=3).neg()
        n_alpha_t = ((t_h_node_emb.unsqueeze(2) - s_n_node_emb.unsqueeze(1)) ** 2).sum(dim=3).neg()

        n_lambda_s = n_mu_s + (att_s_his_s.unsqueeze(2) * n_alpha_s * (torch.exp(delta_s * Variable(d_time_s).to(self.device)).unsqueeze(2))
                               * (Variable(s_h_time_mask).unsqueeze(2)).to(self.device)).sum(dim=1)  # TODO: global_att_s.unsqueeze(1)
        n_lambda_t = n_mu_t + (att_t_his_t.unsqueeze(2) * n_alpha_t * (torch.exp(delta_t * Variable(d_time_t).to(self.device)).unsqueeze(2))
                               * (Variable(t_h_time_mask).unsqueeze(2)).to(self.device)).sum(dim=1)

        return p_lambda, n_lambda_s, n_lambda_t  # max p_lambda, min n_lambda

    def global_forward(self, s_nodes, t_nodes, event_time, node_sum,news_id):
        # print('self.theta',self.theta, 'self.zeta',self.zeta, 'self.gamma',self.gamma,)
        s_node_emb = self.get_emb_from_id(s_nodes,news_id,dim_num=2)
        t_node_emb = self.get_emb_from_id(t_nodes,news_id,dim_num=2)

        beta = torch.sigmoid(self.bilinear(s_node_emb, t_node_emb)).squeeze(-1) # (batch) Equation-11 torch.sigmoid
        # beta  = self.softplus(self.bilinear(s_node_emb, t_node_emb)).squeeze(-1)
        event_time = Variable(event_time).abs().to(self.device)+1e-6
        r_t = beta / torch.pow(event_time, self.theta)
        node_sum = Variable(node_sum).abs().to(self.device)
        # delta_e_pred must be non-negative; node_sum-1
        delta_e_pred = self.softplus( r_t * node_sum * (self.zeta * torch.pow(node_sum, self.gamma))) # Equation-10

        if torch.isnan(delta_e_pred).any():
            print('beta',beta,'event_time',event_time,'self.theta',self.theta,
                  'torch.pow(event_time, self.theta)',torch.pow(event_time, self.theta),
                  'node_sum',node_sum,
                  'self.zeta',self.zeta, 'self.gamma',self.gamma,
                  'torch.pow(node_sum, self.gamma)', torch.pow(node_sum, self.gamma)
                  )
            assert(not torch.isnan(delta_e_pred).any())
        return delta_e_pred

    def local_loss(self, s_nodes, t_nodes, event_time,
                   s_h_nodes, s_h_times, s_h_time_mask,
                   t_h_nodes, t_h_times, t_h_time_mask,
                   neg_s_node,neg_t_node,news_id):

        p_lambdas, n_lambdas_s,n_lambdas_t = self.compute_intensity(s_nodes, t_nodes, event_time,
                                                                 s_h_nodes, s_h_times, s_h_time_mask,
                                                                 t_h_nodes, t_h_times, t_h_time_mask,
                                                                 neg_s_node, neg_t_node,news_id)

        ## Compute the Negative Log-Likelihood as Equation 14 in MMDNE
        local_loss =  - torch.log(p_lambdas.sigmoid() + 1e-6) \
                      - torch.log(n_lambdas_s.neg().sigmoid() + 1e-6).sum(dim=1) \
                      - torch.log(n_lambdas_t.neg().sigmoid() + 1e-6).sum(dim=1) # remove the history of t_node, i.e., ccc

        ## Compute the error of timstamp prediction
        estimate_timestamp = self.timestamp_predict(intensity=p_lambdas)
        timestamp_loss = self.MSE_criterion(torch.exp(estimate_timestamp), torch.exp(Variable(event_time).to(self.device)))

        return local_loss, timestamp_loss

    def global_loss(self,s_nodes, t_nodes, event_time, delta_e_true, node_sum,news_id):
        delta_e_pred = self.global_forward(s_nodes, t_nodes, event_time, node_sum,news_id)

        global_loss = self.MSE_criterion(torch.log(delta_e_pred + 1e-6), torch.log(Variable(delta_e_true).to(self.device)+1e-6))
        # loss = ((delta_e_pred - Variable(delta_e_true))**2).mean(dim=-1)
        if torch.isnan(global_loss):
            print('delta_e_pred',delta_e_pred, 'delta_e_true',delta_e_true,
                  'torch.log(delta_e_pred + 1e-6)',torch.log(delta_e_pred + 1e-6),
                  'torch.log(Variable(delta_e_true) + 1e-6)',torch.log(Variable(delta_e_true).to(self.device) + 1e-6),
                  'loss',global_loss)
            assert (not torch.isnan(global_loss))
        return global_loss, delta_e_pred

    def veracity_predict(self, news_id):

        first_node_id = self.graph_data_dict[news_id].first_node
        first_node_emb = self.get_emb_from_id(first_node_id,news_id,dim_num=1).view(1,-1)# (1, emb_dim)

        all_node_id = self.graph_data_dict[news_id].node_list
        all_node_emb = self.get_emb_from_id(all_node_id,news_id,dim_num=2)

        all_node_emb_mean = torch.mean(all_node_emb,dim=0).view(1,-1)# (1, emb_dim)
        all_node_emb_max_tpl = torch.max(all_node_emb, dim=0)
        all_node_emb_max = all_node_emb_max_tpl.values.view(1, -1)  # (1, emb_dim)
        all_node_emb_pool = torch.cat([all_node_emb_mean, all_node_emb_max, first_node_emb], dim=1)

        aggre_emb = self.softplus(self.dropout_layer(self.aggre_emb(all_node_emb_pool)))
        output_emb = self.softplus(self.dropout_layer(self.node_emb_output(aggre_emb)))
        # output_emb1 = self.softplus(self.dropout_layer(self.node_emb_output1(aggre_emb)))
        # output_emb = self.softplus(self.dropout_layer(self.node_emb_output2(output_emb1)))

        output = F.log_softmax(output_emb, dim=1)

        return output

    def veracity_loss(self, news_id):
        output = self.veracity_predict(news_id)

        label = self.labels[news_id]
        y = torch.tensor(to_label(label)).to(self.device)
        vera_loss = self.CE_criterion(output, y)
        return vera_loss

    def timestamp_predict(self, intensity, print_info=False):
        timestep = 1e-8
        # integral_ = torch.cumsum(timestep * intensity, dim=0)
        integral_ = intensity
        # density for the time-until-next-event law
        density = intensity * torch.exp(-integral_)
        # Check density
        if print_info:
            print("sum of density:", (timestep * density).sum())

        # n_samples = 1000
        # max_timestamp =  1 # after normalization
        # dt_vals = torch.linspace(0, max_timestamp, n_samples + 1).to(self.device)
        # t_pit = dt_vals * density  # integrand for the time estimator
        # trapeze method
        # estimate_dt = (timestep * 0.5 * (t_pit[1:] + t_pit[:-1])).sum()
        estimate_timestamp = density * self.time_W

        return estimate_timestamp

    def update(self, s_nodes, t_nodes, event_time,
               s_h_nodes, s_h_times, s_h_time_mask,
               t_h_nodes, t_h_times, t_h_time_mask,
               neg_s_node,neg_t_node,
               delta_e_true, node_sum,
               news_id):

        ## XXX ##
        max_timestamp = self.max_timestamp_dict[news_id]
        event_time = event_time/max_timestamp
        s_h_times = s_h_times/max_timestamp
        t_h_times = t_h_times/max_timestamp

        local_loss, timestamp_loss = self.local_loss(s_nodes, t_nodes, event_time,
                                     s_h_nodes, s_h_times, s_h_time_mask,
                                     t_h_nodes, t_h_times, t_h_time_mask,
                                     neg_s_node, neg_t_node,news_id)

        global_loss, delta_e_pred = self.global_loss(s_nodes, t_nodes, event_time,
                                      delta_e_true, node_sum,news_id)
        vera_loss = self.veracity_loss(news_id)

        weighted_local_loss = self.epsilon1 * local_loss.mean()
        # weighted_global_loss = weighted_local_loss
        weighted_global_loss = self.epsilon2 * global_loss.mean()
        weighted_timestamp_loss = self.epsilon3 * timestamp_loss.mean()
        weighted_vera_loss = self.epsilon * vera_loss

        loss = weighted_local_loss + weighted_vera_loss #+ weighted_global_loss + weighted_timestamp_loss

        return loss, weighted_local_loss.detach().cpu().numpy(), \
               weighted_global_loss.detach().cpu().numpy(),\
               weighted_vera_loss.detach().cpu().numpy(),\
               weighted_timestamp_loss.detach().cpu().numpy(),\
               delta_e_pred, event_time, delta_e_true
