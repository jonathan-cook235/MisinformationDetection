# coding: utf-8
# author: lu yf
# create date: 2018/11/12

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
from process_data import create_dataset, to_label

from torch import autograd
import pickle
import random


FType = torch.FloatTensor
LType = torch.LongTensor

DID = 0


class MMDNE(nn.Module):
    def __init__(self, file_path, save_graph_path,
                 save_embedding_path, emb_size=32, gat_hidden_size=32, neg_size=10, hist_len=2, directed=False,
                 learning_rate=0.01, batch_size=1000, save_step=10, epoch_num=1, optim='SGD',
                 tlp_flag=False, trend_prediction=False, only_binary=True,seed=64,
                 epsilon=1.0, epsilon1=1.0,epsilon2=1.0, dropout=0.1
    ):
        super(MMDNE, self).__init__()
        self.emb_size = emb_size
        self.neg_size = neg_size
        self.hist_len = hist_len
        # self.gat_hidden_size = gat_hidden_size

        self.only_binary = only_binary

        self.epochs = epoch_num
        self.batch_size = batch_size
        self.save_step = save_step
        self.save_embedding_path = save_embedding_path

        self.seed = seed
        self.dropout = dropout
        self.lr = learning_rate
        self.optim = optim
        self.tlp_flag = tlp_flag
        self.trend_prediction = trend_prediction

        self.epsilon = epsilon
        self.epsilon1=epsilon1
        self.epsilon2=epsilon2

        if self.only_binary:
            self.output_dim = 2
        else:
            self.output_dim = 4

        ## obtain data-related model parameters
        self.graph_data_dict, self.node_dim_dict, self.max_d_time_dict, \
        self.node_dim, self.max_d_time, self.train_ids, self.val_ids, self.test_ids, \
        self.labels, self.news_ids_to_consider, \
        self.preprocessed_tweet_fts, self.preprocessed_user_fts, \
        self.num_tweet_features, self.num_user_features = \
            create_dataset(directed, file_path, hist_len, neg_size, save_graph_path,
                           only_binary, seed, tlp_flag, trend_prediction)

        ## initialize model trainable parameters

        # ??? single delta value for all nodes
        self.delta_s = Variable((torch.ones(1)).type(FType), requires_grad=True)
        self.delta_t = Variable((torch.ones(1)).type(FType), requires_grad=True)

        self.zeta = Variable((torch.ones(1)).type(FType), requires_grad=True)
        self.gamma = Variable((torch.ones(1)).type(FType), requires_grad=True)
        self.theta = Variable((torch.ones(1)).type(FType), requires_grad=True)

        self.a = torch.nn.Parameter(torch.zeros(size=(2 * self.emb_size, 1)),requires_grad=True)
        # self.a = torch.nn.Parameter(torch.zeros(size=(2 * self.gat_hidden_size, 1)))
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # self.fts2emb = nn.Linear(self.num_user_features, self.emb_size) #usage: y = self.fts2emb(x)
        self.fts2emb = nn.Linear(self.num_user_features+self.num_tweet_features, self.emb_size) #usage: y = self.fts2emb(x)
        self.bilinear = nn.Bilinear(self.emb_size, self.emb_size, 1)
        self.aggre_emb = nn.Linear(3*self.emb_size, self.emb_size)
        self.node_emb_output = nn.Linear(self.emb_size, self.output_dim)

        self.dropout_layer = nn.Dropout(self.dropout)
        self.leakyrelu = torch.nn.LeakyReLU(0.2)  # alpha =0.2 for leakyrelu
        # self.W, self.att_param,
        para_to_opt = [self.delta_s, self.delta_t, self.zeta,self.gamma,self.theta, self.a] \
                      + list(self.fts2emb.parameters()) + list(self.node_emb_output.parameters()) \
                      + list(self.bilinear.parameters()) + list(self.aggre_emb.parameters())
                      # + list(self.global_att_linear_layer.parameters()) \

        if self.optim == 'SGD':
            self.opt = SGD(lr=learning_rate,momentum=0.9,weight_decay=0.01, params=para_to_opt)
        elif self.optim == 'Adam':
            self.opt = Adam(lr=learning_rate, weight_decay=0.01, params=para_to_opt)

    def get_emb_from_id(self, user_id, news_id, dim_num=1, dim_size=None):

        graph_data = self.graph_data_dict[news_id]
        user_tweet_dict = graph_data.user_tweet_dict

        # if torch.is_tensor(user_id):#int
        #     batch_size = user_id.size()[0]
        #     user_id = user_id.numpy()

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
        node_fts = torch.cat([user_fts, tweet_fts], dim=-1)
        node_emb = self.fts2emb(node_fts)  # (bach, emb_dim)
        return node_emb

    def local_forward(self, s_nodes, t_nodes, e_times,
                      s_h_nodes, s_h_times, s_h_time_mask,
                      t_h_nodes, t_h_times, t_h_time_mask,
                      s_neg_node,t_neg_node,news_id):

        s_node_emb = self.get_emb_from_id(s_nodes,news_id, dim_num=2)
        t_node_emb = self.get_emb_from_id(t_nodes,news_id, dim_num=2)
        s_h_node_emb = self.get_emb_from_id(s_h_nodes,news_id, dim_num=3, dim_size=self.hist_len)
        # t_h_node_emb = self.get_emb_from_id(t_h_nodes,news_id, dim_num=3,dim_size=self.hist_len)

        delta_s = self.delta_s#.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)  # (b,1)
        d_time_s = torch.abs(e_times.unsqueeze(1) - s_h_times)  # (batch, hist_len)
        # d_time_s = self.leakyrelu(d_time_s / max_d_time)
        # delta_t = self.delta_t#.index_select(0, Variable(t_nodes.view(-1))).unsqueeze(1)  # TODO: delta_t ???
        # d_time_t = torch.abs(e_times.unsqueeze(1) - t_h_times)  # (batch, hist_len)
        # d_time_t = self.leakyrelu(d_time_t / max_d_time)

        # GAT attention_rewrite
        ## delta_s: exponential time-decaying
        for i in range(self.hist_len):
            s_h_node_emb_i = torch.transpose(s_h_node_emb[:, i:(i + 1), :], dim0=1, dim1=2).squeeze()  # (b, dim)
            s_node_emb_i = s_node_emb  # (b, dim)
            d_time_s_i = Variable(d_time_s)[:,i:(i+1)]
            if i == 0:
                # a_input = torch.cat([torch.mm(s_node_emb_i, self.W),torch.mm(s_h_node_emb_i, self.W)],dim=1)  # (b, 2*dim)
                a_input = torch.cat([s_node_emb_i, s_h_node_emb_i],dim=1)  # (b, 2*dim)
                sim_s_s_his = self.leakyrelu(torch.exp(-delta_s * d_time_s_i) * torch.mm(a_input, self.a))  # (b.dim)
            else:
                # a_input = torch.cat([torch.mm(s_node_emb_i, self.W),torch.mm(s_h_node_emb_i, self.W)],dim=1)
                a_input = torch.cat([s_node_emb_i, s_h_node_emb_i], dim=1)
                sim_s_s_his = torch.cat([sim_s_s_his,
                                         self.leakyrelu(torch.exp(-delta_s * d_time_s_i) * torch.mm(a_input, self.a))], dim=1)

        # for i in range(self.hist_len):
        #     t_h_node_emb_i = torch.transpose(t_h_node_emb[:, i:(i + 1), :], dim0=1, dim1=2).squeeze()
        #     t_node_emb_i = t_node_emb
        #     d_time_t_i = Variable(d_time_t)[:, i:(i + 1)]  # (b,1)
        #     if i == 0:
        #         a_input = torch.cat([torch.mm(t_node_emb_i, self.W), torch.mm(t_h_node_emb_i, self.W)], dim=1)  # (b, 2*dim)
        #         sim_t_t_his = self.leakyrelu(torch.exp(-delta_s * d_time_t_i) * torch.mm(a_input, self.a))
        #     else:
        #         a_input = torch.cat([torch.mm(t_node_emb_i, self.W), torch.mm(t_h_node_emb_i, self.W)], dim=1)
        #         sim_t_t_his = torch.cat([sim_t_t_his,
        #                                  self.leakyrelu(torch.exp(-delta_s * d_time_t_i) * torch.mm(a_input, self.a))],
        #                                 dim=1)

        att_s_his_s = softmax(sim_s_s_his, dim=1)  # (batch, h)
        # att_t_his_t = softmax(sim_t_t_his, dim=1)  # (batch, h)

        # s_his_hat_emb_inter = ((att_s_his_s * Variable(s_h_time_mask)).unsqueeze(2) *
        #                        torch.mm(s_h_node_emb.view(s_h_node_emb.size()[0] * self.hist_len, -1), self.W).
        #                        view(s_h_node_emb.size()[0],self.hist_len,-1)).sum(dim=1)  # (b,dim)
        # t_his_hat_emb_inter = ((att_t_his_t * Variable(t_h_time_mask)).unsqueeze(2) *
        #                        torch.mm(t_h_node_emb.view(t_h_node_emb.size()[0] * self.hist_len, -1), self.W).
        #                        view(t_h_node_emb.size()[0],self.hist_len,-1)).sum(dim=1)
        #
        # temporal-self-attention
        # remove the history of t_node, i.e., remove it from global_att
        # global attention is for s-hist and t-hist
        # global_att = softmax(torch.tanh(self.global_att_linear_layer(torch.transpose(
        #     torch.cat([(s_his_hat_emb_inter * torch.exp(-delta_s * Variable(d_time_s.mean(dim=1)).unsqueeze(1))).unsqueeze(2),
        #                (t_his_hat_emb_inter * torch.exp(-delta_t * Variable(d_time_t.mean(dim=1)).unsqueeze(1))).unsqueeze(2)],
        #               dim=2),dim0=1,dim1=2))),dim=1).squeeze(2)  # (dim, 2)
        # global_att_s = global_att[:, 0]
        # global_att_t = global_att[:, 1]
        # self.global_attention = global_att

        p_mu = self.bilinear(s_node_emb, t_node_emb).squeeze(-1)
        p_alpha_s = self.bilinear(s_h_node_emb, t_node_emb.unsqueeze(1).repeat(1, self.hist_len, 1)).squeeze(-1)# batch-size, hist-len, emb-size
        # p_alpha_t = self.bilinear(t_h_node_emb, s_node_emb.unsqueeze(1).repeat(1, self.hist_len, 1)).squeeze(-1)

        aaa = (att_s_his_s * p_alpha_s * torch.exp(delta_s * Variable(d_time_s)) * Variable(s_h_time_mask)).sum(dim=1)
        # aaa = global_att_s * (att_s_his_s * p_alpha_s * torch.exp(delta_s * Variable(d_time_s)) * Variable(s_h_time_mask)).sum(dim=1)
        # bbb = global_att_t * (att_t_his_t * p_alpha_t * torch.exp(delta_t * Variable(d_time_t)) * Variable(t_h_time_mask)).sum(dim=1)

        p_lambda = p_mu \
                   + aaa \
                   # + bbb # remove the history of t_node

        # s_n_node_emb = self.get_emb_from_id(s_neg_node,news_id, dim_num=3, dim_size=self.neg_size)
        t_n_node_emb = self.get_emb_from_id(t_neg_node,news_id,dim_num=3, dim_size=self.neg_size)

        n_mu_s = self.bilinear(s_node_emb.unsqueeze(1).repeat(1, self.neg_size, 1), t_n_node_emb).squeeze(-1)  # (batch, neg_len)
        # n_mu_t = self.bilinear(t_node_emb.unsqueeze(1).repeat(1, self.neg_size, 1), s_n_node_emb).squeeze(-1)

        n_alpha_s = self.bilinear(s_h_node_emb.unsqueeze(2).repeat(1, 1, self.neg_size, 1),
                                  t_n_node_emb.unsqueeze(1).repeat(1, self.hist_len, 1, 1)).squeeze(-1)
        # n_alpha_t = self.bilinear(t_h_node_emb.unsqueeze(2).repeat(1, 1, self.neg_size, 1),
        #                           s_n_node_emb.unsqueeze(1).repeat(1, self.hist_len, 1, 1)).squeeze(-1)

        n_lambda_s = n_mu_s + (att_s_his_s.unsqueeze(2) * n_alpha_s * (torch.exp(delta_s * Variable(d_time_s)).unsqueeze(2))
                               * (Variable(s_h_time_mask).unsqueeze(2))).sum(dim=1)  # TODO: global_att_s.unsqueeze(1)

        # n_lambda_s = n_mu_s \
        #              + global_att_s.unsqueeze(1) * (att_s_his_s.unsqueeze(2) * n_alpha_s
        #                                             * (torch.exp(delta_s * Variable(d_time_s)).unsqueeze(2))
        #                                             * (Variable(s_h_time_mask).unsqueeze(2))).sum(dim=1)   # TODO: global_att_s.unsqueeze(1)
        #
        # n_lambda_t = n_mu_t \
        #              + global_att_t.unsqueeze(1) * (att_t_his_t.unsqueeze(2) * n_alpha_t
        #                                             * (torch.exp(delta_t * Variable(d_time_t)).unsqueeze(2))
        #                                             * (Variable(t_h_time_mask).unsqueeze(2))).sum(dim=1)

        return p_lambda, n_lambda_s#, n_lambda_t  # max p_lambda, min n_lambda

    def global_forward(self, s_nodes, t_nodes, e_times, delta_n_true, node_sum, edge_last_time_sum,news_id):
        s_node_emb = self.get_emb_from_id(s_nodes,news_id,dim_num=2)
        t_node_emb = self.get_emb_from_id(t_nodes,news_id,dim_num=2)

        beta = torch.sigmoid(self.bilinear(s_node_emb, t_node_emb)).squeeze(-1) # (batch) Equation-11
        delta_e_pred = beta / torch.pow(Variable(e_times)+1e-5, self.theta) * Variable(node_sum) * \
                       (self.zeta * torch.pow(Variable(node_sum-1), self.gamma)) # Equation-10

        return delta_e_pred

    def local_loss(self, s_nodes, t_nodes, e_times,
                   s_h_nodes, s_h_times, s_h_time_mask,
                   t_h_nodes, t_h_times, t_h_time_mask,
                   neg_s_node,neg_t_node,news_id):

        p_lambdas, n_lambdas_s = self.local_forward(s_nodes, t_nodes, e_times,
                                                                 s_h_nodes, s_h_times, s_h_time_mask,
                                                                 t_h_nodes, t_h_times, t_h_time_mask,
                                                                 neg_s_node, neg_t_node,news_id)

        aaa  =  - torch.log(p_lambdas.sigmoid() + 1e-5)
        bbb = - torch.log(n_lambdas_s.neg().sigmoid() + 1e-5).sum(dim=1)
        # ccc = - torch.log(n_lambdas_t.neg().sigmoid() + 1e-5).sum(dim=1)
        loss =  aaa + bbb #+ ccc # remove the history of t_node, i.e., ccc

        return loss

    def global_loss(self,s_nodes, t_nodes, e_times, delta_e_true, delta_n_true, node_sum, edge_last_time_sum,news_id):
        delta_e_pred = self.global_forward(s_nodes, t_nodes, e_times, delta_n_true, node_sum, edge_last_time_sum,news_id)
        criterion = torch.nn.MSELoss()
        loss = criterion(torch.log(delta_e_pred + 1e-5), torch.log(Variable(delta_e_true) + 1e-5))
        # loss = ((delta_e_pred - Variable(delta_e_true))**2).mean(dim=-1)
        return loss

    def veracity_predict(self, news_id):

        first_node_id = self.graph_data_dict[news_id].first_node
        first_node_emb = self.get_emb_from_id(first_node_id,news_id,dim_num=1).view(1,-1)# (1, emb_dim)

        all_node_id = self.graph_data_dict[news_id].node_list
        all_node_emb = self.get_emb_from_id(all_node_id,news_id,dim_num=2)

        all_node_emb_mean = torch.mean(all_node_emb,dim=0).view(1,-1)# (1, emb_dim)
        all_node_emb_max_tpl = torch.max(all_node_emb, dim=0)
        all_node_emb_max = all_node_emb_max_tpl.values.view(1, -1)  # (1, emb_dim)
        all_node_emb_pool = torch.cat([all_node_emb_mean, all_node_emb_max, first_node_emb], dim=1)

        aggre_emb = self.leakyrelu(self.dropout_layer(self.aggre_emb(all_node_emb_pool)))
        output = F.log_softmax(self.node_emb_output(aggre_emb), dim=1)

        return output

    def veracity_loss(self, news_id):
        output = self.veracity_predict(news_id)

        criterion = torch.nn.CrossEntropyLoss()
        label = self.labels[news_id]
        y = torch.tensor(to_label(label))
        vera_loss = criterion(output, y)
        return vera_loss

    def update(self, s_nodes, t_nodes, e_times,
               s_h_nodes, s_h_times, s_h_time_mask,
               t_h_nodes, t_h_times, t_h_time_mask,
               neg_s_node,neg_t_node,
               delta_e_true, delta_n_true, node_sum, edge_last_time_sum,
               news_id):

        self.opt.zero_grad()

        ## XXX ##
        max_d_time = self.max_d_time_dict[news_id]
        e_times = e_times/max_d_time
        s_h_times = s_h_times/max_d_time
        t_h_times = t_h_times/max_d_time

        local_loss = self.local_loss(s_nodes, t_nodes, e_times,
                                     s_h_nodes, s_h_times, s_h_time_mask,
                                     t_h_nodes, t_h_times, t_h_time_mask,
                                     neg_s_node, neg_t_node,news_id)

        global_loss = self.global_loss(s_nodes, t_nodes, e_times,
                                       delta_e_true, delta_n_true, node_sum, edge_last_time_sum,news_id)
        vera_loss = self.veracity_loss(news_id)

        weighted_local_loss = self.epsilon1 * local_loss.sum()
        weighted_global_loss = self.epsilon2 * global_loss.sum()
        weighted_vera_loss = self.epsilon * vera_loss

        loss = weighted_local_loss + weighted_vera_loss + weighted_global_loss

        loss.backward()
        self.opt.step()

        return loss, weighted_local_loss, weighted_global_loss, weighted_vera_loss

    def train_func(self):
        i_batch = 0
        news_id_list = list(self.graph_data_dict.keys())
        print('training...')
        for epoch in range(self.epochs):
            self.train()
            random.shuffle(news_id_list)
            for num, news_id in enumerate(news_id_list):
                if news_id not in self.train_ids:
                    continue
                # print('news_id: ',news_id)
                ## for each graph
                graph_data = self.graph_data_dict[news_id]
                loader = DataLoader(graph_data, batch_size=self.batch_size, shuffle=True, num_workers=10)

                total_batch_loss = 0
                total_batch_local_loss = 0
                total_batch_global_loss = 0
                total_batch_vera_loss = 0
                for _, sample_batched in enumerate(loader):
                    # i_batch += 1
                    # if i_batch % 1000 == 0 and i_batch != 0:
                    #     sys.stdout.write('\r\n' + str(i_batch * self.batch_size)
                    #                      + '\tloss: ' + str(self.loss.cpu().numpy() / (self.batch_size * i_batch))
                    #                      + '\tmicro_loss: ' + str(self.micro_loss.cpu().numpy() / (self.batch_size * i_batch))
                    #                      + '\tmacro_loss: ' + str(self.macro_loss.cpu().numpy() / (self.batch_size * i_batch))
                    #                      + '\r\ndelta_s:' + str(self.delta_s.mean().cpu().data.numpy())
                    #                      + '\tdelta_t:' + str(self.delta_t.mean().cpu().data.numpy())
                    #                      + '\tglobal_attention:' + str(self.global_attention.mean(dim=0).cpu().data.numpy())
                    #                      + '\r\nzeta:' + str(self.zeta.mean().cpu().data.numpy())
                    #                      + '\tgamma:' + str(self.gamma.mean().cpu().data.numpy())
                    #                      + '\ttheta:' + str(self.theta.mean().cpu().data.numpy())
                    #     )
                    #     sys.stdout.flush()

                    # batch_loss, batch_local_loss, batch_global_loss, batch_vera_loss = \
                    #     self.update(sample_batched['source_node'].type(LType),
                    #                 sample_batched['target_node'].type(LType),
                    #                 sample_batched['event_time'].type(FType),
                    #                 sample_batched['s_history_nodes'].type(LType),
                    #                 sample_batched['s_history_times'].type(FType),
                    #                 sample_batched['s_history_masks'].type(FType),
                    #                 sample_batched['t_history_nodes'].type(LType),
                    #                 sample_batched['t_history_times'].type(FType),
                    #                 sample_batched['t_history_masks'].type(FType),
                    #                 sample_batched['neg_s_nodes'].type(LType),
                    #                 sample_batched['neg_t_nodes'].type(LType),
                    #                 sample_batched['delta_e_true'].type(FType),
                    #                 sample_batched['delta_n_true'].type(FType),
                    #                 sample_batched['node_sum'].type(FType),
                    #                 sample_batched['edge_last_time_sum'].type(FType),
                    #                 news_id
                    #                 )

                    #str
                    batch_loss, batch_local_loss, batch_global_loss, batch_vera_loss = \
                        self.update(sample_batched['source_node'],
                                    sample_batched['target_node'],
                                    sample_batched['event_time'].type(FType),
                                    sample_batched['s_history_nodes'],
                                    sample_batched['s_history_times'].type(FType),
                                    sample_batched['s_history_masks'].type(FType),
                                    sample_batched['t_history_nodes'],
                                    sample_batched['t_history_times'].type(FType),
                                    sample_batched['t_history_masks'].type(FType),
                                    sample_batched['neg_s_nodes'],
                                    sample_batched['neg_t_nodes'],
                                    sample_batched['delta_e_true'].type(FType),
                                    sample_batched['delta_n_true'].type(FType),
                                    sample_batched['node_sum'].type(FType),
                                    sample_batched['edge_last_time_sum'].type(FType),
                                    news_id
                                    )

                    total_batch_loss += batch_loss.detach().numpy()
                    total_batch_local_loss += batch_local_loss.detach().numpy()
                    total_batch_global_loss += batch_global_loss.detach().numpy()
                    total_batch_vera_loss += batch_vera_loss.detach().numpy()

                if num %10 == 0:
                    sys.stdout.write('\r\nepoch-' + str(epoch) +  ' num-' + str(num) + ' news_id-' + str(news_id) +
                                     ': avg loss = ' + str(total_batch_loss / len(graph_data)) +
                                     ', avg local loss = ' + str(total_batch_local_loss / len(graph_data)) +
                                     ', avg global loss = ' + str(total_batch_global_loss / len(graph_data)) +
                                     ', avg veracity loss = ' + str(total_batch_vera_loss / len(graph_data)) +
                                     '\n')
                    sys.stdout.flush()

            if epoch % 1 == 0:# and epoch != 0:
                self.eval()

                train_acc, train_correct, train_n_samples = self.eval_func(self.train_ids)
                print('--train accuracy: {:.4f}, correct {} out of {}'.format(train_acc, train_correct, train_n_samples))

                val_acc, val_correct, val_n_samples = self.eval_func(self.val_ids)
                print('--validation accuracy: {:.4f}, correct {} out of {}'.format(val_acc, val_correct, val_n_samples))

                test_acc, test_correct, test_n_samples = self.eval_func(self.test_ids)
                print('--test accuracy: {:.4f}, correct {} out of {}'.format(test_acc, test_correct, test_n_samples))

    def eval_func(self, news_id_consider):

        correct = 0
        n_samples = len(news_id_consider)
        samples_per_label = np.zeros(self.output_dim)
        pred_per_label = np.zeros(self.output_dim)
        correct_per_label = np.zeros(self.output_dim)

        news_id_list = list(self.graph_data_dict.keys())

        with torch.no_grad():
            for _, news_id in enumerate(news_id_list):
                if news_id not in news_id_consider:
                    continue

                output = self.veracity_predict(news_id)
                _, pred = output.max(dim=1)
                label = self.labels[news_id]
                y = torch.tensor(to_label(label))
                correct += float(pred.eq(y).sum().item())

                ## buggy
                # for i in range(self.output_dim):
                #     batch_i = y[i]
                #     pred_i = pred.eq(i)
                #     samples_per_label[i] += batch_i.sum().item()
                #     pred_per_label[i] += pred_i.sum().item()
                #     correct_per_label[i] += (batch_i * pred_i).sum().item()

        # print('correct',correct)
        # print('n_samples',n_samples)

        acc = correct / n_samples

        # acc_per_label = correct_per_label / samples_per_label
        # rec_per_label = correct_per_label / pred_per_label
        # for i in range(self.output_dim):
        #     print("Accuracy_{}".format(i), acc_per_label[i])
        #     print("Recall_{}".format(i), rec_per_label[i])

        return  acc, correct, n_samples

    # def save_node_embeddings(self, path):
    #     if torch.cuda.is_available():
    #         embeddings = self.node_emb.cpu().data.numpy()
    #     else:
    #         embeddings = self.node_emb.data.numpy()
    #     ## XXX ##
    #     # for count in self.node_dim:
    #     #     writer = open(path, 'w')
    #     #     writer.write('%d %d\n' % (self.node_dim[count], self.emb_size))
    #     #     for n_idx in range(self.node_dim[count]):
    #     #         writer.write(' '.join(str(d) for d in embeddings[n_idx]) + '\n')
    #     #     writer.close()
    #     #     return embeddings
    #
    #     writer = open(path, 'w')
    #     writer.write('%d %d\n' % (self.node_dim, self.emb_size))
    #     for n_idx in range(self.node_dim):
    #         writer.write(' '.join(str(d) for d in embeddings[n_idx]) + '\n')
    #     writer.close()


if __name__ == '__main__':
    print(time.asctime(time.localtime(time.time())))
    parameters_dict = {
        'file_path': '../rumor_detection_acl2017/twitter15/',
        'save_graph_path':'../rumor_detection_acl2017/twitter15/graph_obj/',
        # 'cl_label_data': 'data/twitter15/label.txt',
        # 'nr_data': 'data/twitter15/twitter15_network_reconstruction_0.01edge.txt',
        'save_embedding_path': '../checkpoints/twitter15/',
        'epoch_num': 1000,
        'batch_size': 32,
        'emb_size': 64,
        # 'gat_hidden_size':32,
        'learning_rate': 1e-4,
        'neg_size': 5,
        'hist_len': 3,
        'directed': False,
        'save_step': 50,
        'optimization': 'Adam',#SGD: NaN in the 247-graph
        'tlp_flag':False,
        'trend_prediction':False,
        'epsilon1': 1, # local loss
        'epsilon2': 1,# global loss
        'epsilon':10}
    print ('parameters: \r\n{}'.format(parameters_dict))

    if not os.path.exists(parameters_dict['save_graph_path']):
        os.mkdir(parameters_dict['save_graph_path'])
    if not os.path.exists(parameters_dict['save_embedding_path']):
        os.mkdir(parameters_dict['save_embedding_path'])

    mmdne = MMDNE(file_path=parameters_dict['file_path'],
                  save_graph_path = parameters_dict['save_graph_path'],
                  # graph_dict=graph_dict,
                  # cl_label_data=parameters_dict['cl_label_data'],
                  # nr_data=parameters_dict['nr_data'],
                  save_embedding_path=parameters_dict['save_embedding_path'],
                  save_step=parameters_dict['save_step'],
                  directed=parameters_dict['directed'],
                  epoch_num=parameters_dict['epoch_num'],
                  # gat_hidden_size=parameters_dict['gat_hidden_size'],
                  hist_len=parameters_dict['hist_len'],
                  emb_size=parameters_dict['emb_size'],
                  neg_size=parameters_dict['neg_size'],
                  learning_rate=parameters_dict['learning_rate'],
                  batch_size=parameters_dict['batch_size'],
                  optim=parameters_dict['optimization'],
                  tlp_flag=parameters_dict['tlp_flag'],
                  trend_prediction=parameters_dict['trend_prediction'],
                  epsilon1=parameters_dict['epsilon1'],
                  epsilon2=parameters_dict['epsilon2'],
                  epsilon=parameters_dict['epsilon'])
    with autograd.detect_anomaly():
        mmdne.train_func()

    print ('parameters: \r\n{}'.format(parameters_dict))


