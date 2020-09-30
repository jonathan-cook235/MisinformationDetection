from __future__ import division
import sys
sys.path.append("../")

from torch.utils.data import Dataset
import numpy as np
import sys
import random
import copy
import glob
import os

from utils import util

np.random.seed(1)


def int_or_root(e):
    #
    #     e = 0
    # else:
    #     e  = int(e)
    return e if e == 'ROOT' else int(e)

class DataHelper(Dataset):
    def __init__(self, file_path, neg_size, hist_len,
                 directed=False, transform=None, tlp_flag=False, trend_pred_flag=False):
        # tweet_fts: dict[tweet_id:int -> tweet - features: np array]
        # user_fts: dict[user_id:int -> user - features: np array]
        self.node2hist = dict()
        self.neg_size = neg_size
        self.hist_len = hist_len
        self.directed = directed
        self.transform = transform

        self.max_d_time = -sys.maxsize  # Time interval [0, T]

        self.NEG_SAMPLING_POWER = 0.75
        self.neg_table_size = int(1e5)#1e8

        self.node_time_nodes = dict()
        self.node_set = set()
        self.degrees = dict()
        self.edge_list = []
        self.node_numbers = []

        self.node_rate = {}
        self.edge_rate = {}
        self.node_sum = {}
        self.edge_sum = {}
        self.time_stamp = []
        self.time_edges_dict = {}
        self.time_nodes_dict = {}

        self.user_tweet_dict = {'0':'0'}#str(user_id):str(tweet_id)
        # self.user_tweet_dict = {0: 0}  # user_id:tweet_id

        with open(file_path, 'rt') as infile:

            # node_int  = 0
            time_shift = 0
            for line in infile.readlines():
                tweet_in, tweet_out, user_in, user_out, time_in, time_out = util.parse_edge_line(line)
                if time_out < 0 and time_shift == 0:
                    # if buggy dataset, and we haven't found the time_shift yet
                    time_shift = -time_out
                if user_in == 'ROOT':
                    # node_int += 1
                    self.first_node = user_out
                    self.user_tweet_dict[user_out] = tweet_out
                    continue # exclude the root marker


                # if user_in == 760576918085337088 or

                s_node = user_in
                # node_int += 1
                # s_node = node_int
                if s_node in self.user_tweet_dict:
                    if  tweet_in != self.user_tweet_dict[s_node]:
                        print('s-user have multiple tweets')
                else:
                    self.user_tweet_dict[s_node] = tweet_in

                t_node = user_out
                # node_int += 1
                # t_node = node_int
                if t_node in self.user_tweet_dict:
                    if tweet_out != self.user_tweet_dict[t_node]:
                        print('t-user have multiple tweets')
                else:
                    self.user_tweet_dict[t_node] = tweet_out

                # TICK-Check This: time_out  or (time_out - time_in). Refer to d_time in  local_forward
                time_out += time_shift  # fix buggy dataset
                assert time_out >= 0
                # d_time = np.abs(time_out - time_in)
                # assert d_time >= 0
                d_time = time_out

                # s_node = user_tweet_dict[user_in] ## node_index --> node_int ##
                # t_node = user_tweet_dict[user_out] ## node_index --> node_int  ##

                self.node_set.update([s_node, t_node])  # node set
                self.edge_list.append((s_node,t_node,d_time))  # edge list
                if not directed:
                    self.edge_list.append((t_node,s_node,d_time))

                # each s_node should have a sequence of history nodes, which means s_node occurs more than once.
                if s_node not in self.node2hist:
                    self.node2hist[s_node] = list()
                self.node2hist[s_node].append((t_node, d_time))
                if not directed:
                    # TICK-Check This: if this is directed, we will lose t_node in the self.node2hist
                    # Can implement direct graphs by removing t-node from local_forward in train_mmdne.py (Equation-2)
                    if t_node not in self.node2hist:
                        self.node2hist[t_node] = list()
                    self.node2hist[t_node].append((s_node, d_time))

                if s_node not in self.node_time_nodes:
                    self.node_time_nodes[s_node] = dict()
                if d_time not in self.node_time_nodes[s_node]:
                    self.node_time_nodes[s_node][d_time] = list()
                self.node_time_nodes[s_node][d_time].append(t_node)
                if not directed:  # undirected
                    if t_node not in self.node_time_nodes:
                        self.node_time_nodes[t_node] = dict()
                    if d_time not in self.node_time_nodes[t_node]:
                        self.node_time_nodes[t_node][d_time] = list()
                    self.node_time_nodes[t_node][d_time].append(s_node)

                if s_node not in self.degrees:
                    self.degrees[s_node] = 0
                if t_node not in self.degrees:
                    self.degrees[t_node] = 0
                self.degrees[s_node] += 1
                self.degrees[t_node] += 1

                ## XXX ## avoid the same timestamp
                if d_time > self.max_d_time:
                    self.max_d_time = d_time  # record the max time

                while d_time in self.time_stamp:
                    d_time = d_time + 1e-5 * np.random.randint(0,100)
                self.time_stamp.append(d_time)

                if d_time not in self.time_edges_dict:
                    self.time_edges_dict[d_time] = []
                self.time_edges_dict[d_time].append((s_node, t_node))
                if d_time not in self.time_nodes_dict:
                    self.time_nodes_dict[d_time] = []
                self.time_nodes_dict[d_time].append(s_node)
                self.time_nodes_dict[d_time].append(t_node)

        self.time_stamp = sorted(list(set(self.time_stamp)))  # !!! time from 0 to 1
        self.node_list = list(self.node_set)
        self.node_dim = len(self.node_set)  # number of nodes 28085

        for user_id in self.node_list:
            if user_id not in self.user_tweet_dict:
                print('Not find user-id {} in user_tweet_dict'.format(user_id))


        self.data_size = 0  # number of edges, undirected x2 = 473788
        for s in self.node2hist:
            hist = self.node2hist[s]
            hist = sorted(hist, key=lambda x: x[1])  # from past(0) to now(1)
            self.node2hist[s] = hist
            self.data_size += len(self.node2hist[s])

        self.max_nei_len = max(map(lambda x: len(x), self.node2hist.values()))  # 955
        # print ('\t#nodes: {}, #edges: {}, #time_stamp: {}'.
        #        format(self.node_dim,len(self.edge_list),len(self.time_stamp)))
        # print ('\tavg. degree: {}'.format(sum(self.degrees.values())/len(self.degrees)))
        # print ('\tmax neighbors length: {}'.format(self.max_nei_len))

        # resolving the problem of s_node being negative
        self.idx2source_id = {} #np.zeros((self.data_size,), dtype=np.int32)
        self.idx2target_id = {} #np.zeros((self.data_size,), dtype=np.int32)
        idx = 0
        for s_node in self.node2hist:
            hist = self.node2hist[s_node]
            # hist_node_int = [node_int for (node_int, timestamp) in hist]
            for t_idx in range(len(hist)):
            # for t_node_int in hist_node_int:
                self.idx2source_id[idx] = s_node
                self.idx2target_id[idx] = t_idx
                # self.idx2target_id[idx] = t_node_int
                idx += 1

        self.get_edge_rate()
        self.neg_table = ['0'] * self.neg_table_size#str
        # self.neg_table = np.zeros((self.neg_table_size,))
        self.init_neg_table() ## Check this: Time-consuming

    def get_edge_rate(self):
        for i in range(len(self.time_stamp)):
            current_nodes = []
            current_edges = []
            current_time_idx = i
            while current_time_idx >= 0:
                current_nodes += self.time_nodes_dict[self.time_stamp[current_time_idx]]
                current_edges += self.time_edges_dict[self.time_stamp[current_time_idx]]
                current_time_idx -= 1
            self.node_sum[self.time_stamp[i]] = len(set(current_nodes))
            self.edge_sum[self.time_stamp[i]] = len(current_edges)

        for i in range(len(self.time_stamp)):
            current_time_idx = i
            if current_time_idx == 0:  # time = 0, delta_node = node_sum[0]
                self.node_rate[self.time_stamp[current_time_idx]] = self.node_sum[self.time_stamp[current_time_idx]]
            else:
                self.node_rate[self.time_stamp[current_time_idx]] = \
                    self.node_sum[self.time_stamp[current_time_idx]] - self.node_sum[self.time_stamp[current_time_idx-1]]
            if current_time_idx == 0:
                self.edge_rate[self.time_stamp[current_time_idx]] = self.edge_sum[
                    self.time_stamp[current_time_idx]]
            else:
                self.edge_rate[self.time_stamp[current_time_idx]] = \
                    self.edge_sum[self.time_stamp[current_time_idx]] - self.edge_sum[
                        self.time_stamp[current_time_idx - 1]]

    def get_node_dim(self):
        return self.node_dim

    def get_max_d_time(self):
        return self.max_d_time

    def init_neg_table(self):
        tot_sum, cur_sum, por = 0., 0., 0.
        # for k in range(self.node_dim):
        for node_id in self.node_set:
            tot_sum += np.power(self.degrees[node_id], self.NEG_SAMPLING_POWER)

        n_id = 0
        for k in range(self.neg_table_size):
            if (k + 1.) / self.neg_table_size > por:
                node_id  = self.node_list[n_id] ## XXX ##
                cur_sum += np.power(self.degrees[node_id], self.NEG_SAMPLING_POWER)
                por = cur_sum / tot_sum
                n_id += 1
            # self.neg_table[k] = n_id - 1
            self.neg_table[k] = self.node_list[n_id - 1]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        # sampling via htne
        s_node = self.idx2source_id[idx]

        hist = self.node2hist[s_node]
        t_idx = self.idx2target_id[idx]
        t_node = hist[t_idx][0]
        e_time = hist[t_idx][1]
        if t_idx - self.hist_len < 0:
            s_his = self.node2hist[s_node][0:t_idx]
        else:
            s_his = self.node2hist[s_node][t_idx - self.hist_len:t_idx]

        # undirected: get the history neighbors for target node
        t_his_list = self.node2hist[t_node]
        s_idx = t_his_list.index((s_node, e_time))
        if s_idx - self.hist_len < 0:
            t_his = t_his_list[:s_idx]
        else:
            t_his = t_his_list[s_idx - self.hist_len:s_idx]

        s_his_nodes = ['0'] * self.hist_len #str
        # s_his_nodes = np.zeros((self.hist_len,))
        s_his_nodes[:len(s_his)] = [h[0] for h in s_his]
        s_his_times = np.zeros((self.hist_len,))
        s_his_times[:len(s_his)] = [h[1] for h in s_his]
        s_his_masks = np.zeros((self.hist_len,))
        s_his_masks[:len(s_his)] = 1.

        t_his_nodes = ['0'] * self.hist_len#str
        # t_his_nodes = np.zeros((self.hist_len,))
        t_his_nodes[:len(t_his)] = [h[0] for h in t_his]
        t_his_times = np.zeros((self.hist_len,))
        t_his_times[:len(t_his)] = [h[1] for h in t_his]
        t_his_masks = np.zeros((self.hist_len,))
        t_his_masks[:len(t_his)] = 1.

        # negative sampling
        neg_s_nodes = self.negative_sampling()
        neg_t_nodes = self.negative_sampling()

        time_idx = self.time_stamp.index(e_time)
        delta_e_true = self.edge_rate[e_time]
        delta_n_true = self.node_rate[e_time]
        node_sum = self.node_sum[e_time]
        if time_idx >= 1:
            edge_last_time_sum = self.edge_sum[self.time_stamp[time_idx-1]]
        else:
            edge_last_time_sum = self.edge_sum[self.time_stamp[time_idx]]

        sample = {
            'source_node': s_node,
            'target_node': t_node,
            'event_time': e_time,
            's_history_nodes': s_his_nodes,
            't_history_nodes': t_his_nodes,## corresponds to H^j(t) in Eq2 of the MMDNE paper
            's_history_times': s_his_times,
            't_history_times': t_his_times,
            's_history_masks': s_his_masks,
            't_history_masks': t_his_masks,
            'neg_s_nodes': neg_s_nodes,
            'neg_t_nodes': neg_t_nodes,
            'delta_e_true': delta_e_true,
            'delta_n_true': delta_n_true,
            'node_sum': node_sum,
            'edge_last_time_sum': edge_last_time_sum
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def negative_sampling(self):
        rand_idx = np.random.randint(0, self.neg_table_size, (self.neg_size,))
        # sampled_nodes = self.neg_table[rand_idx]
        sampled_nodes = [self.neg_table[idx] for idx in rand_idx] # str
        return sampled_nodes

