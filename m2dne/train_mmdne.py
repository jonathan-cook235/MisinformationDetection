from __future__ import division
import sys
sys.path.append("../")

import matplotlib.pyplot as plt
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
from mmdne_model import MMDNE,FType,LType


parser = argparse.ArgumentParser(description='Train the TPP network.')
parser.add_argument('--train_mode', default='True',
                    help='Train a model or evaluate')
parser.add_argument('--dataset', choices=["twitter15", "twitter16"],
                    help='Training dataset', default="twitter15")
parser.add_argument('--optimizer', choices=["Adam", "SGD"],
                    help='optimizer', default="Adam")
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    help='learning rate')
parser.add_argument('--l2', default=1e-2, type=float,
                    help='L2 regularization')
parser.add_argument('--epoch_num', default=100, type=int,
                    help='Number of epochs')
parser.add_argument('--save_epochs', default=10, type=int,
                    help='Save every some number of epochs')
parser.add_argument('--batch_size', default=512, type=int,
                    help='Batch_size')
parser.add_argument('--emb_size', default=128, type=int,
                    help='embedding_size')#64
parser.add_argument('--only_binary', action='store_true',
                    help='Reduces the problem to binary classification')
parser.add_argument('--neg_size', default=5, type=int,
                    help='negative sample number')
parser.add_argument('--hist_len', default=5, type=int,
                    help='history node number')
parser.add_argument('--standardize', action='store_true',
                    help='Standardize features')
parser.add_argument('--features', choices=["all", "text_only", "user_only"],
                    help='Features to consider', default="all")
parser.add_argument('--time_cutoff',
                    help='Time cutoff in mins', default="None")
parser.add_argument('--seed', default=64, type=int,
                    help='Seed for train/val/test split')
parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout probability')
parser.add_argument('--epsilon', type=float, default=10, help='veracity loss co-efficient')
parser.add_argument('--epsilon1', type=float, default=1, help='local loss co-efficient')
parser.add_argument('--epsilon2', type=float, default=0.1, help='global loss co-efficient')
parser.add_argument('--enable_cuda', type=bool, default=True,
                    help='whether to use gpu')
parser.add_argument('--gpu', type=int, default=1, help='Idx for the gpu to use')
parser.add_argument('--backprop_every', type=int, default=50,
                    help='Every how many batches to backprop')
args = parser.parse_args()
print(args)

USE_GPU = args.enable_cuda and torch.cuda.is_available()

device_string = 'cuda:{}'.format(args.gpu) if USE_GPU else 'cpu'
device = torch.device(device_string)
print('Running model on device: ', device)

model_name = 'MMDNE_'+ args.dataset
data_file_path = '../rumor_detection_acl2017/' + args.dataset
save_graph_path = data_file_path + '/graph_obj'
save_model_path = '../checkpoints/' + args.dataset

if not os.path.exists(save_graph_path):
    os.mkdir(save_graph_path)
if not os.path.exists(save_model_path):
    os.mkdir(save_model_path)


def forward_per_graph(news_id):
    # graph_batch_num = 0
    graph_loss = 0
    graph_local_loss = 0
    graph_global_loss = 0
    graph_vera_loss = 0

    graph_data = mmdne.graph_data_dict[news_id]
    num_datapoints = len(graph_data)
    loader = DataLoader(graph_data, batch_size=num_datapoints, shuffle=True, num_workers=10)
    for iii, sample_batched in enumerate(loader):
        # str
        # print(iii)
        optim.zero_grad()
        batch_loss, batch_local_loss, batch_global_loss, batch_vera_loss, delta_e_pred, event_time, delta_e_true = \
            mmdne.update(sample_batched['source_node'],
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
                         # sample_batched['delta_n_true'].type(FType),
                         sample_batched['node_sum'].type(FType),
                         # sample_batched['edge_last_time_sum'].type(FType),
                         news_id
                         )


        # batch_loss.backward()
        # mmdne.opt.step()

        graph_loss += batch_loss
        graph_local_loss += batch_local_loss
        graph_global_loss += batch_global_loss
        graph_vera_loss += batch_vera_loss

    return graph_loss, graph_local_loss, graph_global_loss, graph_vera_loss, num_datapoints, delta_e_pred, event_time, delta_e_true

def train_func(mmdne, optim):
    total_graph_loss = 0
    total_graph_local_loss = 0
    total_graph_global_loss = 0
    total_graph_vera_loss = 0
    total_num_datapoints = 0

    news_id_list = list(mmdne.graph_data_dict.keys())
    train_news_id_list = [news_id for news_id in news_id_list if news_id in mmdne.train_ids]

    best_eval_acc = 0
    cum_times = 0

    print('training...')
    graph_num = 0
    # with mp.Pool(os.cpu_count()) as pool:
    # Unsupported: autograd does not support crossing process boundaries
    for epoch in range(mmdne.epochs):
        mmdne.train()
        random.shuffle(train_news_id_list)
        for num, news_id in enumerate(train_news_id_list):
            # batch_news_id_list = train_news_id_list[num*args.backprop_every:(num+1)*args.backprop_every]
            # graph_batch_loss, graph_batch_local_loss, graph_batch_global_loss, graph_batch_vera_loss = \
            #     pool.map(forward_per_graph, batch_news_id_list)
            graph_num += 1
            graph_loss, graph_local_loss, graph_global_loss, graph_vera_loss, num_datapoints, delta_e_pred, event_time, delta_e_true = \
                forward_per_graph(news_id)

            total_graph_loss += graph_loss
            total_graph_local_loss += graph_local_loss
            total_graph_global_loss += graph_global_loss
            total_graph_vera_loss += graph_vera_loss
            total_num_datapoints += num_datapoints

            if graph_num % mmdne.backprop_every == 0: # itrate 10 grahs
                # back-propagation
                total_graph_loss.backward()
                optim.step()

                print('epoch-{} graph_batch_num-{}: '
                      'avg loss = {:.5f}, '
                      'avg local loss = {:.5f}, '
                      'avg global loss = {:.5f}, '
                      'avg veracity loss = {:.5f}'.format(epoch, graph_num,
                                                  total_graph_loss.detach().cpu().numpy() / mmdne.backprop_every,
                                                  total_graph_local_loss.detach().cpu().numpy() / mmdne.backprop_every,
                                                  total_graph_global_loss.detach().cpu().numpy() / mmdne.backprop_every,
                                                  total_graph_vera_loss.detach().cpu().numpy() / mmdne.backprop_every))

                total_graph_loss = 0
                total_graph_local_loss = 0
                total_graph_global_loss = 0
                total_graph_vera_loss = 0
                total_num_datapoints = 0

        if epoch % args.save_epochs == 0:
            state_dict = mmdne.state_dict()
            torch.save(state_dict, os.path.join(mmdne.save_model_path, mmdne.model_name))

        if epoch % 1 == 0:# and epoch != 0:

            ## Evaluate
            best_eval_acc, cum_times = eval_func(mmdne, best_eval_acc)

        ## early stopping
        if cum_times >= args.patience:
            break

def eval_func(mmdne, best_eval_acc=0, cum_times = 0):
    mmdne.eval()
    print('#############################################')
    val_loss, val_local_loss, val_global_loss, val_vera_loss, val_num_datapoints = \
        eval_temporal_pred(mmdne, mmdne.val_ids)
    print('\tValdation: avg loss = {:.5f}, '
          'avg local loss = {:.5f}, '
          'avg global loss = {:.5f}, '
          'avg veracity loss = {:.5f}\n'.format(val_loss / len(mmdne.val_ids),
                                                val_local_loss / len(mmdne.val_ids),
                                                val_global_loss / len(mmdne.val_ids),
                                                val_vera_loss / len(mmdne.val_ids)))
    test_loss, test_local_loss, test_global_loss, test_vera_loss, test_num_datapoints = \
        eval_temporal_pred(mmdne, mmdne.test_ids)
    print('\tTest: avg loss = {:.5f}, '
          'avg local loss = {:.5f}, '
          'avg global loss = {:.5f}, '
          'avg veracity loss = {:.5f}\n'.format(test_loss / len(mmdne.test_ids),
                                                test_local_loss / len(mmdne.test_ids),
                                                test_global_loss / len(mmdne.test_ids),
                                                test_vera_loss / len(mmdne.test_ids)))
    # Evaluate veracity classification
    train_acc, train_correct, train_n_graphs = eval_veracity_func(mmdne, mmdne.train_ids)
    print('--train accuracy: {:.5f}, correct {} out of {}'.format(train_acc, train_correct, train_n_graphs))
    val_acc, val_correct, val_n_graphs = eval_veracity_func(mmdne, mmdne.val_ids)
    print('--validation accuracy: {:.5f}, correct {} out of {}'.format(val_acc, val_correct, val_n_graphs))
    test_acc, test_correct, test_n_graphs = eval_veracity_func(mmdne, mmdne.test_ids)
    print('--test accuracy: {:.5f}, correct {} out of {}\n'.format(test_acc, test_correct, test_n_graphs))
    print('#############################################')

    if val_acc > best_eval_acc:
        best_eval_acc = val_acc
        cum_times = 0

        # save the best setting
        state_dict = mmdne.state_dict()
        torch.save(state_dict, os.path.join(mmdne.save_model_path, mmdne.model_name))

    else:
        cum_times += 1

    return best_eval_acc, cum_times


def eval_veracity_func(mmdne, news_id_consider):

    correct = 0
    n_graphs = len(news_id_consider)
    samples_per_label = np.zeros(mmdne.output_dim)
    pred_per_label = np.zeros(mmdne.output_dim)
    correct_per_label = np.zeros(mmdne.output_dim)

    news_id_list = list(mmdne.graph_data_dict.keys())

    with torch.no_grad():
        for _, news_id in enumerate(news_id_list):
            if news_id not in news_id_consider:
                continue

            output = mmdne.veracity_predict(news_id)
            _, pred = output.max(dim=1)
            label = mmdne.labels[news_id]
            y = torch.tensor(to_label(label)).to(device)
            correct += float(pred.eq(y).sum().item())

            # ## buggy
            # for i in range(mmdne.output_dim):
            #     batch_i = y[i]
            #     pred_i = pred.eq(i)
            #     samples_per_label[i] += batch_i.sum().item()
            #     pred_per_label[i] += pred_i.sum().item()
            #     correct_per_label[i] += (batch_i * pred_i).sum().item()

    # print('correct',correct)
    # print('n_graphs',n_graphs)

    acc = correct / n_graphs

    # acc_per_label = correct_per_label / samples_per_label
    # rec_per_label = correct_per_label / pred_per_label
    # for i in range(mmdne.output_dim):
    #     print("Accuracy_{}".format(i), acc_per_label[i])
    #     print("Recall_{}".format(i), rec_per_label[i])

    return  acc, correct, n_graphs

def eval_temporal_pred(mmdne, news_id_consider):
    news_id_list = list(mmdne.graph_data_dict.keys())
    eval_news_id_list = [news_id for news_id in news_id_list if news_id in news_id_consider]

    total_graph_loss = 0
    total_graph_local_loss = 0
    total_graph_global_loss = 0
    total_graph_vera_loss = 0
    total_num_datapoints = 0

    with torch.no_grad():
        for _, news_id in enumerate(eval_news_id_list):
            graph_loss, graph_local_loss, graph_global_loss, graph_vera_loss, num_datapoints, delta_e_pred, event_time, delta_e_true = \
                forward_per_graph(news_id)

            # eval_forecasting(delta_e_pred, event_time, delta_e_true)

            total_graph_loss += graph_loss.detach().cpu().numpy()
            total_graph_local_loss += graph_local_loss.detach().cpu().numpy()
            total_graph_global_loss += graph_global_loss.detach().cpu().numpy()
            total_graph_vera_loss += graph_vera_loss.detach().cpu().numpy()
            total_num_datapoints += num_datapoints

    return total_graph_loss, total_graph_local_loss, total_graph_global_loss, total_graph_vera_loss, total_num_datapoints

def eval_forecasting(delta_e_pred, event_time, delta_e_true):
    timestamps = event_time
    for i in delta_e_pred:
        if not delta_e_pred[0]:
            delta_e_pred[i] += delta_e_pred[i-1]
    for i in delta_e_true:
        if not delta_e_true[0]:
            delta_e_true[i] += delta_e_true[i-1]
    plt.plot(timestamps, delta_e_pred)
    plt.plot(timestamps, delta_e_true)
    plt.plot(timestamps, delta_e_pred-delta_e_true)
    plt.show

if __name__ == '__main__':

    mmdne = MMDNE(file_path=data_file_path,
                  save_graph_path = save_graph_path,
                  model_name=model_name,
                  save_model_path=save_model_path,
                  epoch_num=args.epoch_num,
                  hist_len=args.hist_len,
                  emb_size=args.emb_size,
                  neg_size=args.neg_size,
                  # learning_rate=args.learning_rate,
                  batch_size=args.batch_size,
                  # optim=args.optimizer,
                  epsilon1=args.epsilon1,
                  epsilon2=args.epsilon2,
                  epsilon=args.epsilon,
                  backprop_every = args.backprop_every,
                  device=device,
                  gpu=args.gpu).to(device)

    if args.optimizer == 'SGD':
        optim = SGD(lr=args.learning_rate, momentum=0.9, weight_decay=args.l2, params=mmdne.para_to_opt)
    elif args.optimizer == 'Adam':
        optim = Adam(lr=args.learning_rate, weight_decay=args.l2, params=mmdne.para_to_opt)#0.01


    if args.train_mode == 'True':
        # with autograd.detect_anomaly():
        train_func(mmdne, optim)
        state_dict = mmdne.state_dict()
        torch.save(state_dict, os.path.join(mmdne.save_model_path, mmdne.model_name))

    else:
        print('Evaluating...')
        output_model_file = os.path.join(mmdne.save_model_path, mmdne.model_name)
        # if  device.type == 'cpu':
        state_dict = torch.load(output_model_file,map_location = device)
        mmdne.load_state_dict(state_dict)

        eval_func(mmdne)




