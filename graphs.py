#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:01:43 2020

@author: jonnycook
"""

import os
from utils import parse_edge_line

def get_labels(dumps_dir="rumor_detection_acl2017", dataset="twitter15"):
    
    labels = {}
    with open(os.path.join(os.path.join(dumps_dir, dataset), "label.txt")) as label_file:
        for line in label_file.readlines():
            label, tweet_id = line.split(":")
            labels[int(tweet_id)] = label
    return labels

def yield_tree_information(tree_file_name):

    with open(tree_file_name, "rt") as tree_file:
        for line in tree_file.readlines():
            if "ROOT" in line:
                continue
            tweet_in, tweet_out, user_in, user_out, time_in, time_out = parse_edge_line(line)

            dict_to_yield = {"node_in": list(user_in, time_in, tweet_in),
                             "node_out": list(user_out, time_out, tweet_out)
                             }
            yield dict_to_yield
            
def twitter_tree_iterator(dumps_dir="rumor_detection_acl2017", dataset="twitter15", text_features_extractor=None,need_labels=False):
    
    dataset_dir = os.path.join(dumps_dir, dataset)
    trees = get_tree_file_names(dataset_dir)

    for tree_file_name in trees:
         root_tweet_id = int(os.path.splitext(os.path.basename(tree_file_name))[0])
            
def generate_dynamic_graph(dumps_dir, dataset, timestamp):
    """
    :return: simple event graph, directed, unweighted, with edges created up to current timestamp
             dynamic graph, as a sequence of graph snapshots taken each time a new event node is added
    """
    nodes = {}
    edges = []
    dynamic_graph = []
    num_nodes = 0
    dataset_dir = os.path.join(dumps_dir, dataset)
    for data_point in twitter_tree_iterator():
        if data_point['time_out'] < timestamp:
            node_in, node_out = data_point['node_in'], data_point['node_out']
            if node_in not in nodes:
                nodes[node_in] = num_nodes
                num_nodes += 1
            if node_out not in nodes:
                nodes[node_out] = num_nodes
                num_nodes += 1
            edges.append([nodes[node_in], nodes[node_out]])
            dynamic_graph.append(edges)

    return edges, dynamic_graph

def generate_datum(dynamic_graph):
    """
    :return: datum of form (claim, c; list of engagements, S; dynamic graph, G)
    """
    claim = dynamic_graph[0][0][0]
    engagements = dynamic_graph[-1][1:]
    datum = [claim, engagements, dynamic_graph]
    
    return datum



    
