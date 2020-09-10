import numpy as np
from datetime import datetime, timedelta
import os
import glob
import torch


def preprocess_sequences_to_fixed_len(seq_data, cap_len, features_dim):
    X = [x for x,y in seq_data]
    results = []
    count_oversampling = 0
    idx_removed = []
    for i, sequence in enumerate(X):
        if len(sequence.shape) >= 2:
            len_seq = sequence.shape[0]
            if len_seq < cap_len:
                indexes_oversampled = np.random.choice(len_seq, cap_len - len_seq)
                sequence = np.concatenate((sequence, sequence[indexes_oversampled, :]), 0)
                count_oversampling += 1
            else:
                sequence = sequence[:cap_len, :]
            results.append(sequence)
        else:
            idx_removed.append(i)
    results = np.array(results, dtype=float)
    print(f"Fixed-length preprocessing: lost {len(idx_removed)} sequences that were unit-sized, oversampled {count_oversampling} sequences")
    assert len(idx_removed) == (len(X) - results.shape[0])
    return results, idx_removed


def standardize_and_turn_tensor(seq_data_preprocessed, standardize = True):
    print(f"Shape of input seq data is ndarry of shape {seq_data_preprocessed.shape}")
    seq_tensor = torch.from_numpy(seq_data_preprocessed)
    means = seq_tensor.mean(dim=(0, 1), keepdim=True)
    stds = seq_tensor.std(dim=(0, 1), keepdim=True)
    return (seq_tensor - means) / stds if standardize else seq_tensor

def int_or_root(e):
    return e if e == 'ROOT' else int(e)

def parse_edge_line(line):
    orig, dest = line.split("->")
    orig_list = orig.split("'")
    dest_list = dest.split("'")

    tweet_in, tweet_out = int_or_root(orig_list[3]), int_or_root(dest_list[3])
    user_in, user_out = int_or_root(orig_list[1]), int_or_root(dest_list[1])

    time_in, time_out = float(orig_list[5]), float(dest_list[5])

    # ## MODIFICATIONS ##
    #if orig_list[1] != 'ROOT':
    # s_node = int(orig_list[1])  # source node
    # t_node = int(dest_list[1])  # target node
    # d_time = float(dest_list[5])
    # if s_node not in list_of_nodes:
    #     int_node_dict.update({s_node: count})
    #     list_of_nodes.append((s_node, 0))
    #     count += 1
    # if t_node not in list_of_nodes:
    #     int_node_dict.update({t_node: count})
    #     list_of_nodes.append(t_node, d_time)
    #     count += 1

    return tweet_in, tweet_out, user_in, user_out, time_in, time_out #, list_of_nodes


def get_root_id(tree_file_name):
    return int(os.path.splitext(os.path.basename(tree_file_name))[0])


def get_tree_file_names(datadir):
    return glob.glob(os.path.join(datadir, "tree", "*.txt"))


def one_hot_label(label):
    if label == "non-rumor":
        return np.array([[1, 0, 0, 0]])
    if label == "false":
        return np.array([[0, 1, 0, 0]])
    if label == "true":
        return np.array([[0, 0, 1, 0]])
    if label == "unverified":
        return np.array([[0, 0, 0, 1]])


def to_label(label):
    if label == "false":
        return np.array([0])
    if label == "true":
        return np.array([1])
    if label == "non-rumor":
        return np.array([2])
    if label == "unverified":
        return np.array([3])


def from_date_text_to_timestamp(datestr):
    year, month, day = map(int, datestr.split()[0].split("-"))
    return (datetime(year, month, day) - datetime(1970, 1, 1)) / timedelta(days=1)


def cap_sequences(sequential_dataset, cap_len):
    sequential_dataset = [elt for elt in sequential_dataset if len(elt[0].shape) > 1]
    sequential_dataset = [[elt[0][:cap_len, :], elt[1]] for elt in sequential_dataset if elt[0].size(0) >= cap_len]
    return sequential_dataset
