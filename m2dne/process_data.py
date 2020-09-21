import sys
sys.path.append("../")

import os
import glob
import numpy as np
import pickle
import time
import pandas as pd
from utils import util
from dataset_def import DataHelper

from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

Num_User_Features = 11
Num_Tweet_Features  = 768

# def get_root_id(tree_file_name):
#     return os.path.splitext(os.path.basename(tree_file_name))[0]

def to_label(label):
    if label == "false":
        return np.array([0])
    if label == "true":
        return np.array([1])
    if label == "non-rumor":
        return np.array([2])
    if label == "unverified":
        return np.array([3])

def load_labels(dataset_dir):
    """
    Returns:
        labels: dict[news_id:int -> label:int]
    """
    labels = {}
    with open(os.path.join(dataset_dir, "label.txt")) as label_file:
        for line in label_file.readlines():
            label, news_id = line.split(":")
            # labels[int(news_id)] = label
            labels[int(news_id)] = label
    return labels

def get_user_and_tweet_ids_in_train(trees_to_parse, train_ids):
    """ Returns sets of all the user ids and tweet ids that appear in train set """

    user_ids_in_train = set()
    tweet_ids_in_train = set()
    for tree_file_name in trees_to_parse:
        news_id = util.get_root_id(tree_file_name)
        if news_id in train_ids:
            with open(tree_file_name, "rt") as tree_file:
                int_node_dict = {}
                list_of_nodes = []
                count = 0
                for line in tree_file.readlines():
                    if "ROOT" in line:
                        continue
                    tweet_in, tweet_out, user_in, user_out, _, _ = util.parse_edge_line(line)
                    user_ids_in_train.add(user_in)  # user_ids_in_train may be bigger
                    user_ids_in_train.add(user_out)
                    tweet_ids_in_train.add(tweet_in)
                    tweet_ids_in_train.add(tweet_out)
    return user_ids_in_train, tweet_ids_in_train

def load_tweet_features(DATA_DIR = "../rumor_detection_acl2017"):
    """
    Returns:
       tweet_texts: dict[tweet_id:int -> dict[name feature -> feature]]
    """

    print('Load tweet features...')
    text_embeddings = np.load("../rumor_detection_acl2017/output_bert.npy")

    # with open(os.path.join(DATA_DIR, "tweet_features.txt")) as text_file:
    #     # first line contains column names
    #     tweet_feature_names = text_file.readline().rstrip('\n').split(';')

    tweet_features_id = pd.read_csv("../rumor_detection_acl2017/tweet_features.txt",
                                    delimiter=';',names=None).id
    tweet_features = {}
    for i, tweet_id in enumerate(tweet_features_id):
        tweet_features[int(tweet_id)] = {"embedding":text_embeddings[i]}

    return tweet_features

def load_user_features(DATA_DIR = "../rumor_detection_acl2017"):
    """
    Returns:
        user_features: dict[tweet_id:int -> dict[name feature -> feature]]
    """
    print('Load user features...')
    user_features = {}
    with open(os.path.join(DATA_DIR, "user_features.txt")) as text_file:
        # first line contains column names
        user_feature_names = text_file.readline().rstrip('\n').split(';')
        for line in text_file.readlines():
            features = line.rstrip('\n').split(";")
            user_features[int(features[0])] = {user_feature_names[i]: features[i]
                                                for i in range(1, len(features))}

    return user_features

def default_tweet_features():
    """ Return np array of default features sorted by alphabetic order """
    tweet_dict_defaults = {
        'embed': np.zeros((Num_Tweet_Features))
    }
    return np.array([val for key, val in
                    sorted(tweet_dict_defaults.items(), key=lambda x: x[0])]).reshape(-1)

def preprocess_tweet_features(tweet_features, tweet_ids_in_train):
    """ Preprocess all tweet features to transform dicts into fixed-sized array.

    Args:
        tweet_features: dict[tweet_id -> dict[name_feature -> feature]]
    Returns:
        defaultdict[tweet_id -> np.array(n_dim)]

    """
    print('Preprocess tweet features...')

    # new_tweet_features = {key: np.array([]) for key, val in tweet_features.items()}

    new_tweet_features = {key: np.array([key_val[1] for key_val in sorted(value.items(), key=lambda x: x[0])]).reshape(-1)
                        for key, value in tweet_features.items()}

    num_tweet_features = Num_Tweet_Features
    return defaultdict(default_tweet_features, new_tweet_features), num_tweet_features


def default_user_features():
    """ Return np array of default features sorted by alphabetic order """
    user_dict_defaults = {
        'created_at': 0,
        'favourites_count': 0,
        'followers_count': 0,
        'friends_count': 0,
        'geo_enabled': 0,
        'has_description': 0,
        'len_name': 0,
        'len_screen_name': 0,
        'listed_count': 0,
        'statuses_count': 0,
        'verified': 0
    } # Total 11 user features

    return np.array([val for key, val in
                     sorted(user_dict_defaults.items(), key=lambda x: x[0])])


def preprocess_user_features(user_features, user_ids_in_train, standardize_features=True):
    """ Preprocess all user features to transform dicts into fixed-sized array.

    Args:
        user_features: dict[user_id -> dict[name_feature -> feature]]
    Returns:
        defaultdict[user_id -> np.array(n_dim)]
    """

    # Available variables
    # id;
    # created_at;
    # description;
    # favourites_count;
    # followers_count;
    # friends_count;
    # geo_enabled;
    # listed_count;
    # location;
    # name;
    # screen_name;
    # statuses_count;
    # verified

    # Features we use:
    # created_at
    # favourites_count
    # followers_count
    # friends_count
    # geo_enabled
    # has_description
    # len_name
    # len_screen_name
    # listed_count
    # statuses_count
    # verified

    print('Preprocess user features...')
    for user_id, features in user_features.items():

        new_features = {}  # will contain the processed features of current user

        if "created_at" in features:
            new_features['created_at'] = \
                util.from_date_text_to_timestamp(features['created_at'])

        integer_features = [
            "favourites_count",
            "followers_count",
            "friends_count",
            "listed_count",
            "statuses_count",
        ]
        # print(features.keys())
        for int_feature in integer_features:
            new_features[int_feature] = float(features[int_feature])

        new_features["verified"] = float(features['verified'] == 'True')
        new_features["geo_enabled"] = float(features['geo_enabled'] == 'True')
        new_features['has_description'] = float(len(features['description']) > 0)
        new_features['len_name'] = float(len(features['name']))
        new_features['len_screen_name'] = float(len(features['screen_name']))

        user_features[user_id] = new_features

    num_user_features = len(new_features)
    user_features_train_only = {key: val for key, val in user_features.items() if key in user_ids_in_train}

    # Standardizing
    if standardize_features:
        for ft in [
            "created_at",
            "favourites_count",
            "followers_count",
            "friends_count",
            "listed_count",
            "statuses_count",
        ]:
            scaler = StandardScaler().fit(
                np.array([val[ft] for val in user_features_train_only.values()]).reshape(-1, 1)
            )

            # faster to do this way as we don't have to convert to np arrays
            mean, std = scaler.mean_[0], scaler.var_[0] ** (1 / 2)
            for key in user_features.keys():
                user_features[key][ft] = (user_features[key][ft] - mean) / std

            user_features_train_only = {key: val for key, val in user_features.items() if key in user_ids_in_train}

    # the default values of a dictionary
    # global user_dict_defaults
    # user_dict_defaults = {
    #     'created_at': np.median([elt["created_at"] for elt in user_features_train_only.values()]),
    #     'favourites_count': np.median([elt["favourites_count"] for elt in user_features_train_only.values()]),
    #     'followers_count': np.median([elt["followers_count"] for elt in user_features_train_only.values()]),
    #     'friends_count': np.median([elt["friends_count"] for elt in user_features_train_only.values()]),
    #     'geo_enabled': 0,
    #     'has_description': 0,
    #     'len_name': np.median([elt["len_name"] for elt in user_features_train_only.values()]),
    #     'len_screen_name': np.median([elt["len_screen_name"] for elt in user_features_train_only.values()]),
    #     'listed_count': np.median([elt["listed_count"] for elt in user_features_train_only.values()]),
    #     'statuses_count': np.median([elt["statuses_count"] for elt in user_features_train_only.values()]),
    #     'verified': 0
    # } # Total 11 user features

    #  user features: key=uid, value=dict[ftname:valueft]
    np_user_features = {key: np.array([key_val[1] for key_val in sorted(value.items(), key=lambda x: x[0])]) for
                        key, value in user_features.items()}

    return defaultdict(default_user_features, np_user_features), num_user_features


def get_user_tweet_fts(file_path, train_ids, trees_to_parse, only_user=False):
    user_file_path = os.path.join(file_path, 'processed_user_fts')
    if os.path.exists(user_file_path):
        num_user_features = Num_User_Features
        with open(user_file_path, 'rb') as f:
            preprocessed_user_fts = pickle.load(f)
    else:
        user_ids_in_train, tweet_ids_in_train = \
            get_user_and_tweet_ids_in_train(trees_to_parse, train_ids)
        user_features = load_user_features()
        preprocessed_user_fts, num_user_features = \
            preprocess_user_features(user_features, user_ids_in_train,
                                     standardize_features=True)

        with open(user_file_path, 'wb') as f:
            pickle.dump(preprocessed_user_fts, f, pickle.HIGHEST_PROTOCOL)

    if only_user == True:
        preprocessed_tweet_fts = {}
    else:
        tweet_file_path = os.path.join(file_path, 'processed_tweet_fts')
        if os.path.exists(tweet_file_path):
            num_tweet_features = Num_Tweet_Features
            with open(tweet_file_path, 'rb') as f:
                preprocessed_tweet_fts = pickle.load(f)
        else:
            tweet_ids_in_train, tweet_ids_in_train = \
                get_user_and_tweet_ids_in_train(trees_to_parse, train_ids)
            tweet_features = load_tweet_features()
            preprocessed_tweet_fts, num_tweet_features = \
                preprocess_tweet_features(tweet_features, tweet_ids_in_train)

            with open(tweet_file_path, 'wb') as f:
                pickle.dump(preprocessed_tweet_fts, f, pickle.HIGHEST_PROTOCOL)

    return preprocessed_tweet_fts, preprocessed_user_fts, num_tweet_features, num_user_features

def read_graph_obj(directed, news_id, hist_len, neg_size, save_graph_path,
                   tree_file_name,tlp_flag,trend_prediction):
    graph_file_path = os.path.join(save_graph_path, str(news_id))
    ## TICK-Check this: make a balancec between saving time and saving space
    ## reduce self.neg_table_size from 1e8 to 1e6
    if os.path.exists(graph_file_path):
        with open(graph_file_path, 'rb') as f:
            graph_data = pickle.load(f)
    else:
        ## AKA self.data
        graph_data = DataHelper(tree_file_name, neg_size, hist_len, directed,
                                tlp_flag=tlp_flag,
                                trend_pred_flag=trend_prediction
                                )

        with open(graph_file_path, 'wb') as f:
            pickle.dump(graph_data, f, pickle.HIGHEST_PROTOCOL)
    return graph_data


def create_dataset(directed, file_path, hist_len, neg_size, save_graph_path,
                   only_binary, seed, tlp_flag, trend_prediction):
    print('dataset helper...')
    start_time = time.time()

    graph_data_dict = dict()
    node_dim_dict = dict()
    max_d_time_dict = dict()

    trees_to_parse = glob.glob(os.path.join(file_path, "tree/*.txt"))
    number_files = len(trees_to_parse)

    labels = load_labels(file_path)

    # Create train-val-test split
    # Remove useless trees (i.e. with labels that we don't consider)
    news_ids_to_consider = list(labels.keys())
    if only_binary:
        news_ids_to_consider = [news_id for news_id in news_ids_to_consider
                                if labels[news_id] in ['false', 'true']]

    train_ids, val_ids = train_test_split(news_ids_to_consider, test_size=0.1,random_state=seed)
    train_ids, test_ids = train_test_split(train_ids, test_size=0.25, random_state=seed * 7)
    print(f"Len train/val/test {len(train_ids)} {len(val_ids)} {len(test_ids)}")

    preprocessed_tweet_fts, preprocessed_user_fts,num_tweet_features, num_user_features = get_user_tweet_fts(file_path, train_ids, trees_to_parse)
    # print('num_user_features',num_user_features)
    # print(preprocessed_user_fts)

    ##################################################################################

    # for i_file, news_id in enumerate(news_ids_to_consider):
    for i_file, tree_file_name in enumerate(trees_to_parse):
        # if i_file > 100:
        #     continue
        if i_file % 100 == 0:
            print('Loading the {} file out of total {} files: {}'.format(i_file, number_files, tree_file_name))

        news_id = util.get_root_id(tree_file_name)
        if (news_id not in news_ids_to_consider):
            continue

        graph_data = read_graph_obj(directed, news_id, hist_len, neg_size, save_graph_path, tree_file_name,
                                    tlp_flag, trend_prediction
                                    )

        graph_data_dict[news_id] = graph_data
        node_dim_dict[news_id] = graph_data.get_node_dim()
        max_d_time_dict[news_id] = graph_data.get_max_d_time()
        # graph_id += 1

    # Twitter15: 1490  vs 983
    # assert len(news_ids_to_consider)==len(list(graph_data_dict.keys()))

    # graph_num = graph_id# the number of graph-data + 1
    node_dim = int(
        np.max(list(node_dim_dict.values())))  ## Check this: the number of nodes in all graphs?
    max_d_time = np.max(list(max_d_time_dict.values()))  ## ??? ##

    return graph_data_dict, node_dim_dict, max_d_time_dict, node_dim, max_d_time, \
           train_ids, val_ids, test_ids, labels, news_ids_to_consider, \
           preprocessed_tweet_fts, preprocessed_user_fts, num_tweet_features, num_user_features

