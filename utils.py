import numpy as np
import pandas as pd
from collections import defaultdict


def _tmp_tag(x):
    try:
        return np.fromiter(map(int, x.split()), np.int)
    except:
        return []

# takes about 15 seconds
def load_ml20():
    num_nodes = 22396  # wc -l cora/cora.content 2708
    num_feats = 19350
    num_labels = 20
    num_train = 21282
    # num_test = 1114
    
    # first load movies
    movie = pd.read_csv('ml-20m/labels.csv')
    movie.genc = movie.genc.apply(lambda x: np.fromiter(map(int, x[1:-1].split()), np.int))
    
    mv_train = movie[movie.year < 2009].movieId.values
    mv_test = movie[movie.year == 2009].movieId.values
    id_arr = np.concatenate((mv_train, mv_test), axis=0)  # movieId concatenation
    index_mv = {mv: i for i, mv in enumerate(id_arr)}

    tag = pd.read_csv('ml-20m/tag_enc.csv')
    tag.tagenc = tag.tagenc.apply(_tmp_tag)
    
    # create label encode
    labels = np.zeros((num_nodes, num_labels))
    features = np.zeros((num_nodes, num_feats))

    for i, mv in enumerate(id_arr):
        try:
            labels[i, movie[movie.movieId == mv].genc.values] = 1
            features[i, tag[tag.movieId == mv].tagenc.values] = 1
        except:
            pass

    # create edges
    adj_lists = defaultdict(set)  # so that each value in adj_lists is a set
    with open("ml-20m/train_2008.edge") as fp:
        for i, line in enumerate(fp):
            info = list(map(int, line.strip().split()))
            mv1 = index_mv[info[0]]
            mv2 = index_mv[info[1]]
            adj_lists[mv1].add(mv2)
            adj_lists[mv2].add(mv1)
    with open("ml-20m/test_2009.edge") as fp:
        for i, line in enumerate(fp):
            info = list(map(int, line.strip().split()))
            mv1 = index_mv[info[0]]
            mv2 = index_mv[info[1]]
            adj_lists[mv1].add(mv2)
            adj_lists[mv2].add(mv1)
    return num_train, features, labels, adj_lists


if __name__ == '__main__':
    num_train, features, labels, adj_lists = load_ml20()
    print("num_train,", num_train)
    print("features.shape,", features.shape)
    print("labels.shape", labels.shape)
    print("adj_list,", len(adj_lists))

