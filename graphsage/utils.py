import numpy as np
import pandas as pd
from collections import defaultdict


def _tmp_tag(x):
    """
    Used by load_ml20 parse tag
    """
    try:
        return np.fromiter(map(int, x.split()), np.int)
    except:
        return []


def load_ml20():
    """
    takes about 15 seconds
    """
    num_nodes = 22396  # wc -l cora/cora.content 2708
    num_feats = 19350
    num_labels = 20
    # num_train = 21282
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
    return features, labels, adj_lists


def load_ml10(year=2005):
    """
    - Year will be used as test set
    - All previous years will be used as training set
    - Return train_adj, train_feat, train_label, test_adj, test_feat, test_label
    """
    print("Start building graph")
    users = pd.read_csv('ml-10m/users.csv')
    features = pd.read_csv('ml-10m/features.csv', dtype={'features': 'str'})
    labels = pd.read_csv('ml-10m/movies.csv', dtype={'Gencode': 'str'})

    trainset = labels[labels.Year < year].MovieID.values
    testset = labels[labels.Year >= year].MovieID.values
    movieset = np.concatenate([trainset, testset])
    train_shape = trainset.shape[0]
    test_shape = testset.shape[0]
    node_map = {}
    label_list = []
    feature = np.zeros((train_shape + test_shape, 3000))
    adj = defaultdict(set)

    # get labels, also construct node maps
    label_set = np.concatenate([labels[labels.Year < year].values,
                            labels[labels.Year == year].values])
    for index, row in enumerate(label_set):
        node_map[row[0]] = index
        label_list.append(list(map(int, row[4])))

    # get features
    for _, row in features[features.MovieID.isin(movieset)].iterrows():
        feature[node_map[row.MovieID]] = list(map(int, row.features))

    # get adj list
    for _, row in users[users.Year <= year].iterrows():
        mvList = row.Movies[1:-1]
        mvList = mvList.split(',')
        mvList = [int(i) for i in mvList]
        for mvx in mvList:
            try:
                mv1 = node_map[mvx]
                for mvy in mvList:
                    if mvy == mvx:
                        pass
                    mv2 = node_map[mvy]
                    adj[mv1].add(mv2)
                    adj[mv2].add(mv1)
            except:
                pass
    print("Build complete")
    return train_shape, test_shape, adj, feature, np.array(label_list)


def precision_at_k(pred, true, k):
    num = len(pred)
    acc = 0
    for i in range(num):
        _y1 = pred[i].argsort()[-k:]
        _y2 = np.where(true[i] == 1)[0]
        count = 0
        for _y in _y1:
            if _y in _y2:
                count += 1
        acc += (count/k)
    return acc/num


# if __name__ == '__main__':
#     num_train, features, labels, adj_lists = load_ml20()
#     print("num_train,", num_train)
#     print("features.shape,", features.shape)
#     print("labels.shape", labels.shape)
#     print("adj_list,", len(adj_lists))
#
