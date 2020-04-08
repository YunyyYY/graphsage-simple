import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator
from sklearn.metrics import accuracy_score

from graphsage.utils import load_ml10, load_ml20, precision_at_k

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.bce = nn.BCEWithLogitsLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)  # call self.enc.forward(nodes)
        scores = self.weight.mm(embeds)
        # print(embeds.shape)
        # print(scores.shape)
        return scores.t()  # so that each row is a node, entries are scores

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        # print(scores[0].shape)
        # print(labels[0].shape)
        return self.bce(scores, labels.squeeze())


def run_ml10(year):
    np.random.seed(1)
    random.seed(1)
    train_node, test_shape, adj, nodes_feature, labels = load_ml10(year)
    # need to embed whole graph

    print(train_node, test_shape)
    num_nodes = labels.shape[0]
    num_feats = 3000
    features = nn.Embedding(num_nodes, num_feats)
    features.weight = nn.Parameter(torch.FloatTensor(nodes_feature),
                                   requires_grad=False)
    # features.cuda()

    agg1 = MeanAggregator("agg1", features, cuda=False)
    enc1 = Encoder("enc1", features, num_feats, 128, adj, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator("agg2", lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder("enc2", lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128,
                   adj, agg2,base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 10
    enc2.num_samples = 10

    graphsage = SupervisedGraphSage(19, enc2)
    # graphsage.cuda()

    split = int(0.2 * train_node)
    train = np.array(range(0, split))

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        graphsage.parameters()), lr=0.02)
    times = []
    for batch in range(10):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.FloatTensor(labels[batch_nodes])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.data.item())
        # print(batch, loss.data[0])

    # validate
    val = np.array(range(split, train_node))
    val_pred = graphsage.forward(val).detach().numpy()
    val_true = labels[val]

    # test data
    test = np.array(range(train_node, num_nodes))
    test_pred = graphsage.forward(test).detach().numpy()
    test_true = labels[test]

    for k in [1, 3, 5]:
        val_acc = precision_at_k(val_pred, val_true, k)
        test_acc = precision_at_k(test_pred, test_true, k)
        print(k, val_acc, test_acc)

    print("Average batch time:", np.mean(times))


def run_ml20():
    num_nodes = 22396  # wc -l cora/cora.content 2708
    num_feats = 19350
    num_labels = 20
    num_train = 21282
    _features, labels, adj = load_ml20()

    features = nn.Embedding(num_nodes, num_feats)
    features.weight = nn.Parameter(torch.FloatTensor(_features),
                                   requires_grad=False)
    # features.cuda()

    # construct graph
    agg1 = MeanAggregator("agg1", features, cuda=False)
    enc1 = Encoder("enc1", features, num_feats, 128, adj, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator("agg2", lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder("enc2", lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128,
                   adj, agg2, base_model=enc1, gcn=True, cuda=False)
    # change number of samples in each layer
    enc1.num_samples = 10
    enc2.num_samples = 10

    graphsage = SupervisedGraphSage(num_labels, enc2)
    # graphsage.cuda()

    # split train, validate, test set
    split = int(0.8 * num_train)
    train = np.array(range(split))
    val = np.array(range(split, num_train))
    test = np.array(range(num_train, num_nodes))

    # start training
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        graphsage.parameters()), lr=0.02)
    times = []
    for batch in range(100):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes,
                              Variable(torch.FloatTensor(labels[batch_nodes])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)
        print(batch, loss.data.item())

    # validate and prediction
    val_pred = graphsage.forward(val)
    val_true = torch.FloatTensor((labels[val]))
    test_pred = graphsage.forward(test)
    test_true = torch.FloatTensor((labels[num_train:]))

    val_pred = val_pred.data.numpy()
    val_true = val_true.data.numpy()
    val_pred[val_pred > 0] = 1
    val_pred[val_pred < 0] = 0
    test_pred = test_pred.data.numpy()
    test_true = test_true.data.numpy()
    test_pred[test_pred > 0] = 1
    test_pred[test_pred < 0] = 0
    print("Validation accuracy: ", accuracy_score(val_true, val_pred))
    print("Test accuracy: ", accuracy_score(test_true, test_pred))
    print("Average batch time:", np.mean(times))


if __name__ == "__main__":
    run_ml10(2007)
    # run_ml10(1997)
