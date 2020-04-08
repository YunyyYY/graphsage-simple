import torch
import torch.nn as nn
from torch.autograd import Variable

import random

"""
Set of modules for aggregating embeddings of neighbors.
"""


class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, name, features, cuda=False, gcn=True):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()
        self.name = name

        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        
    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # print("{} forward!".format(self.name))

        _set = set  # Local pointers to functions (speed hack)
        if num_sample is not None:  # means to sample neighbors
            _sample = random.sample
            samp_neighs = [_set(
                _sample(to_neigh, num_sample,)
            ) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs  # set of neighbors for each node

        if self.gcn:
            samp_neighs = [samp_neigh | {nodes[i]} for i, samp_neigh in enumerate(samp_neighs)]

        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i,n in enumerate(unique_nodes_list)}

        # row is list of nodes in batch
        # column is nodes appeared as neighbor for this node, not necessarily in batch
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        # for j repeat samp_neighs[i] times,  for each column per entry
        mask[row_indices, column_indices] = 1

        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)  # divide by total neighbors (at most) per row
        mask[mask != mask] = 0

        embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        if self.cuda:  # embed_matrix is of shape #unique_nodes x #feat_dim
            embed_matrix = embed_matrix.cuda()

        to_feats = mask.mm(embed_matrix)  # mask is of shape #nodes x #unique_nodes
        return to_feats
