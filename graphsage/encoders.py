import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, name, features, feature_dim, embed_dim, adj_lists, aggregator, num_sample=10,
                 base_model=None, gcn=False, cuda=False, feature_transform=False):
        super(Encoder, self).__init__()
        self.name = name

        self.features = features
        self.feat_dim = feature_dim
        self.embed_dim = embed_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model is not None:
            self.base_model = base_model
        self.gcn = gcn
        self.cuda = cuda
        self.aggregator.cuda = cuda
        # torch.FloatTensor() will generate a random tensor with the size of the input
        self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform_(self.weight)  # init.xavier_uniform() deprecated. Use xavier_uniform_().

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        # nodes = nodes.squeeze()
        # print("{} forward!".format(self.name))

        neigh_feats = self.aggregator.forward(nodes,
                                              [self.adj_lists[int(node)] for node in nodes],
                                              self.num_sample)
        if self.gcn:  # the node's own feature is included in neigh_feats
            combined = neigh_feats
        else:     # get the embedding for these nodes, each has demension feat_dim
            self_feats = self.features(torch.LongTensor(nodes))
            if self.cuda:
                self_feats = self_feats.cuda()
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.weight.mm(combined.t()))
        return combined
