import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, device):
        support = torch.mm(input, self.weight.to(device))
        # output = torch.sparse.mm(adj, support)
        output = adj @ support
        if self.bias is not None:
            return output + self.bias.to(device)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
class surrogate_GCN(torch.nn.Module):
    def __init__(self, num_features,num_classes, hidden_layer, dropout, device):
        super().__init__()
        self.p = dropout
        # torch.manual_seed(1234567)
        self.conv1 = GCNConv(num_features, hidden_layer)
        self.conv2 = GCNConv(hidden_layer, num_classes)
        self.is_dense = False

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

class GCN(torch.nn.Module):
    def __init__(self,num_features,num_classes, hidden_layer, dropout, device):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.device = device
        self.conv1 = GraphConvolution(num_features, hidden_layer)
        self.conv2 = GraphConvolution(hidden_layer, num_classes)

    def forward(self, x, adj):
        h = self.conv1(x, adj, self.device)
        h = F.relu(h)
        h_d = F.dropout(h, self.dropout, training=self.training)
        h = self.conv2(h_d, adj, self.device)
        # output = F.log_softmax(h, dim=1)
        return h