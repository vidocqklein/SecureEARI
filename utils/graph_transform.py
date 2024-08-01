import numpy as np
import scipy as sp
from scipy.sparse import diags
import torch
import dgl
import torch_geometric
from torch_sparse import SparseTensor
import torch.nn.functional as F
def remove_self_loop(edge_index):
    assert edge_index.shape[0] == 2
    edges = dgl.graph((edge_index[0], edge_index[1])).to(edge_index.device)
    edges = edges.add_self_loop().remove_self_loop()
    edges = [a.long() for a in edges.edges()]
    edge_index = torch_geometric.data.Data(edge_index=torch.stack(edges)).edge_index
    return edge_index

# D-1/2 * A * D-1/2 or D-1 * A，symmetric_norm
def sparse_normalize(adj, symmetric_norm=True):
    assert isinstance(adj, SparseTensor)
    size = adj.size(0)
    ones = torch.ones(size).view(-1, 1).to(adj.device())
    degree = adj @ ones
    if symmetric_norm == False:
        degree = degree ** -1
        degree[torch.isinf(degree)] = 0
        return adj * degree
    else:
        degree = degree ** (-1/2)
        degree[torch.isinf(degree)] = 0
        d = SparseTensor(row=torch.arange(size), col=torch.arange(size), value=degree.squeeze().cpu(),
                         sparse_sizes=(size, size)).to(adj.device())
        adj = adj @ d
        adj = adj * degree
        return adj


def nomarlizeAdj(adj):
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    return adj

# D-1 * A
def normalizeLelf(adj):
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj = deg_inv_sqrt.view(-1, 1) * adj
    return adj

def mask_split(data):
    y = data.y
    random_node_indices = np.random.permutation(y.shape[0])
    training_size = int(len(random_node_indices) * 0.7)
    val_size = int(len(random_node_indices) * 0.1)
    train_node_indices = random_node_indices[:training_size]
    val_node_indices = random_node_indices[training_size:training_size + val_size]
    test_node_indices = random_node_indices[training_size + val_size:]

    train_masks = torch.zeros([y.shape[0]], dtype=torch.uint8)
    train_masks[train_node_indices] = 1
    val_masks = torch.zeros([y.shape[0]], dtype=torch.uint8)
    val_masks[val_node_indices] = 1
    test_masks = torch.zeros([y.shape[0]], dtype=torch.uint8)
    test_masks[test_node_indices] = 1

    data.train_mask = train_masks
    data.val_mask = val_masks
    data.test_mask = test_masks
    return data



def add_random_edges(edge_tensor, add_fraction, num_nodes, device):

    num_edges = edge_tensor.shape[1]
    num_add = int(num_edges * add_fraction)
    existing_edges = set(zip(edge_tensor[0].tolist(), edge_tensor[1].tolist()))

    new_edges = set()
    while len(new_edges) < num_add:
        edge = (np.random.randint(0, num_nodes), np.random.randint(0, num_nodes))
        if edge not in existing_edges and edge[0] != edge[1]:
            new_edges.add(edge)

    new_edges_tensor = torch.tensor(list(new_edges)).T.to(device)
    return torch.cat([edge_tensor, new_edges_tensor], dim=1)


def remove_random_edges(edge_tensor, del_fraction):
    num_edges = edge_tensor.shape[1]
    num_del = int(num_edges * del_fraction)

    indices = torch.randperm(num_edges)
    keep_indices = indices[num_del:]

    return edge_tensor[:, keep_indices]

def normalizePF(mx):
    mx = mx.cpu()
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def privacy_perturbation(data, choice_radio, epsilon):
    sf = normalizePF(data.feature)
    the_h = sf.max(0)
    the_lambda = (the_h / epsilon)

    choice_num = int(data.num_nodes * choice_radio)
    select_nodes_idx = np.random.choice(data.num_nodes, choice_num, replace=False)
    data.feature = F.normalize(data.feature, p=2, dim=1)
    for node in select_nodes_idx:
        noise = np.random.laplace(0, the_lambda, data.feature[node].shape)
        data.feature[node] = data.feature[node] + torch.from_numpy(noise).to(data.device)
    return data, select_nodes_idx
