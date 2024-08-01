import os

import torch
import scipy.io as sio
from torch_geometric.data import Data
from torch_sparse import SparseTensor

from datasets.privacyDataLoader import privacyData_loader
from utils.graph_transform import remove_random_edges, add_random_edges, sparse_normalize, remove_self_loop

device = 'cuda:0'
def getData(dataset_name):
    get_Dataset = privacyData_loader(dataset_name, device)
    adj = get_Dataset.adj
    return get_Dataset, adj
def load_network(file):
    net = sio.loadmat(file)
    X, A, Y = net['attrb'], net['network'], net['group']
    return A, X, Y
def load_network(file):
    net = sio.loadmat(file)
    X, A, Y = net['attrb'], net['network'], net['group']
    return A, X, Y

def data_process(A, X, Y):
    y = torch.tensor(Y)
    y = torch.argmax(y, dim=1)
    x = SparseTensor.from_scipy(X)
    adj = SparseTensor.from_scipy(A)
    row = adj.storage.row()
    col = adj.storage.col()
    adjindex = torch.stack((row, col))
    data = Data(x=x, y=y, edge_index=adjindex)
    return data
def loadData(dataset):
    # dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, dataset)
    file_path = os.path.join(folder_path, str(dataset) + '.mat')
    # print(absolute_path)
    A, X, Y = load_network(file_path)
    # A, X, Y = load_network(dir +'/datasets/' + str(dataset) + '.mat')
    data = data_process(A, X, Y)
    return data

def init_matrix(edge_index,node_nums):
    edge_index = remove_self_loop(edge_index)
    adj = SparseTensor(row=edge_index[0, :], col=edge_index[1, :], sparse_sizes=(node_nums, node_nums))
    adj = adj.to(device)
    adj = sparse_normalize(adj, True).to(device)
    return adj

data = loadData('dblpv7')
node_nums = data.x.sizes()[0]
del_adj = remove_random_edges(data.edge_index.to(device),0.5)
add_adj = add_random_edges(data.edge_index.to(device),0.5,node_nums,device)
torch.save(del_adj, 'dblpv7/random-del-dblpv7-0.5.pt')
torch.save(add_adj, 'dblpv7/random-add-dblpv7-0.5.pt')



