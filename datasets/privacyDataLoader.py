import os
import torch
import scipy.io as sio
import torch_geometric.transforms as T
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_sparse import SparseTensor
from utils.graph_transform import sparse_normalize, remove_self_loop, remove_random_edges, add_random_edges

script_dir = os.path.dirname(os.path.abspath(__file__))

class privacyDataLoader(Dataset):
    def __init__(self, data, folder_path, device):
        self.name = data.name
        self.edge_index = data.edge_index.to(device)
        self.num_edges = data.edge_index.shape[1]
        self.feature = data.x.to(device)
        self.valFeature = self.feature
        self.label = data.y.to(device)
        self.vallabel = data.y.to(device)
        self.device = device
        self.num_label_class = self.label.max().item() + 1
        self.num_nodes = data.num_nodes
        self.in_degrees = degree(self.edge_index[0, :], num_nodes=self.num_nodes, dtype=torch.long).to(device)
        self.id_mask = torch.ones(self.num_nodes).bool().to(device)#which is used to select the nodes
        self.train_mask = data.train_mask.to(device)#5
        self.val_mask = data.val_mask.to(device)#3
        self.test_mask = data.test_mask.to(device)#2
        self.adj = self.init_matrix(self.edge_index)
        self.shift_adj = self.adj#if the adj is changed, the shift_adj will be changed
        self.data = data
        print('load %s dataset successfully!' % self.name)

    def init_matrix(self, edge_index):
        edge_index = remove_self_loop(edge_index)#remove self loop
        adj = SparseTensor(row=edge_index[0, :], col=edge_index[1, :], sparse_sizes=(self.num_nodes, self.num_nodes))
        adj = adj.to(self.device)
        adj = sparse_normalize(adj, True).to(self.device)
        return adj

def other_init_matrix(edge_index, num_nodes, device):
    edge_index = remove_self_loop(edge_index)#remove self loop
    adj = SparseTensor(row=edge_index[0, :], col=edge_index[1, :], sparse_sizes=(num_nodes, num_nodes))
    adj = adj.to(device)
    adj = sparse_normalize(adj, True).to(device)
    return adj


def privacyData_loader(dataset,device):
    data,folder_path = loadData(dataset)
    splitdata = get_SplitData(data, dataset)
    data = privacyDataLoader(splitdata, folder_path, device)
    return data
def load_network(file):
    net = sio.loadmat(file)
    X, A, Y = net['attrb'], net['network'], net['group']#The feature is converted to a sparse matrix
    return A, X, Y

def data_process(A, X, Y):
    y = torch.tensor(Y)
    y = torch.argmax(y, dim=1)
    x = SparseTensor.from_scipy(X).to_dense().to(torch.float32)
    adj = SparseTensor.from_scipy(A)
    row = adj.storage.row()
    col = adj.storage.col()
    adjindex = torch.stack((row, col))
    num_nodes = x.shape[0]
    data = Data(x=x, y=y, edge_index=adjindex, num_nodes=num_nodes)
    return data
def get_SplitData(data,dataset):
    split = T.RandomNodeSplit(num_val=0.2, num_test=0.1)
    graph = split(data)
    graph.name = dataset
    return graph
def loadData(dataset):
    if dataset == 'dblpv7' or dataset == 'acmv9' or dataset == 'citationv1':
        folder_path = os.path.join(script_dir, dataset)  # locate to the folder
        file_path = os.path.join(folder_path, str(dataset) + '.mat')  # locate to the file
        A, X, Y = load_network(file_path)
        data = data_process(A, X, Y)
    else:
        folder_path = os.path.join(script_dir, dataset)
        data = load_citation(dataset)
    return data, folder_path
def load_citation(data_name):
    if data_name == 'cora':
        dataset = CoraGraphDataset(verbose=False)
    elif data_name == 'citeseer':
        dataset = CiteseerGraphDataset(verbose=False)
    elif data_name == 'pubmed':
        dataset = PubmedGraphDataset(verbose=False)
    else:
        raise 'not implement citation dataset'

    g = dataset[0]
    edges = [a.long() for a in g.edges()]
    num_nodes = g.ndata['feat'].shape[0]
    data = Data(x=g.ndata['feat'], y=g.ndata['label'], edge_index=torch.stack(edges), num_nodes=num_nodes)
    return data




