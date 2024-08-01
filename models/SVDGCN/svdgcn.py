import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch_sparse import SparseTensor


class GCNSVD(torch.nn.Module):
    """GCNSVD is a 2 Layer Graph Convolutional Network with Truncated SVD as
    preprocessing. See more details in All You Need Is Low (Rank): Defending
    Against Adversarial Attacks on Graphs,
    https://dl.acm.org/doi/abs/10.1145/3336191.3371789.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    dropout : float
        dropout rate for GCN
    lr : float
        learning rate for GCN
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN. When `with_relu` is True, `weight_decay` will be set to 0.
    with_relu : bool
        whether to use relu activation function. If False, GCN will be linearized.
    with_bias: bool
        whether to include bias term in GCN weights.
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
	We can first load dataset and then train GCNSVD.

    >>> from deeprobust.graph.data import PrePtbDataset, Dataset
    >>> from deeprobust.graph.defense import GCNSVD
    >>> # load clean graph data
    >>> data = Dataset(root='/tmp/', name='cora', seed=15)
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # load perturbed graph data
    >>> perturbed_data = PrePtbDataset(root='/tmp/', name='cora')
    >>> perturbed_adj = perturbed_data.adj
    >>> # train defense model
    >>> model = GCNSVD(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu').to('cpu')
    >>> model.fit(features, perturbed_adj, labels, idx_train, idx_val, k=20)

    """

    def __init__(self, with_relu=True, with_bias=True, device='cpu'):

        super().__init__()
        self.device = device
        self.k = None

    def forward(self,  adj,  k=50,):
        """First perform rank-k approximation of adjacency matrix via
        truncated SVD, and then train the gcn model on the processed graph,
        when idx_val is not None, pick the best model according to
        the validation loss.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        k : int
            number of singular values and vectors to compute.
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        """

        modified_adj = self.truncatedSVD(adj, k=k)
        self.k = k
        return modified_adj


    def truncatedSVD(self, data, k=50):
        """Truncated SVD on input data.

        Parameters
        ----------
        data :
            input matrix to be decomposed
        k : int
            number of singular values and vectors to compute.

        Returns
        -------
        numpy.array
            reconstructed matrix.
        """
        print('=== GCN-SVD: rank={} ==='.format(k))
        data = data.to_scipy()
        if sp.issparse(data):
            data = data.asfptype()
            U, S, V = sp.linalg.svds(data, k=k)
            print("rank_after = {}".format(len(S.nonzero()[0])))
            diag_S = np.diag(S)
        else:
            U, S, V = np.linalg.svd(data)
            U = U[:, :k]
            S = S[:k]
            V = V[:k, :]
            print("rank_before = {}".format(len(S.nonzero()[0])))
            diag_S = np.diag(S)
            print("rank_after = {}".format(len(diag_S.nonzero()[0])))

        return U @ diag_S @ V