import random

from torch import nn
import sys
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch_geometric.nn import dense_mincut_pool


class Interaction_Attention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.):  # multi-head attention
        super().__init__()
        dim_head = dim // heads
        self.heads = heads
        self.scale = dim_head ** -0.5  # scale the dot product between q and k
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)  # split into q, k, v
        for tensor in self.parameters():  # initialize parameters
            nn.init.normal_(tensor, mean=0.0, std=0.05)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # chunk into q, k, v
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
                      qkv)  # output shape is (batch_size, heads, num_nodes, dim_head)
        dots = torch.matmul(q, k.transpose(-1,
                                           -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        attn_score = attn[0][0]
        out = torch.matmul(attn, v)
        out = rearrange(out,
                        'b h n d -> b n (h d)')
        return out


'''
model: EARI
'''


class EARI(nn.Module):
    def __init__(self, adj, in_channels, hidden_channels, out_channels, hops, dropout, activation,
                 feature_inter, inter_layer, feature_fusion, norm_type, epoch_time, p, cluster_num, device):
        super().__init__()
        self.hops = hops
        self.feature_inter_type = feature_inter
        self.feature_fusion = feature_fusion
        self.dropout = nn.Dropout(dropout)
        self.pre = False
        self.adj = adj
        self.max_degree = 0.1
        self.norm_type = norm_type
        self.epoch_time = epoch_time
        self.p = p
        self.cluster_num = cluster_num
        self.select_activation(activation)
        self.device = device

        self.fc = nn.Linear(in_channels, hidden_channels)


        self.hop_embedding = nn.Parameter(torch.randn(1, hops, hidden_channels))
        # interaction
        self.build_feature_inter_layer(feature_inter, hidden_channels, inter_layer)

        # fusion type
        if self.feature_fusion == 'attention':
            self.atten_self = nn.Linear(hidden_channels, 1)
            self.atten_neighbor = nn.Linear(hidden_channels, 1)

        self.fc2 = nn.Linear(hidden_channels, cluster_num)

        # prediction
        self.classifier = nn.Linear(hidden_channels, out_channels)

        # norm
        self.build_norm_layer(hidden_channels, inter_layer * 2 + 2)

    '''
    activation choice
    '''

    def select_activation(self, activation):

        self.activate = F.relu

    '''
    multi-hop feature preprocess
    '''

    def preprocess(self, adj, x):
        h = []
        for i in range(self.hops):
            h.append(x)
            x = adj @ x
        self.h = torch.stack(h, dim=1)
        self.pre = True
        return self.h

    '''
    feature interaction layer build
    '''

    def build_feature_inter_layer(self, feature_inter, hidden_channels, inter_layer):
        self.interaction_layers = nn.ModuleList()
        if feature_inter == 'mlp':
            for i in range(inter_layer):
                mlp = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU())
                self.interaction_layers.append(mlp)
        elif feature_inter == 'gcn':
            for i in range(inter_layer):
                self.interaction_layers.append(Interaction_GCN(hidden_channels))
        elif feature_inter == 'attention':
            for i in range(inter_layer):
                self.interaction_layers.append(
                    Interaction_Attention(hidden_channels, heads=4, dropout=0.1))
        else:
            self.interaction_layers.append(torch.nn.Identity())

    '''
    norm layer build
    '''

    def build_norm_layer(self, hidden_channels, layers):
        self.norm_layers = nn.ModuleList()
        for i in range(layers):

            elif self.norm_type == 'ln':
                self.norm_layers.append(nn.LayerNorm(hidden_channels))

    def norm(self, h, layer_index):
        h = self.norm_layers[layer_index](h)
        return h


    def embedding(self, h):
        h = self.dropout(h)
        h = self.fc(h)
        h = h + self.hop_embedding
        h = self.norm(h, 0)
        return h


    def interaction(self, h):
        inter_layers = len(self.interaction_layers)
        for i in range(inter_layers):
            h_prev = h
            h = self.dropout(h)
            h = self.interaction_layers[i](h)
            h = self.activate(h)
            h = h + h_prev  # residual
            h = self.norm(h, i + 1)
        return h

    '''
    feature fusion
    '''

    def fusion(self, h):
        h = self.dropout(h)
        if self.feature_fusion == 'max':
            h = h.max(dim=1).values
        elif self.feature_fusion == 'attention':
            h_self, h_neighbor = h[:, 0, :], h[:, 1:, :]
            h_self_atten = self.atten_self(h_self).view(-1, 1)
            h_neighbor_atten = self.atten_neighbor(h_neighbor).squeeze()
            h_atten = torch.softmax(F.leaky_relu(h_self_atten + h_neighbor_atten), dim=1)
            h_neighbor = torch.einsum('nhd, nh -> nd', h_neighbor,
                                      h_atten).squeeze()
            h = h_self + h_neighbor
        else:  # mean
            h = h.mean(dim=1)
        h = self.norm(h, -1)
        return h

    '''
    multi-hop feature build
    '''

    def build_hop(self, inputs, batch_adj, hopF):
        if len(inputs.shape) == 3:
            h = inputs
        else:
            if self.pre == False or hopF:
                self.h0 = self.preprocess(batch_adj, inputs)
            # if self.pre == False and batch_adj is not None:
            #     self.h0 = self.preprocess(batch_adj, inputs)
            # elif self.pre == False and batch_adj is None:
            #     self.h0 = self.preprocess(self.adj, inputs)
            h = self.h0
        return h

    def cluster_wise_adaptation_learning(self, h_embedding, h_clu, assignment_matrics):

        index = random.sample(range(0, h_embedding.shape[0]), int(self.p * h_embedding.shape[0]))

        tensor_mask = torch.ones(h_embedding.shape[0], 1).to(self.device)
        tensor_mask[index] = 0

        tensor_selectclu = torch.randint(low=0, high=h_clu.shape[0] - 1, size=(h_embedding.shape[0],)).to(
            self.device)
        Select = torch.argmax(assignment_matrics, dim=1).to(self.device)
        tensor_selectclu[tensor_selectclu == Select] = h_clu.shape[0] - 1

        a1 = torch.unsqueeze(h_embedding, 0)
        a1 = a1.repeat(h_clu.shape[0], 1, 1)
        b1 = h_clu.unsqueeze(1)
        c = a1 - b1
        d = torch.pow(c, 2)

        s = assignment_matrics.t()
        s = s.unsqueeze(1)
        tensor_var_clu = torch.bmm(s, d).squeeze()

        tensor_std_clu = torch.pow(tensor_var_clu + 1e-10, 0.5)

        tensor_mean_emb = h_embedding.mean(1, keepdim=True)
        tensor_std_emb = h_embedding.var(1, keepdim=True).sqrt()

        sigma_mean = h_clu.mean(1, keepdim=True).var(0).add(1e-8).sqrt()
        sigma_std = (tensor_std_clu.var(0) + 1e-10).sqrt()

        tensor_beta = tensor_std_clu[tensor_selectclu] + torch.randn_like(tensor_std_emb) * sigma_std
        tensor_gama = h_clu[tensor_selectclu] + torch.randn_like(tensor_std_emb) * sigma_mean

        h_new = tensor_mask * h_embedding + (1 - tensor_mask) * (
                ((h_embedding - h_clu[Select]) / (tensor_std_clu[Select] + 1e-10)) * tensor_beta + tensor_gama)

        return h_new

    '''
    all forward
    '''

    def forward(self, inputs, batch_adj, stage, epoch_now, hopF):
        # step-1 the first preprocess of hop-information for accerelate training
        hop_feature = self.build_hop(inputs, batch_adj, hopF)
        # step-2 hop-embedding
        h1 = self.embedding(hop_feature)
        # step-3 hop-interaction
        h = self.interaction(h1)
        # step-4 hop-fusion
        h2 = self.fusion(h)

        if stage == 'train':
            assignment_matrics = self.fc2(h2)  # cluster assignment
            assignment_matrics = nn.Softmax(dim=-1)(assignment_matrics)
            _, _, mc_loss, o_loss = dense_mincut_pool(h2, batch_adj.to_dense(),
                                                      assignment_matrics)  # compute mincut loss
            h_pool = torch.matmul(torch.transpose(assignment_matrics, 0, 1),
                                  h2)  # compute cluster center

            if epoch_now % self.epoch_time == 0:
                h2 = self.cluster_wise_adaptation_learning(h2, h_pool, assignment_matrics)

            h2 = self.dropout(h2)
            # step-5 prediction
            h3 = self.classifier(h2)
            return h3, mc_loss, o_loss
            # return h3, 0, 0
        else:
            h2 = self.dropout(h2)
            # step-5 prediction
            h3 = self.classifier(h2)
            return h3
