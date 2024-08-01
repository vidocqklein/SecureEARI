import copy

import numpy as np
import torch
from scipy.sparse.csgraph import laplacian
from torch_geometric.utils import to_dense_adj

from modules.perturber import Perturber
from modules.sampler import Sampler


class privacyProcess():
    def __init__(self, device, args, dataset):
        self.device = device
        self.perturb_nodes_radio = args.perturb_nodes_radio
        self.perturb_sensitive_nodes_radio = args.perturb_sensitive_nodes_radio
        self.dataset = copy.deepcopy(dataset)
        self.sensetive_attr = [1]
        self.threshhold = args.threshhold
        self.current_th = args.threshhold
        self.RAA = False
        self.m = args.m
        self.iter = args.iter
        self.sampler = Sampler()
        self.perturber = Perturber()
        self.privacyidx = None

    def sample_And_Pertrub(self, data_mask, perturb_radio):
        mask_indices = torch.nonzero(data_mask).squeeze().cpu().numpy()
        perturb_nodes_num = int(len(mask_indices) * perturb_radio)

        nodes_ds, idx_ds = self.sampler.sample(self.dataset, mask_indices, perturb_nodes_num)
        nodes_perturbed, perturb_mask = self.perturber.perturb(candidates=nodes_ds, m=self.m, RAA=self.RAA,
                                                               sensetive_attr=self.sensetive_attr,
                                                               perturbation_ratio=self.perturb_sensitive_nodes_radio)
        return idx_ds, nodes_perturbed, perturb_mask

    def feature_propagation(self, feature, stage_adj, stage_mask_indices, stage):

        all_dataset = copy.deepcopy(feature).to(device=self.device)
        perturb_stage_data = all_dataset[self.privacyidx]
        perturb_stage_data_copy = copy.deepcopy(perturb_stage_data).to(device=self.device)

        # mask = np.isin(stage_mask_indices, index)
        # index_in_train_index = np.where(mask)[0]

        # create a global mask, the length is the same as your dataset, the part corresponding to merged_indices is False, and the other part is True
        global_mask = torch.ones(perturb_stage_data_copy.shape[0], dtype=torch.bool, device=self.device)
        global_mask[self.privacyidx] = False

        for _ in range(self.iter):
            out = (stage_adj @ perturb_stage_data_copy).to(device=self.device)
            out[global_mask] = perturb_stage_data[global_mask]
            perturb_stage_data_copy = out.to(device=self.device)
        all_dataset[stage_mask_indices] = perturb_stage_data_copy
        return all_dataset


    def all_feature_propagation(self, adj):
        perturb_all_data  = copy.deepcopy(self.dataset.feature).to(device=self.device)
        merged_indices = np.concatenate([self.train_idx_ds, self.val_idx_ds, self.test_idx_ds])
        global_mask = ~np.isin(np.arange(len(self.dataset.feature)), merged_indices)

        for _ in range(self.iter):
            out = (adj @ perturb_all_data).to(device=self.device)
            out[global_mask] = perturb_all_data[global_mask]
        return out

    def all_privacy_feature_propagation(self, dataset, adj, select_idx):
        perturb_all_data = copy.deepcopy(dataset.feature).to(device=self.device)
        global_mask = ~np.isin(np.arange(len(dataset.feature)), select_idx)
        self.privacyidx = select_idx
        for _ in range(self.iter):
            out = (adj @ perturb_all_data).to(device=self.device)
            out[global_mask] = perturb_all_data[global_mask]
        return out

    def Private_attribute_perceive(self, feature, stage_adj, stage_mask_indices, stage, Y, hopF):
        if stage == 'train':
            if self.privacyidx is not None and len(self.privacyidx) > 0:
                fixed = []
                curr_th = self.threshhold
                cs = self.calculate_confidence_scores(Y[self.privacyidx])
                top_scores = cs.topk(cs.size(0))
                downFlag = True
                for i in range(cs.size(0)):
                    index = (top_scores[1][i]).item()
                    val = (top_scores[0][i]).item()
                    if (self.privacyidx[index] not in fixed) and (val >= self.current_th):
                        fixed.append(self.privacyidx[index])
                        self.privacyidx = np.delete(self.privacyidx, index)
                        self.current_th = self.threshhold
                        downFlag = False
                if downFlag:
                    self.current_th = self.current_th * 0.95
                x_new = self.feature_propagation(feature=feature, stage_adj=stage_adj,
                                                 stage_mask_indices=stage_mask_indices,
                                                 stage=stage)
                hopF = True
                return x_new, hopF
            else:
                hopF = False
                return self.dataset.feature, hopF
        else:
            return self.dataset.feature, hopF



    def calculate_confidence_scores(self,Y):
        sorted, _ = torch.sort(Y)
        max = sorted[:, -1]

        return max
