

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor

from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np


class AttackABC(ABC):

    def __init__(self,
                 attack_config, dataset,
                 model, device):
        self.device = device
        self.attack_config = attack_config

        self.loss_type = attack_config['loss_type']
        # self.attacked_model = deepcopy(model).to(self.device)
        self.attacked_model = model

        self.dataset = deepcopy(dataset)
        # if self.__class__.__name__ not in ['DICE', 'Random']:
        #     pseudo_label = gen_pseudo_label(self.attacked_model, self.pyg_data.y, self.pyg_data.test_mask)
        #     self.pyg_data.y = pseudo_label
        # pseudo_label = gen_pseudo_label(self.attacked_model, self.dataset.label, self.dataset.test_mask)
        # self.dataset.label = pseudo_label
        # self.dataset = self.dataset.to(self.device)


        for p in self.attacked_model.parameters():
            p.requires_grad = False
        self.eval_model = self.attacked_model

        self.attr_adversary = self.dataset.feature
        self.adj_adversary = self.dataset.adj


    @abstractmethod
    def _attack(self, n_perturbations):
        pass

    def attack(self, n_perturbations, **kwargs):
        if n_perturbations > 0:
            return self._attack(n_perturbations, **kwargs)
        else:
            self.attr_adversary = self.dataset.feature
            self.adj_adversary = self.dataset.adj

    def get_perturbations(self):
        adj_adversary, attr_adversary = self.adj_adversary, self.attr_adversary
        if isinstance(self.adj_adversary, torch.Tensor):
            adj_adversary = SparseTensor.from_dense(self.adj_adversary)
        if isinstance(self.attr_adversary, SparseTensor):
            attr_adversary = self.attr_adversary.to_dense()

        return adj_adversary, attr_adversary

    def calculate_loss(self, logits, labels):
        if self.loss_type == 'CW':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = -torch.clamp(margin, min=0).mean()
        elif self.loss_type == 'LCW':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = -F.leaky_relu(margin, negative_slope=0.1).mean()
        elif self.loss_type == 'tanhMargin':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = torch.tanh(-margin).mean()
        elif self.loss_type == 'Margin':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = -margin.mean()
        elif self.loss_type.startswith('tanhMarginCW-'):
            alpha = float(self.loss_type.split('-')[-1])
            assert alpha >= 0, f'Alpha {alpha} must be greater or equal 0'
            assert alpha <= 1, f'Alpha {alpha} must be less or equal 1'
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = (alpha * torch.tanh(-margin) - (1 - alpha) * torch.clamp(margin, min=0)).mean()
        elif self.loss_type.startswith('tanhMarginMCE-'):
            alpha = float(self.loss_type.split('-')[-1])
            assert alpha >= 0, f'Alpha {alpha} must be greater or equal 0'
            assert alpha <= 1, f'Alpha {alpha} must be less or equal 1'

            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )

            not_flipped = logits.argmax(-1) == labels

            loss = alpha * torch.tanh(-margin).mean() + (1 - alpha) * \
                F.cross_entropy(logits[not_flipped], labels[not_flipped])
        elif self.loss_type == 'eluMargin':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = -F.elu(margin).mean()
        elif self.loss_type == 'MCE':
            not_flipped = logits.argmax(-1) == labels
            loss = F.cross_entropy(logits[not_flipped], labels[not_flipped])
        elif self.loss_type == 'NCE':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            loss = -F.cross_entropy(logits, best_non_target_class)
        else:
            loss = F.cross_entropy(logits, labels)
        return loss

