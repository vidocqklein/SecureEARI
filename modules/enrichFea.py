import argparse

import torch

from datasets.privacyDataLoader import privacyData_loader
from models.EARImodel.EARI import EARI
from modules.privacyProcess import privacyProcess
from paramSet import set_modelParams
from trainer.trainer import Trainer
from utils.graph_transform import privacy_perturbation
from utils.utils import load_perturbed_adj, set_seed


# executed in EARI.py
class enrichFea():
    def __init__(self, device, args, num_hop=-1, cluster_num=-1, p=-1, epochtime=-1, cleanFlag=False,noepFlag=False):
        self.device = device
        self.dataset = args.dataset
        self.args = args
        self.modelParams = set_modelParams(self.args.model)
        self.cleanFlag = cleanFlag
        self.noepFlag = noepFlag
        if num_hop > -1:
            self.modelParams.num_hop = num_hop
        if cluster_num > -1:
            self.modelParams.cluster_num = cluster_num
        if p > -1:
            self.modelParams.p = p
        if epochtime > -1:
            self.modelParams.epochtime = epochtime

        set_seed(202)
        if self.args.inductive_setting:  # inductive setting
            get_Dataset, adj, input_dim, hid_dim, out_dim = self.getData(self.dataset)
            get_Dataset_t, adj_t, input_dim_t, hid_dim_t, out_dim_t = self.getData(self.args.tdataset)
            model, get_Dataset = self.Feaproprocess(get_Dataset, adj, input_dim, hid_dim, out_dim, tdataset=get_Dataset_t)
        else:
            get_Dataset, adj, input_dim, hid_dim, out_dim = self.getData(self.dataset)
            model, get_Dataset = self.Feaproprocess(get_Dataset, adj, input_dim, hid_dim, out_dim)



        print('params loading——————', 'lr:', args.lr, 'wd:', args.weight_decay, 'hop:', self.modelParams.num_hop, 'dropout:',
              self.modelParams.dropout,'attack:', self.args.Remarks, 'repeat:',self.args.iterNum,
              self.modelParams.interaction, 'inter_layers:', self.modelParams.num_layer, self.modelParams.fusion,
              self.modelParams.norm_type, 'p:', self.modelParams.p, 'cluster_num:', self.modelParams.cluster_num, 'epoch_time:', self.modelParams.epochtime)
        print('============================================')
        set_seed(self.args.seed)
        if self.args.inductive_setting:
            Trainer(device=self.device, epoch=args.epochs, model=model, dataset=get_Dataset, args=self.args,
                    modelParams=self.modelParams, t_val_mask=get_Dataset_t.val_mask, t_test_mask=get_Dataset_t.test_mask)
        else:
            Trainer(device=self.device, epoch=args.epochs, model=model, dataset=get_Dataset, args=self.args,
                    modelParams=self.modelParams)

    def getData(self, dataset_name):
        get_Dataset = privacyData_loader(dataset_name, self.device)
        adj = get_Dataset.adj
        input_dim, hid_dim, out_dim = get_Dataset.feature.shape[1], self.modelParams.hidden, get_Dataset.num_label_class
        return get_Dataset, adj, input_dim, hid_dim, out_dim

    def setModel(self, adj, input_dim, hid_dim, out_dim):
        if self.args.model == 'eari':
            model = EARI(adj, input_dim, hid_dim, out_dim, self.modelParams.num_hop,
                                      self.modelParams.dropout,
                                      feature_inter=self.modelParams.interaction,
                                      activation=self.modelParams.activation,
                                      inter_layer=self.modelParams.num_layer,
                                      feature_fusion=self.modelParams.fusion,
                                      norm_type=self.modelParams.norm_type,
                                      epoch_time=self.modelParams.epochtime,
                                      p=self.modelParams.p,
                                      cluster_num=self.modelParams.cluster_num,
                                      device=self.device).to(self.device)
        return model

    def Feaproprocess(self, get_Dataset, adj, input_dim, hid_dim, out_dim, tdataset=None):
        model = self.setModel(adj, input_dim, hid_dim, out_dim)
        if self.args.disribution_attack == 'random-add' or self.args.disribution_attack == 'random-del':
            mod_edge_index = load_perturbed_adj(self.args.dataset, self.args.disribution_attack, self.args.ptb_rate,
                                         path='./datasets')
            mod_adj = get_Dataset.init_matrix(mod_edge_index)
        else:
            mod_adj = load_perturbed_adj(self.args.dataset, self.args.disribution_attack, self.args.ptb_rate,
                                     path='./datasets')
        if self.cleanFlag:
            mod_adj = adj
        if self.args.inductive_setting:
            get_Dataset.shift_adj = tdataset.adj
            get_Dataset.vallabel = tdataset.label
            get_Dataset, select_idx = privacy_perturbation(get_Dataset, self.args.perturb_nodes_radio,
                                                           self.args.epsilon)
            tdataset, tselect_idx = privacy_perturbation(tdataset, self.args.perturb_nodes_radio, self.args.epsilon)
            privacyData = privacyProcess(device=self.device, args=self.args, dataset=get_Dataset)
            get_Dataset.feature = privacyData.all_privacy_feature_propagation(get_Dataset, adj, select_idx)
            tdataset.feature = privacyData.all_privacy_feature_propagation(tdataset, tdataset.adj, tselect_idx)

            get_Dataset.valFeature = tdataset.feature

            get_Dataset.feature = model.preprocess(adj, get_Dataset.feature)
            get_Dataset.valFeature = model.preprocess(get_Dataset.shift_adj, get_Dataset.valFeature)
        else:
            get_Dataset.shift_adj = mod_adj
            get_Dataset, select_idx = privacy_perturbation(get_Dataset, self.args.perturb_nodes_radio, self.args.epsilon)
            privacyData = privacyProcess(device=self.device, args=self.args, dataset=get_Dataset)
            if not self.noepFlag:
                get_Dataset.feature = privacyData.all_privacy_feature_propagation(get_Dataset, adj, select_idx)
                get_Dataset.valFeature = privacyData.all_privacy_feature_propagation(get_Dataset, get_Dataset.shift_adj,
                                                                                     select_idx)
            get_Dataset.feature = model.preprocess(adj, get_Dataset.feature)
            get_Dataset.valFeature = model.preprocess(get_Dataset.shift_adj, get_Dataset.valFeature)
        return model, get_Dataset
