import datetime
import os

import numpy as np
import torch
from trainer.metric_gain import metric_compute
from utils.utils import set_seed

class AttackTrainer:
    def __init__(self,device,epoch, model, dataset, args):
        super().__init__()
        self.device = device
        self.epoch = epoch
        self.model = model
        self.features = dataset.feature
        self.valFeatures = dataset.valFeature
        self.labels = dataset.label
        self.adj = dataset.adj
        self.edge_index = dataset.edge_index
        self.train_mask = dataset.train_mask
        self.val_mask = dataset.val_mask
        self.test_mask = dataset.test_mask
        self.args = args
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.metric_get = metric_compute(metricType='acc')
        torch.autograd.set_detect_anomaly(True)
        self.best_test_acc, self.output = self.excute()


    def excute(self):
        train_loss_list, val_loss_list, train_acc_list, val_acc_list, test_acc_list = [], [], [], [], []
        best_val_acc, best_test_acc = 0, 0
        best_epoch = 0
        all_mask =self.train_mask+ self.val_mask+ self.test_mask
        for iter in range(self.epoch):
            train_metrics, train_output = self.loop(stage='train', mask=self.train_mask )
            # val_metrics, val_output = self.loop(stage='val', mask=self.val_mask)
            # test_metrics, test_output = self.loop(stage='test', mask=self.test_mask)
            attack_metrics, attack_output = self.loop(stage='val', mask=all_mask)
            train_loss_list.append(train_metrics['loss'])
            val_loss_list.append(attack_metrics['loss'])
            train_acc, val_acc= train_metrics['acc'], attack_metrics['acc']
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)

            # update best test via val
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = iter

            if (iter + 1) % self.args.log_dur == 0:
                message = (
                    "Epoch {:4d}, Train_loss {:.4f}, Val_loss {:.4f}, train_acc {:.4f}, val_acc {:.4f}".format(
                        iter + 1, np.mean(train_loss_list), np.mean(val_loss_list), train_acc, val_acc))
                print(message)

        message = ("Best at {} epoch, Val Accuracy {:.4f}".format(best_epoch, best_val_acc))
        print(message)
        return best_test_acc, attack_output
    def loop(self, stage, mask):
        mask_indices = torch.nonzero(mask).squeeze().cpu().numpy()
        mask_label = self.labels[mask_indices].to(self.device)

        if stage == 'train':
            self.model.train()
            self.optimizer.zero_grad()
            output, loss = self.metric_get.get_gcn_loss(self.model, mask_indices, self.features, mask_label, self.edge_index)
            with torch.autograd.detect_anomaly():
                loss.backward()
            self.optimizer.step()
        else:
            with torch.no_grad():
                self.model.eval()
                output, loss = self.metric_get.get_gcn_loss(self.model, mask_indices, self.features, mask_label, self.edge_index)
        loss = loss.cpu().item()
        acc = self.metric_get.get_accuracy(output, mask_label)
        log_info = {'loss': loss, 'acc': acc}
        return log_info, output



