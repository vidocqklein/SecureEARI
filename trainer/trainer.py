import datetime
import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from modules.privacyProcess import privacyProcess
from trainer.metric_gain import metric_compute
from utils.utils import set_seed
from tqdm import tqdm
writer = SummaryWriter(log_dir='out_log')

class Trainer:
    def __init__(self,device,epoch, model, dataset, args, modelParams, t_val_mask=None, t_test_mask=None):
        super().__init__()
        self.device = device
        self.epoch = epoch
        self.model = model
        self.features = dataset.feature
        self.valFeatures = dataset.valFeature
        self.labels = dataset.label
        self.vallabels = dataset.vallabel
        self.adj = dataset.adj
        self.shift_adj = dataset.shift_adj
        self.train_mask = dataset.train_mask
        self.t_val_mask = t_val_mask
        self.t_test_mask = t_test_mask
        self.val_mask = dataset.val_mask
        self.test_mask = dataset.test_mask
        self.num_label_class = dataset.num_label_class
        self.args = args
        self.batch_size = args.batch_size
        self.modelParams = modelParams
        self.hopF = False
        # set_seed(202)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.metric_get = metric_compute(metricType='acc')
        # self.privacyProcess = privacyProcess(device=self.device, args=self.args, dataset=dataset)
        torch.autograd.set_detect_anomaly(True)
        self.best_test_acc, self.output = self.excute()


    def excute(self):
        train_loss_list, val_loss_list, train_acc_list, val_acc_list, test_acc_list = [], [], [], [], []
        train_F1score_list, val_F1score_list, test_F1score_list = [], [], []
        best_val_acc, best_test_acc = 0, 0
        best_epoch = 0
        best_val_f1 = 0
        best_test_f1 = 0
        current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # timestamp
        if self.args.Remarks == 'NaN':
            filename = os.path.join('out_log', f"{self.args.dataset}_{self.args.model}_{self.args.iterNum}_{current_time}.txt" )
        else:
            if self.args.inductive_setting:
                filename = os.path.join('out_log',
                                        f"{self.args.dataset}->{self.args.tdataset}_{self.args.model}_{self.args.Remarks}_{self.args.iterNum}_{current_time}.txt")
            else:
                filename = os.path.join('out_log', f"{self.args.dataset}_{self.args.model}_{self.args.Remarks}_{self.args.iterNum}_{current_time}.txt")
        message =('params loading——————', 'dataset:',self.args.dataset, 'model:', self.args.model, 'lr:', self.args.lr, 'wd:', self.args.weight_decay, 'hop:', self.modelParams.num_hop,
              'dropout:',
              self.modelParams.dropout,
              self.modelParams.interaction, 'inter_layers:', self.modelParams.num_layer, self.modelParams.fusion,
              self.modelParams.norm_type, 'disribution_attack:', self.args.Remarks,
                  'ptb_rate:', self.args.ptb_rate, 'perturb_nodes_radio:', self.args.perturb_nodes_radio,
                  'epsilon:', self.args.epsilon, 'p:', self.modelParams.p, 'cluster_num:', self.modelParams.cluster_num, 'epoch_time:', self.modelParams.epochtime)
        self.log_message(message, filename)
        total_time = 0
        # for iter in tqdm(range(self.epoch)):
        for iter in range(self.epoch):
            start_time = time.time()
            train_metrics, train_output = self.loop(stage='train', mask=self.train_mask, epoch_now=iter)
            if self.args.inductive_setting:
                val_metrics, val_output = self.loop(stage='val', mask=self.t_val_mask, epoch_now=iter)
                test_metrics, test_output = self.loop(stage='test', mask=self.t_test_mask, epoch_now=iter)
            else:
                val_metrics, val_output = self.loop(stage='val', mask=self.val_mask, epoch_now=iter)
                test_metrics, test_output = self.loop(stage='test', mask=self.test_mask, epoch_now=iter)

            train_loss_list.append(train_metrics['loss'])
            val_loss_list.append(val_metrics['loss'])
            train_acc, val_acc, test_acc = train_metrics['acc'], val_metrics['acc'], test_metrics['acc']
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            test_acc_list.append(test_acc)
            train_F1, val_F1, test_F1 = train_metrics['F1score'], val_metrics['F1score'], test_metrics['F1score']
            train_F1score_list.append(train_metrics['F1score'])
            val_F1score_list.append(val_metrics['F1score'])
            test_F1score_list.append(test_metrics['F1score'])

            # update best test via val
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_F1
                best_epoch = iter
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_f1 = test_F1
            if (iter + 1) % self.args.log_dur == 0:
                # message = (
                #     "Epoch {:4d}, Train_loss {:.4f}, Val_loss {:.4f}, train_acc {:.4f}, train_F1 {:.4f}, val_acc {:.4f}, val_F1 {:.4f}, test_acc{:.4f}, test_F1 {:.4f}".format(
                #         iter + 1, np.mean(train_loss_list), np.mean(val_loss_list), train_acc, train_F1, val_acc, val_F1, test_acc, test_F1))
                message = (
                    "Epoch {:4d}, Train_loss {:.4f}, Val_loss {:.4f}, train_acc {:.4f}, train_F1 {:.4f}, val_acc {:.4f}, val_F1 {:.4f}, test_acc{:.4f}, test_F1 {:.4f}".format(
                        iter + 1, np.mean(train_loss_list), np.mean(val_loss_list), train_acc, train_F1,  val_acc, val_F1, test_acc, test_F1))
                print(message)
                self.tensorBoard_write(np.mean(train_loss_list), np.mean(val_loss_list), train_acc, train_F1, val_acc, val_F1, test_acc, test_F1, iter + 1)

                self.log_message(message, filename)
            end_time = time.time()
            epoch_time = end_time - start_time
            total_time += epoch_time
        writer.close()
        message = ("Best at {} epoch, Val Accuracy {:.2%}, Val F1-Score {:.2%}, Test Accuracy {:.2%}, Test F1-Score {:.2%}".format(best_epoch, best_val_acc, best_val_f1, best_test_acc, best_test_f1))

        print(message)
        self.log_message(message, filename)
        average_time_per_epoch = total_time / self.epoch
        message=(f"Average time per epoch: {average_time_per_epoch} seconds")
        self.log_message(message, filename)
        return best_test_acc, val_output
    def loop(self, stage, mask, epoch_now ):
        total_loss = []
        total_output = []
        total_label = []
        mask_indices = torch.nonzero(mask).squeeze().cpu().numpy()
        for i in range(0, len(mask_indices), self.batch_size):
            # generate batch index , features, label

            index = mask_indices[i:i + self.batch_size]

            if stage == 'train':
                batch_label = self.labels[index].to(self.device)
                batch_features = self.features[index].to(self.device)
                batch_adj = self.adj[index][:, index]
                self.model.train()
                self.optimizer.zero_grad()
                output, h_loss, mc_loss, o_loss = self.metric_get.get_loss(self.model, batch_features, batch_label,
                                                                           batch_adj, stage, epoch_now, self.hopF)
                loss = 0.5 * h_loss + 0.3 * mc_loss + 0.2 * o_loss
                with torch.autograd.detect_anomaly():
                    loss.backward()
                self.optimizer.step()
                total_label.append(batch_label)
            else:
                if self.args.inductive_setting:
                    batch_val_label = self.vallabels[index].to(self.device)
                else:
                    batch_val_label = self.labels[index].to(self.device)
                with torch.no_grad():
                    batch_features = self.valFeatures[index].to(self.device)
                    batch_adj = self.shift_adj[index][:, index]
                    # batch_adj = self.adj[index][:, index]
                    self.model.eval()
                    output, loss = self.metric_get.get_loss(self.model, batch_features, batch_val_label, batch_adj, stage, epoch_now, self.hopF)
                    total_label.append(batch_val_label)
            total_loss.append(loss.cpu().item())
            total_output.append(output)
            # total_label.append(batch_label)
        # self.features, self.hopF = self.privacyProcess.Private_attribute_perceive(self.features, self.adj[mask_indices][:, mask_indices], mask_indices, stage, total_output, self.hopF)
        loss = np.mean(total_loss)
        total_output = torch.cat(total_output, dim=0)
        total_label = torch.cat(total_label)
        acc = self.metric_get.get_accuracy(total_output, total_label)
        F1score = self.metric_get.F1score(total_output, total_label, self.num_label_class)
        log_info = {'loss': loss, 'acc': acc, 'F1score': F1score}
        return log_info, total_output

    def log_message(self,message, filename):
        with open(filename, 'a') as f:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{timestamp} - {message}\n")

    def tensorBoard_write(self, train_loss, val_loss, train_acc, train_F1, val_acc, val_F1, test_acc, test_F1, iter):
        writer.add_scalar(tag="train_loss",
                          scalar_value=train_loss,
                          global_step=iter
                          )
        writer.add_scalar(tag="val_loss",
                          scalar_value=val_loss,
                          global_step=iter
                          )
        writer.add_scalar(tag="train_acc",
                          scalar_value=train_acc,
                          global_step=iter
                          )
        writer.add_scalar(tag="train_F1",
                          scalar_value=train_F1,
                          global_step=iter
                          )
        writer.add_scalar(tag="val_acc",
                          scalar_value=val_acc,
                          global_step=iter
                          )
        writer.add_scalar(tag="val_F1",
                          scalar_value=val_F1,
                          global_step=iter
                          )
        writer.add_scalar(tag="test_acc",
                          scalar_value=test_acc,
                          global_step=iter
                          )
        writer.add_scalar(tag="test_F1",
                          scalar_value=test_F1,
                          global_step=iter
                          )


