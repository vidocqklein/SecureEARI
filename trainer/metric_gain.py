import torch
from torch_geometric.utils import f1_score


class metric_compute:
    def __init__(self, metricType):
        super().__init__()
        self.metricType = metricType
        self.total = torch.tensor(0)
        self.correct = torch.tensor(0)
        self.reduction = 'mean'

    def update_state(self, output, labels):
        output = output.cpu()
        labels = labels.cpu()
        if len(labels.shape) == 2:
            labels = torch.argmax(labels, dim=1)
        pred_y = torch.max(output, dim=1)[1]
        self.correct += torch.sum(pred_y == labels)
        self.total += labels.numel()

    def result(self):
        if self.total == 0:
            return None
        if self.reduction == 'mean':
            return self.correct.float() / self.total
        elif self.reduction == 'sum':
            return self.correct
        else:
            pass
    def reset_states(self):
        self.total = torch.tensor(0)
        self.correct = torch.tensor(0)

    def F1score(self, output, labels, num_classes):
        if len(labels.shape) == 2:
            labels = torch.argmax(labels, dim=1)
        pred_y = torch.max(output, dim=1)[1]
        f1 = f1_score(pred_y, labels, num_classes)
        macro_f1_score = f1.mean().item()
        return macro_f1_score
    def get_accuracy(self, output, labels):
        if len(labels.shape) == 2:
            labels = torch.argmax(labels, dim=1)
        pred_y = torch.max(output, dim=1)[1]
        correct = torch.sum(pred_y == labels)
        return correct.item() * 1.0 / len(labels)

    def get_loss(self, model, features, labels, adj, stage, epoch_now, hopF):
        loss_func = torch.nn.CrossEntropyLoss()
        if stage == 'train':
            output, mc_loss, o_loss = model(features, adj, stage, epoch_now, hopF)
            loss = loss_func(output, labels) + 1e-9
            return output, loss, mc_loss, o_loss
        else:
            output = model(features, adj, stage, epoch_now, False)
            loss = loss_func(output, labels) + 1e-9
            return output, loss



    def get_gcn_loss(self, model, mask_indices, features, labels, edge_index):
        loss_func = torch.nn.CrossEntropyLoss()
        output = model(features, edge_index)
        loss = loss_func(output[mask_indices], labels) + 1e-9
        return output[mask_indices], loss


    def get_surrogate_batch_loss(self, model, features, labels, adj):
        loss_func = torch.nn.CrossEntropyLoss()
        output = model(features, adj)
        loss = loss_func(output, labels) + 1e-9
        return output, loss


    def get_surrogate_idx_loss(self, model, features, labels, adj):
        loss_func = torch.nn.CrossEntropyLoss()
        output = model(features, adj)
        loss = loss_func(output, labels) + 1e-9
        return output, loss


    def get_surrogate_loss(self, model, mask_indices, features, labels, adj):
        loss_func = torch.nn.CrossEntropyLoss()
        output = model(features, adj)
        loss = loss_func(output[mask_indices], labels) + 1e-9
        return output[mask_indices], loss
