import argparse
import sys

from utils.utils import str2bool

# 获取命令行参数
argv = sys.argv
# 判断是否有命令行参数
if len(sys.argv) > 1:
    argv = sys.argv
    dataset = argv[1]
else:  # 没有命令行参数就默认为NaN（否则会报错）
    dataset = 'NaN'
    print("No command line arguments found.")


# Training hyper-parameters
def dblpv7_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dblpv7', help='name of dataset')
    parser.add_argument('--seed', type=int, default=202, help='set Random seed.')
    parser.add_argument('--cuda_id', type=str, default='0', help='CUDA id')
    parser.add_argument('--model', type=str, default='eari', help='select model.')
    parser.add_argument('--epochs', type=int, default=2000, help='number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=2048, help="batch size for the mini-batch")
    parser.add_argument('--disribution_attack', type=str, default='prbcd', help="disribution attack",
                        choices=['prbcd', 'greedy-rbcd', 'pga', 'GraD', 'random-add','random-del'])
    parser.add_argument('--ptb_rate', type=float, default=0.05, help="disribution attack ptb rate")
    parser.add_argument('--log_dur', type=int, default=50,
                        help='interval of epochs for log during training.')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay (L2 loss on parameters).')
    parser.add_argument('--perturb_nodes_radio', type=float, default=0.5, help='perturb nodes num radio.')
    parser.add_argument('--perturb_sensitive_nodes_radio', type=float, default=1, help='perturb sensitive nodes radio.')
    parser.add_argument('--threshhold', type=float, default=0.8, help='threshhold of fix attrs.')
    parser.add_argument('--m', type=int, default=2, help='the num of privacy attrs.')
    parser.add_argument('--iter', type=int, default=10, help="num of iteration")
    parser.add_argument('--epsilon', type=float, default=8, help="privacy budget")
    parser.add_argument('--seed_list', type=int, default=[202, 66, 99], help="seed list")
    parser.add_argument('--Remarks', type=str, default='NaN', help="Remarks")
    parser.add_argument('--iterNum', type=int, default=0, help="run nums")
    parser.add_argument('--tdataset', type=str, default='citationv1', help='target dataset')
    parser.add_argument('--inductive_setting', type=bool, default=False, help='switch of inductive')
    args, _ = parser.parse_known_args()
    return args

def citationv1_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='citationv1', help='name of dataset')
    parser.add_argument('--seed', type=int, default=202, help='set Random seed.')
    parser.add_argument('--cuda_id', type=str, default='0', help='CUDA id')
    parser.add_argument('--model', type=str, default='eari', help='select model.')
    parser.add_argument('--epochs', type=int, default=2000, help='number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=2048, help="batch size for the mini-batch")
    parser.add_argument('--disribution_attack', type=str, default='prbcd', help="disribution attack",
                        choices=['prbcd', 'greedy-rbcd', 'pga', 'GraD', 'random-add','random-del'])
    parser.add_argument('--ptb_rate', type=float, default=0.05, help="disribution attack ptb rate")
    parser.add_argument('--log_dur', type=int, default=50,
                        help='interval of epochs for log during training.')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay (L2 loss on parameters).')
    parser.add_argument('--perturb_nodes_radio', type=float, default=0.5, help='perturb nodes num radio.')
    parser.add_argument('--perturb_sensitive_nodes_radio', type=float, default=1, help='perturb sensitive nodes radio.')
    parser.add_argument('--threshhold', type=float, default=0.8, help='threshhold of fix attrs.')
    parser.add_argument('--m', type=int, default=2, help='the num of privacy attrs.')
    parser.add_argument('--iter', type=int, default=10, help="num of iteration")
    parser.add_argument('--epsilon', type=float, default=8, help="privacy budget")
    parser.add_argument('--seed_list', type=int, default=[202, 66, 99], help="seed list")
    parser.add_argument('--Remarks', type=str, default='NaN', help="Remarks")
    parser.add_argument('--iterNum', type=int, default=0, help="run nums")
    parser.add_argument('--tdataset', type=str, default='citationv1', help='target dataset')
    parser.add_argument('--inductive_setting', type=bool, default=False, help='switch of inductive')
    args, _ = parser.parse_known_args()
    return args

def acmv9_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='acmv9', help='name of dataset')
    parser.add_argument('--seed', type=int, default=202, help='set Random seed.')
    parser.add_argument('--cuda_id', type=str, default='0', help='CUDA id')
    parser.add_argument('--model', type=str, default='eari', help='select model.')
    parser.add_argument('--epochs', type=int, default=2000, help='number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=2048, help="batch size for the mini-batch")
    parser.add_argument('--disribution_attack', type=str, default='prbcd', help="disribution attack",
                        choices=['prbcd', 'greedy-rbcd', 'pga', 'GraD', 'random-add','random-del'])
    parser.add_argument('--ptb_rate', type=float, default=0.05, help="disribution attack ptb rate")
    parser.add_argument('--log_dur', type=int, default=50,
                        help='interval of epochs for log during training.')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay (L2 loss on parameters).')
    parser.add_argument('--perturb_nodes_radio', type=float, default=0.5, help='perturb nodes num radio.')
    parser.add_argument('--perturb_sensitive_nodes_radio', type=float, default=1, help='perturb sensitive nodes radio.')
    parser.add_argument('--threshhold', type=float, default=0.8, help='threshhold of fix attrs.')
    parser.add_argument('--m', type=int, default=2, help='the num of privacy attrs.')
    parser.add_argument('--iter', type=int, default=10, help="num of iteration")
    parser.add_argument('--epsilon', type=float, default=8, help="privacy budget")
    parser.add_argument('--seed_list', type=int, default=[202, 66, 99], help="seed list")
    parser.add_argument('--Remarks', type=str, default='NaN', help="Remarks")
    parser.add_argument('--iterNum', type=int, default=0, help="run nums")
    parser.add_argument('--tdataset', type=str, default='citationv1', help='target dataset')
    parser.add_argument('--inductive_setting', type=bool, default=False, help='switch of inductive')
    args, _ = parser.parse_known_args()
    return args


def cora_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='name of dataset')
    parser.add_argument('--seed', type=int, default=202, help='set Random seed.')
    parser.add_argument('--cuda_id', type=str, default='0', help='CUDA id')
    parser.add_argument('--model', type=str, default='eari', help='select model.')
    parser.add_argument('--epochs', type=int, default=2000, help='number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=2048, help="batch size for the mini-batch")
    parser.add_argument('--disribution_attack', type=str, default='prbcd', help="disribution attack",
                        choices=['prbcd', 'greedy-rbcd', 'pga', 'GraD', 'random-add','random-del'])
    parser.add_argument('--ptb_rate', type=float, default=0.05, help="disribution attack ptb rate")
    parser.add_argument('--log_dur', type=int, default=50,
                        help='interval of epochs for log during training.')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay (L2 loss on parameters).')
    parser.add_argument('--perturb_nodes_radio', type=float, default=0.5, help='perturb nodes num radio.')
    parser.add_argument('--perturb_sensitive_nodes_radio', type=float, default=1, help='perturb sensitive nodes radio.')
    parser.add_argument('--threshhold', type=float, default=0.8, help='threshhold of fix attrs.')
    parser.add_argument('--m', type=int, default=2, help='the num of privacy attrs.')
    parser.add_argument('--iter', type=int, default=10, help="num of iteration")
    parser.add_argument('--epsilon', type=float, default=8, help="privacy budget")
    parser.add_argument('--seed_list', type=int, default=[202, 66, 99], help="seed list")
    parser.add_argument('--Remarks', type=str, default='NaN', help="Remarks")
    parser.add_argument('--iterNum', type=int, default=0, help="run nums")
    parser.add_argument('--tdataset', type=str, default='citationv1', help='target dataset')
    parser.add_argument('--inductive_setting', type=bool, default=False, help='switch of inductive')
    args, _ = parser.parse_known_args()
    return args

def citeseer_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='citeseer', help='name of dataset')
    parser.add_argument('--seed', type=int, default=202, help='set Random seed.')
    parser.add_argument('--cuda_id', type=str, default='0', help='CUDA id')
    parser.add_argument('--model', type=str, default='eari', help='select model.')
    parser.add_argument('--epochs', type=int, default=2000, help='number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=2048, help="batch size for the mini-batch")
    parser.add_argument('--disribution_attack', type=str, default='prbcd', help="disribution attack",
                        choices=['prbcd', 'greedy-rbcd', 'pga', 'GraD', 'random-add','random-del'])
    parser.add_argument('--ptb_rate', type=float, default=0.05, help="disribution attack ptb rate")
    parser.add_argument('--log_dur', type=int, default=50,
                        help='interval of epochs for log during training.')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay (L2 loss on parameters).')
    parser.add_argument('--perturb_nodes_radio', type=float, default=0.5, help='perturb nodes num radio.')
    parser.add_argument('--perturb_sensitive_nodes_radio', type=float, default=1, help='perturb sensitive nodes radio.')
    parser.add_argument('--threshhold', type=float, default=0.8, help='threshhold of fix attrs.')
    parser.add_argument('--m', type=int, default=2, help='the num of privacy attrs.')
    parser.add_argument('--iter', type=int, default=10, help="num of iteration")
    parser.add_argument('--epsilon', type=float, default=8, help="privacy budget")
    parser.add_argument('--seed_list', type=int, default=[202, 66, 99], help="seed list")
    parser.add_argument('--Remarks', type=str, default='NaN', help="Remarks")
    parser.add_argument('--iterNum', type=int, default=0, help="run nums")
    parser.add_argument('--tdataset', type=str, default='citationv1', help='target dataset')
    parser.add_argument('--inductive_setting', type=bool, default=False, help='switch of inductive')

    args, _ = parser.parse_known_args()
    return args

def pubmed_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pubmed', help='name of dataset')
    parser.add_argument('--seed', type=int, default=202, help='set Random seed.')
    parser.add_argument('--cuda_id', type=str, default='0', help='CUDA id')
    parser.add_argument('--model', type=str, default='eari', help='select model.')
    parser.add_argument('--epochs', type=int, default=2000, help='number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=2048, help="batch size for the mini-batch")
    parser.add_argument('--disribution_attack', type=str, default='prbcd', help="disribution attack",
                        choices=['prbcd', 'greedy-rbcd', 'pga', 'GraD', 'random-add','random-del'])
    parser.add_argument('--ptb_rate', type=float, default=0.05, help="disribution attack ptb rate")
    parser.add_argument('--log_dur', type=int, default=50,
                        help='interval of epochs for log during training.')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay (L2 loss on parameters).')
    parser.add_argument('--perturb_nodes_radio', type=float, default=0.5, help='perturb nodes num radio.')
    parser.add_argument('--perturb_sensitive_nodes_radio', type=float, default=1, help='perturb sensitive nodes radio.')
    parser.add_argument('--threshhold', type=float, default=0.8, help='threshhold of fix attrs.')
    parser.add_argument('--m', type=int, default=2, help='the num of privacy attrs.')
    parser.add_argument('--iter', type=int, default=10, help="num of iteration")
    parser.add_argument('--epsilon', type=float, default=8, help="privacy budget")
    parser.add_argument('--seed_list', type=int, default=[202, 66, 99], help="seed list")
    parser.add_argument('--Remarks', type=str, default='NaN', help="Remarks")
    parser.add_argument('--iterNum', type=int, default=0, help="run nums")
    parser.add_argument('--tdataset', type=str, default='citationv1', help='target dataset')
    parser.add_argument('--inductive_setting', type=bool, default=False, help='switch of inductive')

    args, _ = parser.parse_known_args()
    return args

# Model hyper-parameters
def EARI_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', type=int, default=256, help='number of hidden dim')
    parser.add_argument('--num_layer', type=int, default=2, help='number of interaction layer')
    parser.add_argument('--num_hop', type=int, default=6, help='number of hop information')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate (1 - keep probability).')
    parser.add_argument('--interaction', type=str, default='attention',
                        help='feature interaction type of HopGNN,choose attention,gcn,mlp')
    parser.add_argument('--fusion', type=str, default='mean', help='feature fusion type')
    parser.add_argument('--activation', type=str, default='relu', help="activation function")
    parser.add_argument('--norm_type', type=str, default='ln', help="the normalization type")
    parser.add_argument('--epochtime', type=int, default=5, help="every epochtime to transfer")
    parser.add_argument('--p', type=float, default="0.2", help="the ratio of transferred nodes")
    parser.add_argument('--cluster_num', type=int, default=100, help="num of clusters")

    args, _ = parser.parse_known_args()
    return args

def GCN_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=0, help="batch size for the mini-batch")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--log_dur', type=int, default=50,
                        help='interval of epochs for log during training.')
    args, _ = parser.parse_known_args()
    return args

def GUARD_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=0, help="batch size for the mini-batch")  #
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--log_dur', type=int, default=50,  # 50
                        help='interval of epochs for log during training.')
    args, _ = parser.parse_known_args()
    return args
def JaccardGCN_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=0, help="batch size for the mini-batch")  #
    parser.add_argument('--log_dur', type=int, default=50,  # 50
                        help='interval of epochs for log during training.')
    args, _ = parser.parse_known_args()
    return args

def SVDGCN_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=200)#500
    parser.add_argument('--batch_size', type=int, default=0, help="batch size for the mini-batch")  #
    parser.add_argument('--log_dur', type=int, default=50,  # 50
                        help='interval of epochs for log during training.')  #
    args, _ = parser.parse_known_args()
    return args
def ElasticGNN_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--lambda1', type=float, default=3)
    parser.add_argument('--lambda2', type=float, default=3)
    parser.add_argument('--L21', type=str2bool, default=True)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=5)#500
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=0, help="batch size for the mini-batch")  #
    parser.add_argument('--log_dur', type=int, default=1,#50
                        help='interval of epochs for log during training.')  #


    args, _ = parser.parse_known_args()
    return args

def MedianGCN_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=0, help="batch size for the mini-batch")  #
    parser.add_argument('--log_dur', type=int, default=50,
                        help='interval of epochs for log during training.')  #


    args, _ = parser.parse_known_args()
    return args

def ProGNN_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug mode')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--only_gcn', action='store_true',
                        default=False, help='test the performance of gcn without other components')
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
    parser.add_argument('--alpha', type=float, default=5e-4, help='weight of l1 norm')
    parser.add_argument('--beta', type=float, default=1.5, help='weight of nuclear norm')
    parser.add_argument('--gamma', type=float, default=1, help='weight of l2 norm')
    parser.add_argument('--lambda_', type=float, default=0, help='weight of feature smoothing')
    parser.add_argument('--phi', type=float, default=0, help='weight of symmetric loss')
    parser.add_argument('--inner_steps', type=int, default=2, help='steps for inner optimization')
    parser.add_argument('--outer_steps', type=int, default=1, help='steps for outer optimization')
    parser.add_argument('--lr_adj', type=float, default=0.01, help='lr for training adj')
    parser.add_argument('--symmetric', action='store_true', default=False,
                        help='whether use symmetric matrix')
    parser.add_argument('--batch_size', type=int, default=0, help="batch size for the mini-batch")  #


    args, _ = parser.parse_known_args()
    return args


def NoisyGNN_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=15, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200,#200
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--beta_max', type=float, default=0.15)
    parser.add_argument('--beta_min', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=0, help="batch size for the mini-batch")  #
    parser.add_argument('--log_dur', type=int, default=50,#50
                        help='interval of epochs for log during training.')  #

    args, _ = parser.parse_known_args()
    return args

def RGCN_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=15, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200,#200
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batch_size', type=int, default=0, help="batch size for the mini-batch")  #
    parser.add_argument('--log_dur', type=int, default=50,#50
                        help='interval of epochs for log during training.')  #

    args, _ = parser.parse_known_args()
    return args

def set_modelParams(model_name):
    if model_name == "eari":
        modelParams = EARI_params()
    elif model_name == "gcn":
        modelParams = GCN_params()
    elif model_name == "GUARD":
        modelParams = GUARD_params()
    elif model_name == "ElasticGNN":
        modelParams = ElasticGNN_params()
    elif model_name == "MedianGCN":
        modelParams = MedianGCN_params()
    elif model_name == "NoisyGNN":
        modelParams = NoisyGNN_params()
    elif model_name == "JaccardGCN":
        modelParams = JaccardGCN_params()
    elif model_name == "SVDGCN":
        modelParams = SVDGCN_params()
    elif model_name == "RGCN":
        modelParams = RGCN_params()
    return modelParams


def set_trainParams(input_dataset="NaN"):
    if dataset == "dblpv7" or input_dataset == "dblpv7":
        args = dblpv7_params()
    elif dataset == "citationv1" or input_dataset == "citationv1":
        args = citationv1_params()
    elif dataset == "acmv9" or input_dataset == "acmv9":
        args = acmv9_params()
    elif dataset == "cora" or input_dataset == "cora":
        args = cora_params()
    elif dataset == "citeseer" or input_dataset == "citeseer":
        args = citeseer_params()
    elif dataset == "pubmed" or input_dataset == "pubmed":
        args = pubmed_params()

    return args
