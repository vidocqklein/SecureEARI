import argparse
import os

import torch
import yaml
from yaml import SafeLoader
import sys
sys.path.append('..')
from attacks import attacks_map
from attacks.attacktrianer import AttackTrainer
from datasets.privacyDataLoader import privacyData_loader
from models import model_map
from paramSet import set_modelParams
import os.path as osp

from utils.utils import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=str, default='0', help='CUDA id')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'pubmed', 'dblpv7', 'citationv1', 'acmv9'])
    parser.add_argument('--seed', type=int, default=202, help='set Random seed.')
    parser.add_argument('--surrogate', type=str, default='surrogate_GCN', help='select model.')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate.')  # 初始学习率,5*10的﹣2次方=0.05, 1e-2
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay (L2 loss on parameters).')
    parser.add_argument('--log_dur', type=int, default=50,
                        help='interval of epochs for log during training.')  # 训练期间的时间日志记录时间间隔
    parser.add_argument('--log_attack_dur', type=int, default=10,
                        help='interval of epochs for log during attack training.')
    parser.add_argument('--attack', type=str, default='pga',
                        choices=['prbcd', 'lrbcd', 'greedy-rbcd', 'pga', 'random'])
    parser.add_argument('--victim', type=str, default='normal')
    parser.add_argument('--ptb_rate', type=float, default=0.2)
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--save_prefix', type=str, default="")
    args, _ = parser.parse_known_args()

    device = 'cuda:{}'.format(args.cuda_id) if args.cuda_id != 'cpu' else 'cpu'
    attack_dataset = ['cora', 'citeseer', 'pubmed', 'dblpv7', 'citationv1', 'acmv9']
    attacks = ['prbcd', 'greedy-rbcd', 'pga', 'lrbcd']


    for dataset in attack_dataset:
        args.dataset = dataset
        for att in attacks:
            args.attack = att
            get_Dataset = privacyData_loader(args.dataset, device)
            config_file = osp.join(osp.expanduser('./configs'), args.attack + '.yaml')
            attack_config = yaml.load(open(config_file), Loader=SafeLoader)[args.dataset]
            surrogate = attack_config['surrogate']
            surrogateParams = set_modelParams('gcn')
            input_dim, hid_dim, out_dim = get_Dataset.feature.shape[1], surrogateParams.hidden, get_Dataset.num_label_class
            surrogate_model = model_map[surrogate](input_dim, out_dim, hid_dim, surrogateParams.dropout,
                                                   device=device).to(device)
            print('params loading——————', 'lr:', args.lr, 'wd:', args.weight_decay,
                  'dropout:', surrogateParams.dropout, 'surrogate_model:', surrogate)
            print('============================================')
            attacktrainer = AttackTrainer(device=device, epoch=args.epochs, model=surrogate_model, dataset=get_Dataset,
                                          args=args)
            output = attacktrainer.output
            attacker = attacks_map[args.attack](attack_config=attack_config, dataset=get_Dataset,
                                                model=surrogate_model, device=device, dataset_name=args.dataset)
            ptb_rates = [0.03, 0.05, 0.1, 0.15, 0.2]
            for ptb_rate in ptb_rates:
                n_perturbs = int(ptb_rate * (get_Dataset.num_edges // 2))

                print('attack start——————', 'attack name:', args.attack, 'dataset:', dataset, 'ptb_rate:',
                      ptb_rate)
                print('============================================')
                attacker.attack(n_perturbs)
                mod_adj, _ = attacker.get_perturbations()
                print('attack complicated!!!')
                print('============================================')
                dataset_path = os.path.dirname(os.getcwd())
                filename = os.path.join(dataset_path, 'datasets', dataset,
                                        f"{args.attack}-{dataset}-{ptb_rate}.pt")
                torch.save(mod_adj, filename)






if __name__ == '__main__':
    main()