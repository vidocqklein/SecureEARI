import argparse

from attacks.attacktrianer import AttackTrainer
from datasets.privacyDataLoader import privacyData_loader
from models import GCN
from paramSet import set_modelParams
from trainer.defenseTrainer import modelTrainer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=str, default='0', help='CUDA id')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'pubmed', 'dblpv7', 'citationv1', 'acmv9'])
    parser.add_argument('--seed', type=int, default=202, help='set Random seed.')
    parser.add_argument('--surrogate', type=str, default='surrogate_GCN', help='select model.')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=0, help="batch size for the mini-batch")  #
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate.')  #
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay (L2 loss on parameters).')
    parser.add_argument('--log_dur', type=int, default=50,
                        help='interval of epochs for log during training.')
    args, _ = parser.parse_known_args()
    device = 'cuda:{}'.format(args.cuda_id) if args.cuda_id != 'cpu' else 'cpu'
    get_Dataset = privacyData_loader(args.dataset, device)
    # load attack config

    # load surrogate
    surrogateParams = set_modelParams('gcn')
    input_dim, hid_dim, out_dim = get_Dataset.feature.shape[1], surrogateParams.hidden, get_Dataset.num_label_class
    surrogate_model = GCN(input_dim, out_dim, hid_dim, surrogateParams.dropout, device=device).to(
        device)
    print('params loading——————', 'lr:', args.lr, 'wd:', args.weight_decay,
          'dropout:', surrogateParams.dropout, 'surrogate_model:', 'gcn')
    print('============================================')
    trainer = modelTrainer(device=device, epoch=args.epochs, model=surrogate_model, dataset=get_Dataset,
                                  args=args)
    trainer.fit(val_do=True)
    output = trainer.predict()

    # attacktrainer = AttackTrainer(device=device, epoch=args.epochs, model=surrogate_model, dataset=get_Dataset,
    #                               args=args)
    # aoutput = attacktrainer.output
