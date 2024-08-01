from modules.enrichFea import enrichFea
from paramSet import set_trainParams
from utils.utils import set_seed

# python main.py dblpv7 --model hop_represent
if __name__ == "__main__":
    args = set_trainParams()
    device = 'cuda:{}'.format(args.cuda_id) if args.cuda_id != 'cpu' else 'cpu'
    # args.Remarks = 'test'
    enrichFea(device, args)





