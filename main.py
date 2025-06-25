import torch
import argparse
import shutil
from loguru import logger
from data.data_loader import load_data
import os
import numpy as np
import random
from datetime import datetime

import MDH_train 


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def run():
    seed_everything(42)
    args = load_config()
    timestamp = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")

    base_path = 'save/'+args.info+'-'+str(args.code_length)+'-'+ timestamp +'/'
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    args.base_path = base_path
    logger.add(base_path + 'train.log')
    logger.info(args)

    ######
    code_path = base_path+'code/'
    os.makedirs(code_path)
    folders_to_copy = ["models"]  #
    files_to_copy = ["MDH_train.py", "main.py", "run.sh"]  #
    #
    for folder in folders_to_copy:
        src_folder = os.path.join('./', folder)
        dst_folder = os.path.join(code_path, folder)
        shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)

    #
    for file in files_to_copy:
        src_file = os.path.join('./', file)
        dst_file = os.path.join(code_path, file)
        shutil.copy2(src_file, dst_file)
    ######

    torch.backends.cudnn.benchmark = True

    # Load dataset
    query_dataloader, train_dataloader, retrieval_dataloader = load_data(
        args.dataset,
        args.root,
        args.num_query,
        args.num_samples,
        args.batch_size,
        args.num_workers,
    )

    for code_length in args.code_length:
        mAP = MDH_train.train(query_dataloader, train_dataloader, retrieval_dataloader, code_length, args)
        logger.info('[code_length:{}][map:{:.4f}]'.format(code_length, mAP))


def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('--dataset',
                        help='Dataset name.')
    parser.add_argument('--root',
                        help='Path of dataset')
    parser.add_argument('--batch-size', default=16, type=int,
                        help='Batch size.(default: 16)')
    parser.add_argument('--lr', default=2.5e-4, type=float,
                        help='Learning rate.(default: 1e-4)')
    parser.add_argument('--wd', default=1e-4, type=float,
                        help='Weight Decay.(default: 1e-4)')
    parser.add_argument('--optim', default='SGD', type=str,
                        help='Optimizer')
    parser.add_argument('--code-length', default='12,24,32,48', type=str,
                        help='Binary hash code length.(default: 12,24,32,48)')
    parser.add_argument('--max-iter', default=40, type=int,
                        help='Number of iterations.(default: 40)')
    parser.add_argument('--max-epoch', default=30, type=int,
                        help='Number of epochs.(default: 30)')
    parser.add_argument('--num-query', default=1000, type=int,
                        help='Number of query data points.(default: 1000)')
    parser.add_argument('--num-samples', default=2000, type=int,
                        help='Number of sampling data points.(default: 2000)')
    parser.add_argument('--num-workers', default=4, type=int,
                        help='Number of loading data threads.(default: 4)')
    parser.add_argument('--topk', default=-1, type=int,
                        help='Calculate map of top k.(default: all)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='Using gpu.(default: False)')
    parser.add_argument('--gamma', default=200, type=float,
                        help='Hyper-parameter.(default: 200)')
    parser.add_argument('--info', default='Trivial',
                        help='Train info')
    parser.add_argument('--save_ckpt', default='checkpoints/',
                        help='result_save')
    parser.add_argument('--lr-step', default='40', type=str,
                        help='lr decrease step.(default: 40)')
    parser.add_argument('--align-step', default=50, type=int,
                        help='Step of start aligning.(default: 50)')
    parser.add_argument('--pretrain', action='store_true',
                        help='Using image net pretrain')
    parser.add_argument('--quan-loss', action='store_true',
                        help='Using quan_loss')
    parser.add_argument('--lambd-sp', default=0.1, type=float,
                        help='Hyper-parameter.(default: 1)')
    parser.add_argument('--lambd-ch', default=0.1, type=float,
                        help='Hyper-parameter.(default: 1)')
    parser.add_argument('--lambd', default=0.1, type=float,
                        help='Hyper-parameter.(default: 1)')
    parser.add_argument('--momen', default=0.9, type=float,
                        help='Hyper-parameter.(default: 0.9)')
    parser.add_argument('--nesterov', action='store_true',
                        help='Using SGD nesterov')
    parser.add_argument('--num_classes', default=2000, type=int,
                        help='Number of sampling data points.(default: 2000)')
    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    # Hash code length
    args.code_length = list(map(int, args.code_length.split(',')))
    args.lr_step = list(map(int, args.lr_step.split(',')))

    return args


if __name__ == '__main__':
    run()
