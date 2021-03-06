import argparse
import os
import random
import sys
import torch

parser = argparse.ArgumentParser()

def get_run_script():
    run_script = 'python'
    for e in sys.argv:
        run_script += (' ' + e)
    return run_script


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_params(params):
    if params.tags is None or params.num_epochs < 10:
        assert False

def get_args():
    params = parser.parse_args()
    params.run_script = get_run_script()

    # tag&save
    params.tags = [e for e in params.tags.split(',')] if params.tags is not None else ['test']
    params.tags.append(params.name)

    # random_seed
    torch.manual_seed(params.random_seed)
    torch.cuda.manual_seed_all(params.random_seed)
    random.seed(params.random_seed)

    # params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params.device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        if params.gpu is not None:
            params.gpu = [int(e) for e in params.gpu.split(',')]
            torch.cuda.set_device(params.gpu[0])

    params.weight_path = f'weight/{params.name}/'
    params.save_path = f'log/{params.name}.log'
    os.makedirs(params.weight_path, exist_ok=True)

    return params

def print_args(params):
    info = '\n[args]\n'
    for sub_args in parser._action_groups:
        if sub_args.title in ['positional arguments', 'optional arguments']:
            continue
        size_sub = len(sub_args._group_actions)
        info += f'  {sub_args.title} ({size_sub})\n'
        for i, arg in enumerate(sub_args._group_actions):
            prefix = '-'
            info += f'      {prefix} {arg.dest:20s}: {getattr(params, arg.dest)}\n'
    info += '\n'
    print(info)

base_args = parser.add_argument_group('Base args')
base_args.add_argument('--run_script')
base_args.add_argument('--use_wandb', type=str2bool, default='0')
base_args.add_argument('--save_path')
base_args.add_argument('--device', type=str, default='cpu')
base_args.add_argument('--gpu', type=str)
base_args.add_argument('--num_workers', type=int, default=1)
base_args.add_argument('--machine_name', type=str, default='local')
base_args.add_argument('--base_path', type=str)
base_args.add_argument('--test_per_epoch', type=int, default=2)

wandb_args = parser.add_argument_group('wandb args')
wandb_args.add_argument('--project', type=str)
wandb_args.add_argument('--name', type=str)
wandb_args.add_argument('--tags')

train_args = parser.add_argument_group('Train args')
train_args.add_argument('--random_seed', type=int, default=2)
train_args.add_argument('--num_epochs', type=int, default=50)
train_args.add_argument('--train_batch', type=int, default=16)
train_args.add_argument('--test_batch', type=int, default=16)
train_args.add_argument('--learning_rate', type=float, default=0.001)

network_args = parser.add_argument_group('Network args')
network_args.add_argument('--model_arc', type=str, default='CategoryCNN')
network_args.add_argument('--dropout_rate', type=float, default=0.1)

args = get_args()

if __name__ == '__main__':
    args = get_args()
    print_args(args)