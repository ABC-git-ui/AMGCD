import time
import torch
import argparse

# Training settings
parser = argparse.ArgumentParser()


parser.add_argument('--epoch_pre', type=int, default=10, help='Number of epochs for pretrain')

parser.add_argument('--epoch_stu', type=int, default=100, help='Max number of epochs for gcn. Default is 400.')

parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')

parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,  help='Weight decay (L2 loss on parameters).')


parser.add_argument('--dataset', type=str, default="CardiovascularInfections", help='dataset.')

parser.add_argument('--no_cuda', action='store_false', default=True, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=25, help='Random seed.')
args = parser.parse_args()
args.device = torch.device('cuda:0' if args.no_cuda and torch.cuda.is_available() else 'cpu')
