from __future__ import division
from __future__ import print_function
from utils import load_citation, accuracy
import time
import argparse
import torch
import numpy as np
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
from scipy import sparse
from torch.optim.lr_scheduler import MultiStepLR,StepLR

import torch.nn.functional as F
import torch.optim as optim
from models import GCN
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="cora",help='Dataset to use.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--patience', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--l1', type=float, default=0.05,
                    help='Weight decay (L1 loss on parameters).')
parser.add_argument('--hid1', type=int, default=13,
                    help='Number of hidden units.')
parser.add_argument('--hid2', type=int, default=25,
                    help='Number of hidden units.')
parser.add_argument('--smoo', type=float, default=0.5,
                    help='Smooth for Res layer')
parser.add_argument('--dropout', type=float, default=0.9,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--normalization', type=str, default='AugNormAdj',
                    choices=['AugNormAdj'],
                    help='Normalization method for the adjacency matrix.')

parser.add_argument('--order_1',type=int, default=1)
parser.add_argument('--sct_inx1', type=int, default=1)
parser.add_argument('--order_2',type=int, default=1)
parser.add_argument('--sct_inx2', type=int, default=3)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
#adj, features, labels, idx_train, idx_val, idx_test = load_data()
adj,A_tilde,adj_sct1,adj_sct2,adj_sct4,features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, args.normalization,args.cuda)
# Model and optimizer
model = GCN(nfeat=features.shape[1],
            para3=args.hid1,
            para4=args.hid2,
            nclass=labels.max().item() + 1,
            dropout=args.dropout,
            smoo=args.smoo)



PATH = "state_dict_model.pt"
model.load_state_dict(torch.load(PATH))
if args.cuda:
    model = model.cuda()
    features = features.cuda()
    A_tilde = A_tilde.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
scheduler = StepLR(optimizer, step_size=50, gamma=0.9)

def test():
    model.eval()
    output = model(features,adj,A_tilde,adj_sct1,adj_sct2,adj_sct4,[args.order_1,args.sct_inx1],[args.order_2,args.sct_inx2])
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
# Testing
test()



