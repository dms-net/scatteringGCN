import os.path as osp
import scipy.sparse as sp
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import   CitationFull
from torch_geometric.utils import to_scipy_sparse_matrix
import torch_geometric.transforms as T
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'DBLP')
from torch_geometric.utils import to_scipy_sparse_matrix
from utils import normalize_adjacency_matrix,normalizemx
from DBLP_utils import SCAT_Red
from utils import normalize_adjacency_matrix,sparse_mx_to_torch_sparse_tensor
from layers import GC_withres
#from torch_geometric.nn import GATConv

#dataset = TUDataset(root= path,name='REDDIT-BINARY')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = CitationFull(path,name = 'dblp',transform=T.TargetIndegree())
data = dataset[0]
# Num of feat:1639
adj = to_scipy_sparse_matrix(edge_index = data.edge_index)
adj = adj+ adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
A_tilde = sparse_mx_to_torch_sparse_tensor(normalize_adjacency_matrix(adj,sp.eye(adj.shape[0]))).to(device)
adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
#print(dataset)
#print(data.x.shape)
#print(data.y.shape)


tp = SCAT_Red(in_features=1639,med_f0=10,med_f1=10,med_f2=10,med_f3=10,med_f4=10).to(device)
tp2 = SCAT_Red(in_features=40,med_f0=30,med_f1=10,med_f2=10,med_f3=10,med_f4=10).to(device)

#out_tp = tp(data.x.to(device),A_tilde= A_tilde.to(device),adj = adj.to(device))

#out_tp = tp2(out_tp,A_tilde= A_tilde.to(device),adj = adj.to(device))

#out_tp = GC_withres(60,1,smooth=0.1).to(device)(out_tp,adj = A_tilde.to(device))
#print(out_tp.shape) # should be torch.Size([17716, 40])
#adj = to_scipy_sparse_matrix(edge_index = data.edge_index)
#train_mask = torch.cat((torch.ones(10000),torch.zeros(2000),torch.zeros(2000),torch.zeros(3716)),0)>0 
train_mask = torch.cat((torch.ones(10000),torch.zeros(2000),torch.zeros(2000),torch.zeros(3716)),0)>0
val_mask = torch.cat((torch.zeros(10000),torch.ones(2000),torch.zeros(2000),torch.zeros(3716)),0)>0
test_mask = torch.cat((torch.zeros(10000),torch.zeros(2000),torch.ones(2000),torch.zeros(3716)),0)>0




class Net(torch.nn.Module):
    def __init__(self,dropout=0.6):
        super(Net, self).__init__()
        self.sct1 = SCAT_Red(in_features=1639,med_f0=40,med_f1=20,med_f2=20,med_f3=20,med_f4=20)
        self.sct2 = SCAT_Red(in_features=120,med_f0=40,med_f1=20,med_f2=20,med_f3=20,med_f4=20)
        self.res1 = GC_withres(120,4,smooth=0.1)
        self.dropout = dropout
    def forward(self):
        x = torch.FloatTensor.abs_(self.sct1(data.x,A_tilde= A_tilde,adj = adj))**1
        x = torch.FloatTensor.abs_(self.sct2(x,A_tilde= A_tilde,adj = adj))**1
#        x = self.sct1(data.x,A_tilde= A_tilde,adj = adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.res1(x, A_tilde)
        return F.log_softmax(x, dim=1)

model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)



def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[train_mask], data.y[train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(), []
    for mask in [train_mask, val_mask,test_mask]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


for epoch in range(1, 101):
    train()
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, *test()))




