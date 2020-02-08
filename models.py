import torch.nn as nn
import torch 
import torch.nn.functional as F
from layers import GC_withres,NGCN
class GCN(nn.Module):
    def __init__(self, nfeat, para3,para4, nclass, dropout,smoo):
        super(GCN, self).__init__()

        self.gc1 = NGCN(nfeat,med_f0=15,med_f1=15,med_f2=15,med_f3=para3,med_f4=para4)
        self.gc11 = GC_withres(45+para3+para4, nclass,smooth=smoo)
        self.dropout = dropout

    def forward(self, x,adj, A_tilde,adj_sct1,adj_sct2,adj_sct4,adj_sct8,adj_sct16,sct_index1,sct_index2):
        scat_dict = [adj_sct1,adj_sct2,adj_sct4,adj_sct8,adj_sct16]
        x = F.relu(torch.FloatTensor.abs_(self.gc1(x,adj,A_tilde,scat_dict[sct_index1],scat_dict[sct_index2]))**4)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc11(x, adj)
        return F.log_softmax(x, dim=1)
