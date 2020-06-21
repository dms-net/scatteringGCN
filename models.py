import torch.nn as nn
import torch 
import torch.nn.functional as F
from layers import GC_withres,NGCN
class GCN(nn.Module):
    def __init__(self, nfeat, para3,para4, nclass, dropout,smoo):
        super(GCN, self).__init__()

        self.gc1 = NGCN(nfeat,med_f0=10,med_f1=10,med_f2=10,med_f3=para3,med_f4=para4)
#        self.gc1 = NGCN(nfeat,med_f0=28,med_f1=1,med_f2=1,med_f3=para3,med_f4=para4)
#        self.gc2 = NGCN(30+para3+para4,med_f0=28,med_f1=1,med_f2=1,med_f3=para3,med_f4=para4)
        self.gc11 = GC_withres(30+para3+para4, nclass,smooth=smoo)
        self.dropout = dropout

    def forward(self, x,adj, A_tilde,s1_sct,s2_sct,s3_sct,\
            sct_index1,\
            sct_index2):
        x = torch.FloatTensor.abs_(self.gc1(x,adj,A_tilde,\
                s1_sct,s2_sct,s3_sct,\
                adj_sct_o1 = sct_index1,\
                adj_sct_o2 = sct_index2))**4
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc11(x, adj)
        return F.log_softmax(x, dim=1)
