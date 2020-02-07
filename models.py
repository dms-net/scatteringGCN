import torch.nn as nn
import torch 
import torch.nn.functional as F
from layers import  GC
from layers import GC_sct,GC_sct_res,GC_nGCN,GC_nGSN,GC_withres,NGCN
class GCN(nn.Module):
    def __init__(self, nfeat, para3,para4, nclass, dropout,smoo):
        super(GCN, self).__init__()

        self.gc1 = NGCN(nfeat,med_f0=15,med_f1=15,med_f2=15,med_f3=para3,med_f4=para4)
#        self.gc2 = NGCN(30+para3+para4,med_f0=28,med_f1=1,med_f2=1,med_f3=para3,med_f4=para4)
#        self.gc3 = NGCN(20+para3+para4,med_f0=18,med_f1=1,med_f2=1,med_f3=para3,med_f4=para4)

#        self.gc2 = NGCN(30+para3+para4,med_f0=10,med_f1=10,med_f2=10,med_f3=para3,med_f4=para4)
#        self.gc3 = NGCN(20+para3+para4,med_f0=18,med_f1=1,med_f2=1,med_f3=para3,med_f4=para4)
#        self.gc3 = NGCN(60+para3+para4,med_f0=58,med_f1=1,med_f2=1,med_f3=para3,med_f4=para4)
#        self.gc3 = NGCN(34+para3+para4,med_f0=32,med_f1=1,med_f2=1,med_f3=para3,med_f4=para4)
#        self.gc4 = NGCN(34+para3+para4,med_f0=32,med_f1=1,med_f2=1,med_f3=para3,med_f4=para4)
#        self.gc3 = NGCN(34+para3+para4,med_f0=32,med_f1=1,med_f2=1,med_f3=para3,med_f4=para4)    
#        self.gc4 = NGCN(34+para3+para4,med_f0=32,med_f1=1,med_f2=1,med_f3=para3,med_f4=para4)
#        self.gc5 = NGCN(20+para3+para4,med_f0=10,med_f1=5,med_f2=5,med_f3=para3,med_f4=para4)
#        self.gc6 = NGCN(20+para3+para4,med_f0=10,med_f1=5,med_f2=5,med_f3=para3,med_f4=para4)
#        self.gc7 = NGCN(20+para3+para4,med_f0=10,med_f1=5,med_f2=5,med_f3=para3,med_f4=para4)
#        self.gc8 = NGCN(20+para3+para4,med_f0=10,med_f1=5,med_f2=5,med_f3=para3,med_f4=para4)
        #self.gc4 = GC(nfeat, 32)
        #self.gc6 = GC_sct_res(nhid1,nhid1)
        #self.gc7 = GC(4*nhid1, nhid1)
        #self.gc8 = GC(nhid1, nhid1)
        #self.gc9 = GC_sct_res(nhid1, nhid1)
        #self.gc10 = GC(32,nclass)
#        self.gc11 = GC(30+para3+para4, nclass)
        self.gc11 = GC_withres(45+para3+para4, nclass,smooth=smoo)
        self.dropout = dropout

    def forward(self, x,adj, A_tilde,adj_sct1,adj_sct2,adj_sct4,adj_sct8,adj_sct16,sct_index1,sct_index2):
        scat_dict = [adj_sct1,adj_sct2,adj_sct4,adj_sct8,adj_sct16]
        ### x = F.relu(self.gc4(x, adj_sct3,adj_sct7))
        x = F.relu(torch.FloatTensor.abs_(self.gc1(x,adj,A_tilde,scat_dict[sct_index1],scat_dict[sct_index2]))**4)
#        x = F.relu(self.gc1(x,adj,A_tilde,scat_dict[sct_index1],scat_dict[sct_index2]))
#        x = F.leaky_relu(self.gc2(x,adj,A_tilde,scat_dict[sct_index1],scat_dict[sct_index2]))
#        x = F.relu(self.gc3(x,adj,A_tilde,scat_dict[sct_index1],scat_dict[sct_index2]))
#        x = F.relu(self.gc2(x,adj,A_tilde,scat_dict[sct_index1],scat_dict[sct_index2]))
#        x = F.leaky_relu(self.gc3(x,adj,A_tilde,scat_dict[sct_index1],scat_dict[sct_index2]))
#        x = F.leaky_relu(self.gc4(x,adj,A_tilde,scat_dict[sct_index1],scat_dict[sct_index2]))
        #x = F.relu(self.gc5(x, adj))
        #x = F.relu(self.gc6(x, adj_sct1,adj_sct16))
        #x = F.relu(self.gc7(x, adj))
        #x = F.relu(self.gc8(x, adj))
        #x = F.relu(self.gc9(x, adj_sct2,adj_sct8))
        #x = F.relu(self.gc10(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc11(x, adj)
        return F.log_softmax(x, dim=1)
