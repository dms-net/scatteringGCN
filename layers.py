import math
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix


import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from utils import sparse_mx_to_torch_sparse_tensor
from utils import normalize 
class GC(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, input, adj):
        # adj is extracted from the graph structure
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_withres(Module):
    """
    res conv
    """
    def __init__(self, in_features, out_features,smooth,bias=True):
        super(GC_withres, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.smooth = smooth
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, input, adj):
        # adj is extracted from the graph structure
        support = torch.mm(input, self.weight)
        I_n = sp.eye(adj.shape[0])
        I_n = sparse_mx_to_torch_sparse_tensor(I_n).cuda()
        output = torch.spmm((I_n+self.smooth*adj)/(1+self.smooth), support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'






class NGCN(Module):
    """
    Bandpass model, consider 3 Lap matrix
    """
    def __init__(self, in_features,med_f0,med_f1,med_f2,med_f3,med_f4,bias=True):
        super(NGCN, self).__init__()
        self.in_features = in_features
        self.med_f0 = med_f0
        self.med_f1 = med_f1
        self.med_f2 = med_f2
        self.med_f3 = med_f3
        self.med_f4 = med_f4

        self.weight0 = Parameter(torch.FloatTensor(in_features, med_f0))
        self.weight1 = Parameter(torch.FloatTensor(in_features, med_f1))
        self.weight2 = Parameter(torch.FloatTensor(in_features, med_f2))
        self.weight3 = Parameter(torch.FloatTensor(in_features, med_f3))
        self.weight4 = Parameter(torch.FloatTensor(in_features, med_f4))


        #self.weight = Parameter(torch.FloatTensor((med_f0+med_f1+med_f2), out_features))

        if bias:
            self.bias1 = Parameter(torch.FloatTensor(med_f1))
            self.bias0 = Parameter(torch.FloatTensor(med_f0))
            self.bias2 = Parameter(torch.FloatTensor(med_f2))
            self.bias3 = Parameter(torch.FloatTensor(med_f3))
            self.bias4 = Parameter(torch.FloatTensor(med_f4))

        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        stdv0 = 1. / math.sqrt(self.weight0.size(1))
        stdv1 = 1. / math.sqrt(self.weight1.size(1))
        stdv2 = 1. / math.sqrt(self.weight2.size(1))

        stdv3 = 1. / math.sqrt(self.weight3.size(1))
        stdv4 = 1. / math.sqrt(self.weight4.size(1))
        torch.nn.init.xavier_uniform(self.weight0)
        torch.nn.init.xavier_uniform(self.weight2)
        torch.nn.init.xavier_uniform(self.weight1)
        torch.nn.init.xavier_uniform(self.weight3)
        torch.nn.init.xavier_uniform(self.weight4)
        if self.bias0 is not None:
            self.bias1.data.uniform_(-stdv1, stdv1)
            self.bias0.data.uniform_(-stdv0, stdv0)
            self.bias2.data.uniform_(-stdv2, stdv2)

            self.bias3.data.uniform_(-stdv3, stdv3)
            self.bias4.data.uniform_(-stdv4, stdv4)

    def forward(self, input, adj,A_tilde,adj_sct_o1,adj_sct_o2):
        # adj is extracted from the graph structure
        support0 = torch.mm(input, self.weight0)
        output0 = torch.spmm(A_tilde, support0) + self.bias0
        support1 = torch.mm(input, self.weight1)
        output1 = torch.spmm(A_tilde, support1)
        output1 = torch.spmm(A_tilde, output1)+ self.bias1

        support2 = torch.mm(input, self.weight2)
        output2 = torch.spmm(A_tilde, support2)
        output2 = torch.spmm(A_tilde, output2)
        output2 = torch.spmm(A_tilde, output2)+ self.bias2


        support3 = torch.mm(input, self.weight3) 
        output3 = torch.spmm(adj_sct_o1.cuda(), support3)+ self.bias3

        support4 = torch.mm(input, self.weight4)
        output4 = torch.spmm(adj_sct_o2.cuda(), support4)+ self.bias4


        support_3hop = torch.cat((output0,output1,output2,output3,output4), 1)
        output_3hop = support_3hop
        if self.bias0 is not None:
            return output_3hop  
            #return output_3hop
        else:
            return output_3hop
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
                + str(self.in_features) + ' -> ' \
                + str(self.out_features) + ')'





