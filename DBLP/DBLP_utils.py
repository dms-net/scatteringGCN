import torch
import torch.sparse
import math
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
from utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def normalize_adjacency_tensor_matrix(A, I_n):
#     # A is a torch sparse tensor
#     A_tilde = torch.sparse.FloatTensor.add(A,I_n)
#     degrees = torch.sparse.sum(A_tilde,dim=0).to_dense().pow(-0.5)
#     D_inv = torch.sparse_coo_tensor(I_n._indices(),degrees,size=I_n.size())
#     A_tilde_hat = torch.sparse.mm(A_tilde,D_inv.to_dense())
#     A_tilde_hat = torch.sparse.mm(D_inv.to_sparse(),A_tilde_hat)
#     return A_tilde_hat


def normalizem_tentor_mx(mx,I_n):
    # mx is a torch sparse tensor
    degrees = torch.sparse.sum(mx,dim=0).to_dense().pow(-1)
    D_inv = torch.sparse_coo_tensor(I_n._indices(),degrees,size=I_n.size())
    # torch.sparse_coo_tensor(t._indices(), t._values(), size = (n,n))
    # turn degree into a dense tensor
    mx = torch.sparse.mm(mx,D_inv.to_dense())
    # return a dense tensor
    return mx

def red_gene_sct(sparse_tensor,dense_tensor,order):
    for i in range(0,order):
        dense_tensor = torch.sparse.mm(sparse_tensor,dense_tensor)
    return dense_tensor
class SCAT_Red(nn.Module):
    def __init__(self,in_features,med_f0,med_f1,med_f2,med_f3,med_f4,bias=True):
        super(SCAT_Red, self).__init__()
        # self.features = features
        self.in_features = in_features
        # self.adjacency_mx = adjacency_mx
        self.med_f0 = med_f0
        self.med_f1 = med_f1
        self.med_f2 = med_f2
        self.bias = bias
        self.med_f3 = med_f3
        self.med_f4 = med_f4
        # features shape (N_of_nodes,N_of_feature)
        # adjacency_mx shape(N_of_nodes,N_of_nodes)
        # in_features is N_of_feature
        self.weight0 = Parameter(torch.FloatTensor(in_features, med_f0))
        self.weight1 = Parameter(torch.FloatTensor(in_features, med_f1))
        self.weight2 = Parameter(torch.FloatTensor(in_features, med_f2))
        self.weight3 = Parameter(torch.FloatTensor(in_features, med_f3))
        self.weight4 = Parameter(torch.FloatTensor(in_features, med_f4))
        if bias:
            print('Processing first three')
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
        torch.nn.init.xavier_uniform_(self.weight0)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight3)
        torch.nn.init.xavier_uniform_(self.weight4)
        if self.bias is not None:
            self.bias1.data.uniform_(-stdv1, stdv1)
            self.bias0.data.uniform_(-stdv0, stdv0)
            self.bias2.data.uniform_(-stdv2, stdv2)
            self.bias3.data.uniform_(-stdv3, stdv3)
            self.bias4.data.uniform_(-stdv4, stdv4)
    def forward(self,features,A_tilde,adj,order1 = 1,order2 = 2):
        # adj is extracted from the graph structure
        # features: torch tensor
        # adjacency_mx: sparse tensor
        # adj = adjacency_mx
        # adj = adj+ adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # A_tilde = normalize_adjacency_matrix(adj,sp.eye(adj.shape[0]))
        # adj = normalizemx(adj)
        I_n = sp.eye(adj.size()[0]) # requires GPU here
        I_n = sparse_mx_to_torch_sparse_tensor(I_n).to(device)
        input = features
        # A_tilde = normalize_adjacency_tensor_matrix(adj,I_n)
        # A_tilde = A_tilde
        # A_tilde = sparse_mx_to_torch_sparse_tensor(A_tilde)
        # do A \cdot Feature_Matrix
        # torch.sparse.mm(mat1, mat2) sparse matrix mat1 and dense matrix mat2
#        print('Processing first three')
        support0 = torch.mm(torch.sparse.mm(A_tilde,input),self.weight0)  + self.bias0
        support1 = torch.mm(torch.sparse.mm(A_tilde,torch.sparse.mm(A_tilde,input)),self.weight1)  + self.bias1
        support2 = torch.mm(torch.sparse.mm(A_tilde,torch.sparse.mm(A_tilde,torch.sparse.mm(A_tilde,input))),self.weight2)  + self.bias2


        # torch.sparse.FloatTensor.mul_(A,B)
        # A,B sparse tensor
        # A and B has to has the same size (n,n), perform A\cdotB
        # return a sparse tensor
        # torch.sparse.FloatTensor.matmul(mx,mx.to_dense())
        # support0 = torch.sparse.FloatTensor.matmul(torch.sparse.mm(A_tilde,input),self.weight0)  + self.bias0
        # support1 = torch.sparse.FloatTensor.matmul(torch.sparse.FloatTensor.matmul(A_tilde,torch.sparse.FloatTensor.matmul(A_tilde,input)).to_sparse(),self.weight1)  + self.bias1
        # # support1 = torch.sparse.mm(torch.sparse.FloatTensor.mul_(A_tilde,torch.sparse.mm(A_tilde,input)),self.weight1)  + self.bias1
        # support2 = torch.sparse.mm(torch.sparse.FloatTensor.mul_(A_tilde,torch.sparse.FloatTensor.mul_(A_tilde,torch.sparse.mm(A_tilde,input))),self.weight2) + self.bias2
        #
        #



        # scattering 1
        # generata first scatter feature layer
        # input adj: a torch tensor
        adj = normalizem_tentor_mx(adj,I_n) #A \cdot D^(-1)
        adj_power = 0.5 * torch.sparse.FloatTensor.add(adj.to_sparse(),I_n) # the  P, transfer adj to adj.to_sparse() saves a lots of time
        support3 = torch.mm(red_gene_sct(adj_power,input,order1),self.weight3)-\
        torch.mm(red_gene_sct(adj_power,input,2*order1),self.weight3)
        support3 = support3 + self.bias3


        support4 = torch.mm(red_gene_sct(adj_power,input,order2),self.weight4)-\
        torch.mm(red_gene_sct(adj_power,input,2*order2),self.weight4)
        support4 = support4 + self.bias4

        # support2 = torch.sparse.mm(torch.sparse.mm(A_tilde,torch.sparse.mm(A_tilde,torch.sparse.mm(A_tilde,input))),self.weight2)  + self.bias2
        support_3hop = torch.cat((support0,support1,support2,support3,support4), 1)

        output_3hop = support_3hop
        return output_3hop
