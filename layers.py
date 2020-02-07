import math
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix


import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from utils_sct import sparse_mx_to_torch_sparse_tensor
from utils_sct import normalize 
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
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
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

# class GC_sct(Module):
#     """
#     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#     """
#     def __init__(self, in_features, out_features, bias=True):
#         super(GC_sct, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.FloatTensor(in_features*3, out_features))#scattering matrix the input with (N,2*F)
#         if bias:
#             self.bias = Parameter(torch.FloatTensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
#     def forward(self, input, adj_sct_o1,adj_sct_o2):
#         # adj is extracted from the graph structure
#         #support = torch.mm(input, self.weight)
#         output1 = torch.spmm(adj_sct_o1.cuda(), input)
#         ## do no linear operator
#         input.data = abs(input.data)
#         output1.data = abs(output1.data)
#         output2 = torch.spmm(adj_sct_o2.cuda(),output1)
#         output2.data=abs(output2.data)
#         support = torch.cat((input, output1,output2), 1)
#         support = torch.mm(support, self.weight)
#         #output2 = torch.spmm(adj_sct_o2.cuda(),output1)
#         #output2.data=abs(output2.data)
#         #support.data = abs(support.data)
#         #output1 = output1.cuda()
#         #output2 = output2.cuda()
#         #support = support.cuda()
#         if self.bias is not None:
#             return support + self.bias
#         else:
#             return support
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_features) + ' -> ' \
#                + str(self.out_features) + ')'

# class GC_sct_res(Module):
#     """
#     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#     """
#     def __init__(self, in_features, out_features, bias=True):
#         super(GC_sct_res, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.FloatTensor(in_features*4, out_features))#scattering matrix the input with (N,2*F)
#         if bias:
#             self.bias = Parameter(torch.FloatTensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
#     def forward(self, input,adj,adj_sct_o1,adj_sct_o2):
#         # adj is extracted from the graph structure
#         #support = torch.mm(input, self.weight)
#         output1 = torch.spmm(adj_sct_o1.cuda(), input)
#         ## do no linear operator
#         #output0 = input
#         Lap_one_hop = torch.spmm(adj,input)
#         #output0.data = abs(input.data)
#         output0 = Lap_one_hop
#         output0.data = abs(Lap_one_hop.data)
#         output1.data = abs(output1.data)
#         output2 = torch.spmm(adj_sct_o2.cuda(),output1)
#         output2.data=abs(output2.data)
#         #support = torch.cat((input,output0, output1,output2), 1)
#         support = torch.cat((Lap_one_hop,output0, output1,output2), 1)
#         support = torch.mm(support, self.weight)
#         #output2 = torch.spmm(adj_sct_o2.cuda(),output1)
#         #output2.data=abs(output2.data)
#         #support.data = abs(support.data)
#         #output1 = output1.cuda()
#         #output2 = output2.cuda()
#         #support = support.cuda()
#         if self.bias is not None:
#             return support + self.bias
#         else:
#             return support
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_features) + ' -> ' \
#                + str(self.out_features) + ')'




# class GC_nGSN(Module):
#     """
#     N-GCN model, consider 3 Lap matrix
#     """
#     def __init__(self, in_features,med_features, out_features, bias=True):
#         super(GC_nGSN, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight0 = Parameter(torch.FloatTensor(in_features*1, med_features))
#         self.weight1 = Parameter(torch.FloatTensor(in_features*1, med_features))
#         self.weight2 = Parameter(torch.FloatTensor(in_features*1, med_features))
#         self.weight3 = Parameter(torch.FloatTensor(in_features*1, med_features))
#         self.weight4 = Parameter(torch.FloatTensor(in_features*1, med_features))
#         self.weight = Parameter(torch.FloatTensor(med_features*3, out_features))
# #pipeline 2
#         self.p2_weight = Parameter(torch.FloatTensor(in_features, out_features))


#         if bias:
#             self.bias1 = Parameter(torch.FloatTensor(med_features))
#             self.bias0 = Parameter(torch.FloatTensor(med_features))
#             self.bias2 = Parameter(torch.FloatTensor(med_features))
#             self.bias3 = Parameter(torch.FloatTensor(med_features))
#             self.bias4 = Parameter(torch.FloatTensor(med_features))

#             self.bias = Parameter(torch.FloatTensor(out_features))
# #pipeline 2
#             self.p2_bias = Parameter(torch.FloatTensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         stdv0 = 1. / math.sqrt(self.weight0.size(1))
#         stdv1 = 1. / math.sqrt(self.weight1.size(1))
#         stdv2 = 1. / math.sqrt(self.weight2.size(1))
#         stdv3 = 1. / math.sqrt(self.weight3.size(1))
#         stdv4 = 1. / math.sqrt(self.weight4.size(1))


# #pipeline 2
#         p2_stdv = 1. / math.sqrt(self.p2_weight.size(1))
        
#         self.weight0.data.uniform_(-stdv0, stdv0)
#         self.weight1.data.uniform_(-stdv1, stdv1)
#         self.weight2.data.uniform_(-stdv2, stdv2)
#         self.weight3.data.uniform_(-stdv3, stdv3)
#         self.weight4.data.uniform_(-stdv4, stdv4)

#         self.weight.data.uniform_(-stdv, stdv)

# #pipeline 2
#         self.p2_weight.data.uniform_(-p2_stdv,p2_stdv)

#         if self.bias is not None:
#             self.bias1.data.uniform_(-stdv1, stdv1)
#             self.bias0.data.uniform_(-stdv0, stdv0)
#             self.bias2.data.uniform_(-stdv2, stdv2)
#             self.bias3.data.uniform_(-stdv3, stdv3)
#             self.bias4.data.uniform_(-stdv4, stdv4)
#             self.bias.data.uniform_(-stdv, stdv)

# #pipeline 2
#             self.p2_bias.data.uniform_(-p2_stdv,p2_stdv)

#     def forward(self, input, adj,adj_sct_o1,adj_sct_o2):
#         # adj is extracted from the graph structure
#         #support = torch.mm(input, self.weight)
#         output0 = torch.spmm(adj, input)
#         support0 = torch.mm(output0, self.weight0)+self.bias0
#         output1 = torch.spmm(adj, output0)
#         support1 = torch.mm(output1, self.weight1)+self.bias1

#         output3 = torch.spmm(adj, output1)
#         support3 = torch.mm(output3, self.weight3)+self.bias3

#         abs_input = input
#         abs_input.data = abs(input.data)
#         output2 = torch.spmm(adj_sct_o1.cuda(),abs_input)
#         output2.data=abs(output2.data)
#         support2 = torch.mm(output2, self.weight2)+self.bias2

#         output4 = torch.spmm(adj_sct_o2.cuda(),abs_input)
#         output4.data=abs(output4.data)
#         support4 = torch.mm(output4, self.weight4)+self.bias4

#         #support_3hop = support0
#         #support_3hop = torch.cat((support0,support1,support2,support3,support4), 1)
#         support_3hop = torch.cat((support0,support2,support4), 1)
#         support_3hop = F.relu(support_3hop)
#         #support_3hop = F.dropout(support_3hop,0.5, training=self.training)

#         output_3hop = torch.mm(support_3hop, self.weight)


# #pipeline 2
#         p2_output = torch.spmm(adj, input)
#         p2_output = torch.mm(p2_output, self.p2_weight)

#         ## do no linear operator
#         #output2 = torch.spmm(adj_sct_o2.cuda(),output1)
#         #output2.data=abs(output2.data)
#         #support.data = abs(support.data)
#         #output1 = output1.cuda()
#         #output2 = output2.cuda()
#         #support = support.cuda()
#         if self.bias is not None:
#             return torch.cat((output_3hop+self.bias,p2_output+self.p2_bias),1)
#         else:
#             return torch.cat((output_3hop,p2_output),1)
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                 + str(self.in_features) + ' -> ' \
#                 + str(self.out_features) + ')'



# class GC_nGCN(Module):
#     """
#     N-GCN model, consider 3 Lap matrix
#     """
#     def __init__(self, in_features,med_features, out_features, bias=True):
#         super(GC_nGCN, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight0 = Parameter(torch.FloatTensor(in_features*1, med_features))
#         self.weight1 = Parameter(torch.FloatTensor(in_features*1, med_features))
#         self.weight2 = Parameter(torch.FloatTensor(in_features*1, med_features))
#         self.weight = Parameter(torch.FloatTensor(med_features*3, out_features))
#         if bias:
#             self.bias1 = Parameter(torch.FloatTensor(med_features))
#             self.bias0 = Parameter(torch.FloatTensor(med_features))
#             self.bias2 = Parameter(torch.FloatTensor(med_features))
#             self.bias = Parameter(torch.FloatTensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         stdv0 = 1. / math.sqrt(self.weight0.size(1))
#         stdv1 = 1. / math.sqrt(self.weight1.size(1))
#         stdv2 = 1. / math.sqrt(self.weight2.size(1))
#         self.weight0.data.uniform_(-stdv0, stdv0)
#         self.weight1.data.uniform_(-stdv1, stdv1)
#         self.weight2.data.uniform_(-stdv2, stdv2)
#         self.weight.data.uniform_(-stdv, stdv)

#         if self.bias is not None:
#             self.bias1.data.uniform_(-stdv1, stdv1)
#             self.bias0.data.uniform_(-stdv0, stdv0)
#             self.bias2.data.uniform_(-stdv2, stdv2)
#             self.bias.data.uniform_(-stdv, stdv)
#     def forward(self, input, adj):
#         # adj is extracted from the graph structure
#         #support = torch.mm(input, self.weight)
#         output0 = torch.spmm(adj, input)
#         support0 = torch.mm(output0, self.weight0)
#         output1 = torch.spmm(adj, output0)
#         support1 = torch.mm(output1, self.weight1)
#         output2 = torch.spmm(adj, output1)
#         support2 = torch.mm(output2, self.weight2)
#         support_3hop = torch.cat((support0, support1,support2), 1)
#         support_3hop = F.relu(support_3hop)
#         output_3hop = torch.mm(support_3hop, self.weight)
#         ## do no linear operator
#         #output2 = torch.spmm(adj_sct_o2.cuda(),output1)
#         #output2.data=abs(output2.data)
#         #support.data = abs(support.data)
#         #output1 = output1.cuda()
#         #output2 = output2.cuda()
#         #support = support.cuda()
#         if self.bias is not None:
#             return output_3hop + self.bias
#         else:
#             return output_3hop
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                 + str(self.in_features) + ' -> ' \
#                 + str(self.out_features) + ')'



class NGCN(Module):
    """
    N-GCN model, consider 3 Lap matrix
    """
    def __init__(self, in_features,med_f0,med_f1,med_f2,med_f3,med_f4,bias=True):
        super(NGCN, self).__init__()
        self.in_features = in_features
#        self.out_features = out_features
        self.med_f0 = med_f0
        self.med_f1 = med_f1
        self.med_f2 = med_f2
        self.med_f3 = med_f3
        self.med_f4 = med_f4
#        self.med_f5 = med_f5

        self.weight0 = Parameter(torch.FloatTensor(in_features, med_f0))
        self.weight1 = Parameter(torch.FloatTensor(in_features, med_f1))
        self.weight2 = Parameter(torch.FloatTensor(in_features, med_f2))
        self.weight3 = Parameter(torch.FloatTensor(in_features, med_f3))
        self.weight4 = Parameter(torch.FloatTensor(in_features, med_f4))
#        self.weight5 = Parameter(torch.FloatTensor(in_features, med_f5))


        #self.weight = Parameter(torch.FloatTensor((med_f0+med_f1+med_f2), out_features))

        if bias:
            self.bias1 = Parameter(torch.FloatTensor(med_f1))
            self.bias0 = Parameter(torch.FloatTensor(med_f0))
            self.bias2 = Parameter(torch.FloatTensor(med_f2))
            self.bias3 = Parameter(torch.FloatTensor(med_f3))
            self.bias4 = Parameter(torch.FloatTensor(med_f4))
#            self.bias5 = Parameter(torch.FloatTensor(med_f5))

#            self.bias = Parameter(torch.FloatTensor(med_f0+med_f1+med_f2+med_f3+med_f4))
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





