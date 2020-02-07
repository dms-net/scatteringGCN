import numpy as np
import scipy.sparse as sp
import torch

def scattering1st(spmx,order):
    I_n = sp.eye(spmx.shape[0])
    adj_sct = 0.5*(I_n + spmx)
    adj_power = adj_sct
    for i in range(order-1):
        adj_power = adj_power.dot(adj_sct)
    adj_int = (I_n-adj_power).dot(adj_power)
#    adj_int.data = abs(adj_int.data)
    #adj_power = adj_power.dot(adj_int)
    return adj_int

def scattering2nd(spmx,order1,order2):
    I_n = sp.eye(spmx.shape[0])
    adj_sct = 0.5*(I_n + spmx)
    adj_power1 = adj_sct
    for i in range(order1-1):
        adj_power1 = adj_power1.dot(adj_sct)
    adj_int = (I_n-adj_power1).dot(adj_power1)
#    adj_int.data = abs(adj_int.data)
    adj_power2 = adj_sct
    for i in range(order2-1):
        adj_power2 = adj_power2.dot(adj_sct)
    adj_power = (adj_power2.dot(I_n-adj_power2)).adj_int
#    adj_power.data = abs(adj_power.data)
    return adj_power
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data_sct(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    ### graph scatterig
    adj_sct2 = scattering1st(sparse_mx_to_torch_sparse_tensor(adj),2)
    adj_sct4 = scattering1st(sparse_mx_to_torch_sparse_tensor(adj),4)
    adj_sct8 = scattering1st(sparse_mx_to_torch_sparse_tensor(adj),8)
    adj_sct16 = scattering1st(sparse_mx_to_torch_sparse_tensor(adj),16)
    adj_sct1 = scattering1st(sparse_mx_to_torch_sparse_tensor(adj),1)

    adj_sct1 = sparse_mx_to_torch_sparse_tensor(adj_sct1)
    adj_sct2 = sparse_mx_to_torch_sparse_tensor(adj_sct2)
    adj_sct4 = sparse_mx_to_torch_sparse_tensor(adj_sct4)
    adj_sct8 = sparse_mx_to_torch_sparse_tensor(adj_sct8)
    adj_sct16 = sparse_mx_to_torch_sparse_tensor(adj_sct16)



    adj = sparse_mx_to_torch_sparse_tensor(adj)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj,adj_sct1,adj_sct2,adj_sct4,adj_sct8,adj_sct16, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
