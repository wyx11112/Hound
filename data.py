from __future__ import division
from torch.utils.data import Dataset
import numpy as np
import torch
import random
import scipy.sparse as sp
import networkx as nx
from gformer import QuickGCN


def laplacian_positional_encoding(edge_index, pos_enc_dim, num_nodes):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian

    #adjacency_matrix(transpose, scipy_fmt="csr")
    edge_index = edge_index.numpy()
    edge_index_t = edge_index.T
    DG = nx.DiGraph()
    DG.add_edges_from(edge_index_t)
    degree_list = []
    for node, degree in (DG.in_degree()):
        degree_list.append(degree)
    degree = np.array(degree_list)
    del DG
    A = sp.csr_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                      shape=(num_nodes, num_nodes), dtype=np.float32)
    N = sp.diags(degree.clip(1) ** -0.5, dtype=float)
    L = sp.eye(num_nodes) - N * A * N

    # Eigenvectors with scipy
    #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    lap_pos_enc = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float()

    return lap_pos_enc


def re_features(adj, features, K, device):
    quickgcn = QuickGCN().to(device)

    #传播之后的特征矩阵,size= (N, 1, K+1, d )
    nodes_features = torch.empty(features.shape[0], 1, K+1, features.shape[1])

    for i in range(features.shape[0]):

        nodes_features[i, 0, 0, :] = features[i]

    x = features + torch.zeros_like(features)

    for i in range(K):

        x = quickgcn(adj.to(device), x.to(device))

        for index in range(features.shape[0]):

            nodes_features[index, 0, i + 1, :] = x[index]

    nodes_features = nodes_features.squeeze()
    return nodes_features


class DataHelper(Dataset):
    def __init__(self, edge_index, args, directed=False, transform=None):
        # self.num_nodes = len(node_list)
        self.transform = transform
        self.degrees = dict()
        self.node_set = set()
        self.neighs = dict()
        self.args = args

        idx, degree = np.unique(edge_index, return_counts=True)
        for i in range(idx.shape[0]):
            self.degrees[idx[i]] = degree[i].item()

        self.node_dim = idx.shape[0]
        print('lenth of dataset', self.node_dim)

        train_edge_index = edge_index
        self.final_edge_index = train_edge_index.T

        for i in range(self.final_edge_index.shape[0]):
            s_node = self.final_edge_index[i][0].item()
            t_node = self.final_edge_index[i][1].item()

            if s_node not in self.neighs:
                self.neighs[s_node] = []
            if t_node not in self.neighs:
                self.neighs[t_node] = []

            self.neighs[s_node].append(t_node)
            if not directed:
                self.neighs[t_node].append(s_node)

        # self.neighs = sorted(self.neighs)
        self.idx = idx

    def __len__(self):
        return self.node_dim

    def __getitem__(self, idx):
        s_n = self.idx[idx].item()
        t_n = [np.random.choice(self.neighs[s_n], replace=True).item() for _ in range(self.args.neigh_num)]
        t_n = np.array(t_n)

        sample = {
            's_n': s_n,  # e.g., 5424
            't_n': t_n,  # e.g., 5427
            # 'neg_n': neg_n
        }

        if self.transform:
            sample = self.transform(sample)

        return sample





