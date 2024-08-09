import torch
import numpy as np


def drop_nodes(num_nodes, edge_index):
    drop_num = int(num_nodes / 10)
    idx_drop = np.random.choice(num_nodes, drop_num, replace=False)
    # idx_nodrop = [n for n in range(num_nodes) if n not in idx_drop]

    adj = np.zeros((num_nodes, num_nodes))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = np.nonzero(adj)
    edge_index = np.concatenate(([edge_index[0]], [edge_index[1]]), axis=0)

    return edge_index


def permute_edges(num_nodes, edge_index, ratio=0.1):
    edge_num = edge_index.shape[1]
    permute_num = int(edge_num * 0.1)
    #edge_index = edge_index.transpose()
    edge_index = edge_index.T
    edge_index = edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)]
    #edge_index = edge_index.transpose()
    edge_index = edge_index.T
    return edge_index


def mask_nodes(num_nodes, feature):
    feat_dim = feature.shape[-1]
    mask_num = int(num_nodes / 10)

    idx_mask = np.random.choice(num_nodes, mask_num, replace=False)
    feature[idx_mask] = np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim))
   
    return feature

