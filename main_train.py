from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn import preprocessing
import numpy as np
import argparse
import torch
from random import sample
import random
import math
import time
import model as model_zero
import model_few
from model_few import tokenize

from data import DataHelper
from sklearn import preprocessing
from aug import *


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    setup_seed(seed)

    if args.pretrain_model == "few":
        model = model_few.CLIP(args)
    elif args.pretrain_model == "zero":
        model = model_zero.CLIP(args)

    model = model.to(device)
    # print(model)

    Data = DataHelper(arr_edge_index, args)
    model.train()

    for j in range(1, args.epoch_num + 1):
        loader = DataLoader(Data, batch_size=args.batch_size, shuffle=True, num_workers=10)
        loss_list = []
        pbar = tqdm(loader)
        pbar.set_description("Epoch {}".format(j))
        for batch in pbar:
            s_n, t_n = batch['s_n'], batch['t_n']
            s_n_arr, t_n_arr = s_n.numpy(), t_n.numpy().reshape(-1)  # .reshape((1, -1))
            s_n_text, t_n_text = np.array(tit_list)[s_n_arr].tolist(), np.array(tit_list)[t_n_arr].tolist()
            s_n_text, t_n_text = tokenize(s_n_text, context_length=args.context_length).to(device), tokenize(t_n_text, context_length=args.context_length).to(device)
            loss = model.forward(node_f.to(device), edge_index.to(device), aug_edge_index.to(device), s_n, t_n, s_n_text, t_n_text, device)
            loss_list.append(loss)
            pbar.set_postfix(loss=loss)

        pbar.set_postfix(loss=round(sum(loss_list)/len(loss_list), 4))
    path = './res/{}/{}_node_ttgt_8&12_0.1_epoch2.pkl'.format(args.data_name, args.data_name)
    torch.save(model.state_dict(), path)
    print('Saved model checkpoint in ' + path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--aggregation_times', type=int, default=2, help='Aggregation times')
    parser.add_argument('--epoch_num', type=int, default=2, help='epoch number')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--edge_coef', type=float, default=10)
    parser.add_argument('--neigh_num', type=int, default=3)
    parser.add_argument('--coop_n_ctx', type=int, default=4)
    parser.add_argument('--pretrain_model', type=str, default='few',
                        help='pretrain model for few-/zero-shot, selected in [zero, few]')

    parser.add_argument('--gnn', type=str, default='GFormer', help='GCN, GFormer')
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--gnn_layers', type=int, default=2)
    parser.add_argument('--trans_layers', type=int, default=1)
    parser.add_argument('--gnn_input', type=int, default=128)
    parser.add_argument('--gnn_hid', type=int, default=128)
    parser.add_argument('--gnn_output', type=int, default=128)
    parser.add_argument('--gnn_dropout', type=float, default=0.5)
    parser.add_argument('--trans_dropout', type=float, default=0.2)
    parser.add_argument('--graph_weight', type=float, default=0.8)
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='weight for residual link')
    parser.add_argument('--use_bn', action='store_true', help='use layernorm')
    parser.add_argument('--use_residual', action='store_true',
                        help='use residual link for each GNN layer')
    parser.add_argument('--use_weight', action='store_true',
                        help='use weight for GNN convolution')
    parser.add_argument('--use_init', action='store_true', help='use initial feat for each GNN layer')
    parser.add_argument('--use_act', action='store_true', help='use activation for each GNN layer')
    parser.add_argument('--aggregate', type=str, default='add',
                        help='aggregate type, add or cat.')
    parser.add_argument('--loss', type=str, default='con_sum_aug',
                        help='con: contrastive loss, sum: summary loss, mar: margin loss,'
                             'tm: text matching loss, np: node perturbation loss,'
                             'mar loss is only used for zero-shot pre-training')
    parser.add_argument('--nn_weight', type=float, default=0.5, help='loss weight for tm loss')
    parser.add_argument('--aug_weight', type=float, default=0.5, help='loss weight for np loss')
    parser.add_argument('--topK', type=int, default=1, help='top-k similar texts')
    parser.add_argument('--aug_ratio', type=float, default=0.1, help='ratio for node perturbation')


    parser.add_argument('--context_length', type=int, default=128)

    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--transformer_heads', type=int, default=8)
    parser.add_argument('--transformer_layers', type=int, default=12)
    parser.add_argument('--transformer_width', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=49408)  # 49408
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_name', type=str, default='cora')

    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print('device:', device)

    num_nodes = 0
    tit_list = []
    with open('data/cora/train_text.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            tit_list.append(line[2])
            num_nodes += 1

    print('num_nodes', num_nodes)

    raw_edge_index = [[], []]
    with open('data/cora/mapped_edges.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            raw_edge_index[0].append(int(line[0]))
            raw_edge_index[1].append(int(line[1]))

    print('num of edges', len(raw_edge_index[0] + raw_edge_index[1]))

    edge_index = [raw_edge_index[0] + raw_edge_index[1], raw_edge_index[1] + raw_edge_index[0]]
    arr_edge_index = np.array(edge_index)
    edge_index = np.array(edge_index)
    aug_edge_index = permute_edges(num_nodes, edge_index, ratio=args.aug_ratio)
    
    edge_index = torch.from_numpy(edge_index)
    aug_edge_index = torch.from_numpy(aug_edge_index)

    node_f = np.load('data/cora/node_f.npy')
    node_f = preprocessing.StandardScaler().fit_transform(node_f)

    node_f = torch.from_numpy(node_f)

    start = time.perf_counter()
    seed = 1
    print(vars(args))
    main(args)

    end = time.perf_counter()
    print('aug_ratio:', args.aug_ratio)
    print("time consuming {:.2f}".format(end - start))
