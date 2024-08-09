import numpy as np
import argparse
import torch
from random import sample
import random
import math
import time
from model_few import CLIP, tokenize
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score
from multitask_amazon import multitask_data_generator
from model_g_coop import CoOp
import json
from data_graph import DataHelper
from torch.utils.data import DataLoader


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    task_list, train_idx, val_idx, test_idx = multitask_data_generator(lab_list, labeled_ids, labels, args.k_spt,
                                                                       args.k_val, args.k_qry, args.n_way)
    acc_list = []
    f1_list = []
    for j in range(len(task_list)):

        train_idx_ts = torch.from_numpy(np.array(train_idx[j])).to(device)
        val_idx_ts = torch.from_numpy(np.array(val_idx[j])).to(device)
        test_idx_ts = torch.from_numpy(np.array(test_idx[j])).to(device)

        train_truth = []
        for a in train_idx[j]:
            train_truth.append(id_lab_dict[str(a)])

        val_truth = []
        for a in val_idx[j]:
            val_truth.append(id_lab_dict[str(a)])

        test_truth = []
        for a in test_idx[j]:
            test_truth.append(id_lab_dict[str(a)])

        task_lables_arr = np.array(labels)[task_list[j]]
        task_labels_dict = dict()
        for i in range(task_lables_arr.shape[0]):
            task_labels_dict[task_lables_arr[i]] = i

        train_truth_ts = [task_labels_dict[train_truth[i]] for i in range(len(train_truth))]
        train_truth_ts = torch.from_numpy(np.array(train_truth_ts)).to(device)

        val_truth_ts = [task_labels_dict[val_truth[i]] for i in range(len(val_truth))]
        val_truth_ts = torch.from_numpy(np.array(val_truth_ts)).to(device)

        test_truth_ts = [task_labels_dict[test_truth[i]] for i in range(len(test_truth))]
        test_truth_ts = torch.from_numpy(np.array(test_truth_ts)).to(device)

        task_lables = task_lables_arr.tolist()
        Data = DataHelper(arr_edge_index, args, train_idx[j])
        loader = DataLoader(Data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        for i_batch, sample_batched in enumerate(loader):
            s_n = sample_batched['s_n'].numpy()
            t_n = sample_batched['t_n'].numpy()
        s_n = s_n.reshape(args.num_labels, args.k_spt)
        t_n = t_n.reshape(args.num_labels, args.k_spt * args.neigh_num)
        temp = []
        for i in range(args.num_labels):
            temp.append(np.concatenate((s_n[i], t_n[i])))
        g_texts = []
        for i in range(len(temp)):
            g_text = [new_dict[a] for a in temp[i]]
            g_texts.append(g_text)

        model = CoOp(args, task_lables, clip_model, g_texts, device)

        best_val = 0
        patience = 10
        counter = 0

        for epoch in range(1, args.ft_epoch + 1):
            # print('----epoch:' + str(epoch))
            model.train()
            model.forward(train_idx_ts, node_f, edge_index, train_truth_ts)

            model.eval()
            with torch.no_grad():
                res = model.forward(val_idx_ts, node_f, edge_index, val_truth_ts, training=False)
                val_acc = accuracy_score(val_truth_ts.cpu(), res.argmax(dim=1).cpu())
                if val_acc <= best_val:
                    counter += 1
                    if counter >= patience:
                        break
                else:
                    best_val = val_acc
                    torch.save(model, './res/{}/few_g_coop.pkl'.format(args.data_name))
                    counter = 0
        # print('{}th_task_best_val'.format(j), round(best_val, 4))

        best_model = torch.load('./res/{}/few_g_coop.pkl'.format(args.data_name))
        best_model.eval()
        with torch.no_grad():
            res = model.forward(test_idx_ts, node_f, edge_index, test_truth_ts, training=False)
            test_acc = accuracy_score(test_truth_ts.cpu(), res.argmax(dim=1).cpu())
            acc_list.append(test_acc)
            f1 = f1_score(test_truth_ts.cpu(), res.argmax(dim=1).cpu(), average='macro')
            f1_list.append(f1)

    _acc_mean = np.mean(acc_list)
    _f1_mean = np.mean(f1_list)
    all_acc_list.append(_acc_mean)
    all_f1_list.append(_f1_mean)
    print('acc mean: {}'.format(round(_acc_mean, 4)))
    print('f1 mean: {}'.format(round(_f1_mean, 4)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--aggregation_times', type=int, default=2, help='Aggregation times')
    parser.add_argument('--ft_epoch', type=int, default=50, help='fine-tune epoch')
    parser.add_argument('--lr', type=float, default=0.01)

    parser.add_argument('--gnn', type=str, default='GFormer', help='GCN, GFormer')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--gnn_layers', type=int, default=2)
    parser.add_argument('--trans_layers', type=int, default=1)
    parser.add_argument('--gnn_input', type=int, default=128)
    parser.add_argument('--gnn_hid', type=int, default=128)
    parser.add_argument('--gnn_output', type=int, default=128)
    parser.add_argument('--gnn_dropout', type=float, default=0.5)
    parser.add_argument('--trans_dropout', type=float, default=0.5)
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
    parser.add_argument('--loss', type=str, default='con_sum',
                        help='con: contrastive loss, sum: summary loss, mar: margin loss,'
                             'tm: text matching loss, np: node perturbation loss')
    parser.add_argument('--edge_coef', type=float, default=0.1)
    parser.add_argument('--neigh_num', type=int, default=3)
    parser.add_argument('--aug_ratio', type=float, default=0.1)

    parser.add_argument('--num_labels', type=int, default=5)
    parser.add_argument('--k_spt', type=int, default=5)
    parser.add_argument('--k_val', type=int, default=5)
    parser.add_argument('--k_qry', type=int, default=50)
    parser.add_argument('--n_way', type=int, default=5)

    parser.add_argument('--context_length', type=int, default=128)
    parser.add_argument('--coop_n_ctx', type=int, default=4)

    parser.add_argument('--position', type=str, default='end')
    parser.add_argument('--class_specific', type=bool, default=False)
    parser.add_argument('--ctx_init', type=bool, default=True)

    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--transformer_heads', type=int, default=8)
    parser.add_argument('--transformer_layers', type=int, default=12)
    parser.add_argument('--transformer_width', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=49408)
    parser.add_argument('--data_name', type=str, default="Arts_Crafts_and_Sewing")
    parser.add_argument('--gpu', type=int, default=2)
    
    parser.add_argument('--topK', type=int, default=1, help='top-k nearest neighbor texts')
    parser.add_argument('--bank_size', type=int, default=2**16)

    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print('device:', device)

    num_nodes = 0
    tit_list = []
    tit_dict = json.load(open('/data/{}/{}_text.json'.format(args.data_name, args.data_name)))
    new_dict = {}

    for i in range(len(tit_dict)):
        num_nodes += 1
        new_dict[i] = tit_dict[str(i)]

    print('num_nodes', num_nodes)

    edge_index = np.load('/data/{}/{}_edge.npy'.format(args.data_name, args.data_name))

    arr_edge_index = edge_index

    edge_index = torch.from_numpy(edge_index).to(device)

    node_f = np.load('/data/{}/{}_f_m.npy'.format(args.data_name, args.data_name))
    node_f = preprocessing.StandardScaler().fit_transform(node_f)
    node_f = torch.from_numpy(node_f).to(device)
    node_f = node_f.to(torch.float32)
    args.gnn_input = node_f.shape[-1]

    id_lab_dict = json.load(
        open('/data/{}/{}_id_labels.json'.format(args.data_name, args.data_name)))
    id_lab_list = sorted(id_lab_dict.items(), key=lambda d: int(d[0]))

    labeled_ids = []
    lab_list = []
    for i in id_lab_list:
        if i[1] != 'nan' or i[1] != '' or i[1] != ' ':
            labeled_ids.append(int(i[0]))
            lab_list.append(i[1])

    labels = sorted(list(set(lab_list)))
    clip_model = CLIP(args)
    load_path = './res/{}/{}_node_ttgt_8&12_0.1_epoch2.pkl'.format(args.data_name, args.data_name)
    
    clip_model.load_state_dict(torch.load(load_path, map_location=device), strict=False)
    print("Load pretrain model: {}".format(load_path))
    start = time.perf_counter()
    all_acc_list = []
    all_f1_list = []
    for i in [2,4,6,8,10]:
       seed = i
       setup_seed(seed)
       main(args)

    end = time.perf_counter()
    acc_mean, acc_std = np.mean(all_acc_list), np.std(all_acc_list)
    f1_mean, f1_std = np.mean(all_f1_list), np.std(all_f1_list)
    print("acc mean and std: ", round(acc_mean, 4), round(acc_std, 4))
    print("f1 mean and std: ", round(f1_mean, 4), round(f1_std, 4))
    print("time consuming {:.2f}".format(end - start))
