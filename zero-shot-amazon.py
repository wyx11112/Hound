import os

import numpy as np
import argparse
import torch
from random import sample
import random
import math
import time
from model import CLIP, tokenize
from torch import nn, optim
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score
from multitask_amazon import multitask_data_generator
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
    pred_list = []
    test_gt = []
    for j in range(len(task_list)):
        for a in test_idx[j]:
            test_gt.append(id_lab_dict[str(a)])
        model.eval()
        task_lables_arr = np.array(labels)[task_list[j]]
        task_lables = task_lables_arr.tolist()

        task_prompt = []
        for a in range(len(task_lables)):
            prompt = the_template + ' ' + task_lables[a]
            task_prompt.append(prompt)
        # print('task_prompt', task_prompt)
        test_labels = tokenize(task_prompt, context_length=args.context_length).to(device)
        with torch.no_grad():
            syn_class = model.encode_text(test_labels)
            syn_class_no = model.encode_text(test_labels, "no")

        Data = DataHelper(edge_index.cpu().numpy(), args, test_idx[j])

        loader = DataLoader(Data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        node_feas = []
        for i_batch, sample_batched in enumerate(loader):
            idx_train = sample_batched['s_n'].to(device)
            with torch.no_grad():
                node_fea = model.encode_graph(node_f.float(), edge_index, training=False)[idx_train]
                node_feas.append(node_fea)

        node_feas = torch.cat(node_feas, dim=0)

        syn_class /= syn_class.norm(dim=-1, keepdim=True)
        syn_class_no /= syn_class_no.norm(dim=-1, keepdim=True)
        node_feas /= node_feas.norm(dim=-1, keepdim=True)
        similarity_yes = (100.0 * node_feas @ syn_class.T).softmax(dim=-1)
        
        similarity_no = (100.0 * node_feas @ syn_class_no.T).softmax(dim=-1)
        similarity_yes_hat = 1. - similarity_no
        similarity = (similarity_yes + similarity_yes_hat) / 2

        pred = similarity.argmax(dim=-1)
        pred = pred.cpu().numpy().reshape(-1)
        y_pred = task_lables_arr[pred]
        pred_list.append(y_pred)
    pred_list = np.concatenate(pred_list, axis=0)
    test_gt = np.array(test_gt)
    acc = accuracy_score(test_gt, pred_list)
    all_acc_list.append(acc)
    f1 = f1_score(test_gt, pred_list, average='macro')
    all_f1_list.append(f1)
    print('acc', round(acc, 4))
    print('macro f1', round(f1, 4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--aggregation_times', type=int, default=2, help='Aggregation times')
    parser.add_argument('--ft_epoch', type=int, default=50, help='fine-tune epoch')
    parser.add_argument('--lr', type=float, default=2e-5)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gnn', type=str, default='GFormer', help='GCN, GFormer')
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
    parser.add_argument('--hops', type=int, default=3)
    parser.add_argument('--pe_dim', type=int, default=8)

    parser.add_argument('--nn_weight', type=float, default=0.5, help='loss weight for tm loss')
    parser.add_argument('--aug_weight', type=float, default=0.5, help='loss weight for np loss')
    parser.add_argument('--topK', type=int, default=1, help='top-k similar texts')
    parser.add_argument('--half', action='store_true', help='use half float to accelerate training')
    parser.add_argument('--edge_coef', type=float, default=0.1)
    parser.add_argument('--neigh_num', type=int, default=3)

    parser.add_argument('--loss', type=str, default='con_sum_aug',
                        help='con: contrastive loss, sum: summary loss, mar: margin loss,'
                             'tm: text matching loss, np: node perturbation loss,'
                             'mar loss is only used for zero-shot pre-training')
    parser.add_argument('--num_labels', type=int, default=5)
    parser.add_argument('--k_spt', type=int, default=5)
    parser.add_argument('--k_val', type=int, default=5)
    parser.add_argument('--k_qry', type=int, default=50)
    parser.add_argument('--n_way', type=int, default=5)

    parser.add_argument('--context_length', type=int, default=128)
    parser.add_argument('--coop_n_ctx', type=int, default=4)
    parser.add_argument('--prompt_lr', type=float, default=0.01)

    parser.add_argument('--position', type=str, default='end')
    parser.add_argument('--class_specific', type=bool, default=False)
    parser.add_argument('--ctx_init', type=bool, default=True)

    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--transformer_heads', type=int, default=8)
    parser.add_argument('--transformer_layers', type=int, default=12)
    parser.add_argument('--transformer_width', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=49408)
    # parser.add_argument('--data_name', type=str, default="Arts_Crafts_and_Sewing")
    # parser.add_argument('--data_name', type=str, default="Industrial_and_Scientific")
    # parser.add_argument('--data_name', type=str, default="Musical_Instruments")
    parser.add_argument('--data_name', type=str, default="Fitness",
                        help='Ele-Computer,Ele-Photo,Arts_Crafts_and_Sewing,History,Children,Industrial_and_Scientific')

    parser.add_argument('--gpu', type=int, default=3)

    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print('device:', device)
    path=os.path.dirname(__file__)
    num_nodes = 0
    tit_list = []
    tit_dict = json.load(open('/data/{}/{}_text.json'.format(args.data_name,args.data_name)))
    new_dict = {}

    for i in range(len(tit_dict)):
        num_nodes += 1
        new_dict[i] = tit_dict[str(i)]

    print('num_nodes', num_nodes)

    edge_index = np.load('/data/{}/{}_edge.npy'.format(args.data_name,args.data_name))

    arr_edge_index = edge_index

    edge_index = torch.from_numpy(edge_index).to(device)

    node_f = np.load('/data/{}/{}_f_m.npy'.format(args.data_name,args.data_name))
    node_f = preprocessing.StandardScaler().fit_transform(node_f)
    node_f = torch.from_numpy(node_f).to(device)
    args.gnn_input = node_f.shape[-1]

    id_lab_dict = json.load(open('/data/{}/{}_id_labels.json'.format(args.data_name,args.data_name)))
    id_lab_list = sorted(id_lab_dict.items(), key=lambda d: int(d[0]))

    labeled_ids = []
    lab_list = []
    for i in id_lab_list:
        if i[1] != 'nan' or i[1] != '' or i[1] != ' ':
            labeled_ids.append(int(i[0]))
            lab_list.append(i[1])

    labels = sorted(list(set(lab_list)))
    start = time.perf_counter()
    
    the_list = ['', 'a ', 'an ', 'of ', 'art ', 'sewing ', 'art of ', 'sewing of ', 'arts crafts of ', 'arts crafts or sewing of ', 'an arts crafts or sewing of ']
    the_template = the_list[0]
    model = CLIP(args, training=False).to(device)
    path = './res/{}/{}_node_ttgt_8&12_0.1_epoch2.pkl'.format(args.data_name, args.data_name)
    model.load_state_dict(torch.load(path, map_location=device), strict=False)
    print('Load model from', path)
    all_acc_list = []
    all_f1_list = []
    for i in [1,2,4,8,16]:
        seed = i
        setup_seed(i)
        print("+ Begin {}-th test...".format(i))
        main(args)
    print('loss:',args.loss)
    end = time.perf_counter()
    ans = round(np.mean(all_acc_list).item(), 4)
    ans_std = round(np.std(all_acc_list).item(), 4)
    print('zero shot acc mean and std: ', ans, ans_std)

    ans = round(np.mean(all_f1_list).item(), 4)
    ans_std = round(np.std(all_f1_list).item(), 4)
    print('zero shot macro f1 mean and std: ', ans, ans_std)
    print("time consuming {:.2f}".format(end - start))


