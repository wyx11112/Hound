from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Union, List
from simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from torch import nn, optim
from memory_queue import NNMemoryBankModule

_tokenizer = _Tokenizer()


def cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:

    sim = vec1 @ vec2.T
    sim = torch.exp(sim)
    return sim


def text_semantic_opposite_loss(all_text_features, all_text_features_no, mode="L2"):
    if mode == "L2":
        l2_distance = 2 - 2 * (all_text_features * all_text_features_no).sum(
            -1) + 1e-4  # epsilon = 1e-4, used to get rid of inifity gradient
        loss = 2 - torch.sqrt(l2_distance)  # \in [0,2]
    if mode == "cosine":
        loss = (all_text_features * all_text_features_no).sum(-1) + 1.0  # \in [0,2]
    return loss.mean()


def full_attention_conv(qs, ks, vs, output_attn=False):
    # normalize input
    qs = qs / torch.norm(qs, p=2)  # [N, H, M]
    ks = ks / torch.norm(ks, p=2)  # [L, H, M]
    N = qs.shape[0]

    # numerator
    kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
    attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
    attention_num += N * vs

    # denominator
    all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
    attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

    # attentive aggregated results
    attention_normalizer = torch.unsqueeze(
        attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
    attention_normalizer += torch.ones_like(attention_normalizer) * N
    attn_output = attention_num / attention_normalizer  # [N, H, D]

    # compute attention for visualization if needed
    if output_attn:
        attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)  # [N, N]
        normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdims=True)  # [N,1]
        attention = attention / normalizer

    if output_attn:
        return attn_output, attention
    else:
        return attn_output


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class TransConvLayer(nn.Module):
    '''
    transformer with fast attention
    '''

    def __init__(self, in_channels,
                 out_channels,
                 num_heads,
                 use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, edge_index=None, output_attn=False):
        # feature transformation
        query = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            value = self.Wv(source_input).reshape(-1,  self.num_heads, self.out_channels)
        else:
            value = source_input.reshape(-1, 1, self.out_channels)

        # compute full attentive aggregation
        if output_attn:
            attention_output, attn = full_attention_conv(query, key, value, output_attn)  # [N, H, D]
        else:
            attention_output = full_attention_conv(query, key, value)  # [N, H, D]

        final_output = attention_output
        final_output = final_output.mean(dim=1)

        if output_attn:
            return final_output, attn
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, num_heads=1,
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=False):
        super().__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                TransConvLayer(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index):
        layer_ = []
        # input MLP layer
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with full attention aggregation
            x = conv(x, x, edge_index)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]


class GFormer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, num_heads=1,
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=False,
                 graph_weight=0.8, gnn=None, aggregate='add'):
        super().__init__()
        self.trans_conv = TransConv(in_channels, hidden_channels, num_layers, num_heads, alpha, dropout, use_bn,
                                    use_residual, use_weight)
        self.gnn = gnn
        self.graph_weight = graph_weight
        self.use_act = use_act

        self.aggregate = aggregate

        if aggregate == 'add':
            self.fc = nn.Linear(hidden_channels, out_channels)
        elif aggregate == 'cat':
            self.fc = nn.Linear(2 * hidden_channels, out_channels)
        else:
            raise ValueError(f'Invalid aggregate type:{aggregate}')

        self.params1 = list(self.trans_conv.parameters())
        self.params2 = list(self.gnn.parameters()) if self.gnn is not None else []
        self.params2.extend(list(self.fc.parameters()))

    def forward(self, x, adj):
        x1 = self.trans_conv(x, adj)

        x2 = self.gnn(x, adj)
        if self.aggregate == 'add':
            x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
        else:
            x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x

    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x)  # [layer num, N, N]

        return attns

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.gnn.reset_parameters()


class GNN(MessagePassing):
    def __init__(self, args, **kwargs):
        super(GNN, self).__init__(aggr='add', **kwargs)
        self.vars = nn.ParameterList()

        w = nn.Parameter(torch.ones([args.gnn_hid, args.gnn_input]))
        torch.nn.init.xavier_uniform_(w)
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros(args.gnn_hid)))

        w = nn.Parameter(torch.ones([args.gnn_output, args.gnn_hid]))
        torch.nn.init.xavier_uniform_(w)
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros(args.gnn_output)))

    @staticmethod
    def norm(edge_index, num_nodes, improved=False, dtype=None):
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)

        fill_value = 1.0 if not improved else 2.0
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, vars=None):
        if vars is None:
            vars = self.vars
        improved = False

        w, b = vars[0], vars[1]
        edge_index, norm = self.norm(edge_index, x.size(self.node_dim), improved, x.dtype)
        x = self.propagate(edge_index, x=x, norm=norm)
        x = F.linear(x, w, b)
        x = F.leaky_relu(x)

        w, b = vars[2], vars[3]
        edge_index, norm = self.norm(edge_index, x.size(self.node_dim), improved, x.dtype)
        x = self.propagate(edge_index, x=x, norm=norm)
        x = F.linear(x, w, b)

        return x

    def parameters(self):
        return self.vars


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=True, use_bn=True, ):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        # self.convs.append(
        #     GCNConv(in_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            # self.convs.append(
            #     GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # self.convs.append(
        #     GCNConv(hidden_channels, out_channels, cached=not save_mem, normalize=not save_mem))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=not save_mem))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class CLIP(nn.Module):
    def __init__(self, args, training=True):
        super().__init__()

        self.context_length = args.context_length
        self.args = args
        self.edge_coef = args.edge_coef
        self.adapter = Adapter(args.gnn_hid, 4)
        self.training = training

        if args.gnn == "GCN":
            self.gnn = GNN(args)
        elif args.gnn == "GFormer":
            gcn = GCN(in_channels=args.gnn_input,
                      hidden_channels=args.gnn_hid,
                      out_channels=args.gnn_output,
                      num_layers=args.gnn_layers,
                      dropout=args.gnn_dropout,
                      use_bn=args.use_bn)
            self.gnn = GFormer(args.gnn_input, args.gnn_hid, args.gnn_output, num_layers=args.trans_layers,
                                alpha=args.alpha, dropout=args.trans_dropout, num_heads=args.num_heads,
                                use_bn=args.use_bn, use_residual=args.use_residual,
                                use_weight=args.use_weight, use_act=args.use_act,
                                graph_weight=args.graph_weight, gnn=gcn, aggregate=args.aggregate)
        else:
            raise NotImplementedError

        self.transformer = Transformer(
            width=args.transformer_width,
            layers=args.transformer_layers,
            heads=args.transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.transformer_no = Transformer(
            width=args.transformer_width,
            layers=args.transformer_layers,
            heads=args.transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = args.vocab_size
        self.token_embedding = nn.Embedding(args.vocab_size, args.transformer_width)  # the embedding for all possible tokens
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, args.transformer_width))
        self.ln_final = LayerNorm(args.transformer_width)
        self.text_projection = nn.Parameter(torch.empty(args.transformer_width, args.embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.token_embedding_no = nn.Embedding(args.vocab_size, args.transformer_width)
        self.positional_embedding_no = nn.Parameter(torch.empty(self.context_length, args.transformer_width))
        self.ln_final_no = LayerNorm(args.transformer_width)
        self.text_projection_no = nn.Parameter(torch.empty(args.transformer_width, args.embed_dim))
        self.num_no_texts = 16
        self.prompt_no = nn.Parameter(torch.zeros(self.num_no_texts, self.context_length, args.transformer_width))

        if training and "nn" in self.args.loss.split("_"):
            self.nn_text = NNMemoryBankModule(size=args.bank_size, topk=args.topK)
        if args.half:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        self.optim = optim.Adam([{'params': self.token_embedding.weight},
                                 {'params': self.gnn.parameters()},
                                 {'params': self.positional_embedding},
                                 {'params': self.transformer.parameters()},
                                 {'params': self.text_projection},
                                 {'params': self.text_projection_no},
                                 {'params': self.token_embedding_no.weight},
                                 {'params': self.positional_embedding_no},
                                 {'params': self.prompt_no},
                                 {'params': self.transformer_no.parameters()},
                                 ], lr=args.lr)

        self.initialize_parameters()

        self.margin_loss = nn.TripletMarginWithDistanceLoss(distance_function=cosine_similarity, margin=1.,)

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.positional_embedding_no, std=0.01)
        nn.init.normal_(self.token_embedding_no.weight, std=0.02)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        proj_std = (self.transformer_no.width ** -0.5) * ((2 * self.transformer_no.layers) ** -0.5)
        attn_std = self.transformer_no.width ** -0.5
        fc_std = (2 * self.transformer_no.width) ** -0.5
        for block in self.transformer_no.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
            nn.init.normal_(self.text_projection_no, std=self.transformer_no.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_graph(self, x, adj, training=True):
        embs = self.gnn(x, adj)
        if training:
            return embs
        else:
            graph_adapter = self.adapter(embs)
            ratio = 0.5
            graph_features = ratio * graph_adapter + (1 - ratio) * embs
            return graph_features


    def encoder_text_feature(self, x, text, mode):
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x) if mode == "yes" else self.transformer_no(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) if mode == "yes" else self.ln_final_no(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection if mode == "yes" else x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection_no
        return x

    def encode_text(self, text, mode="yes"):
        batch = text.size(0)
        if mode == "yes":
            x = self.token_embedding(text).type(self.dtype) + self.positional_embedding.type(self.dtype)
            x = self.encoder_text_feature(x, text, mode)
        else:
            x = self.token_embedding_no(text).type(self.dtype) + self.positional_embedding_no.type(self.dtype)
            idx = np.random.randint(0, self.num_no_texts + 1, (batch))
            prompt_no = torch.cat([self.prompt_no, torch.mean(self.prompt_no, dim=0, keepdim=True)], 0)[idx]
            x = x + prompt_no
            x = self.encoder_text_feature(x, text, mode)
        return x

    def forward(self, x, adj, adj_aug, s_n, t_n, s_n_text, t_n_text, device, training=True):
        s_graph_features = self.encode_graph(x, adj)[s_n]
        s_text_features = self.encode_text(s_n_text)
        s_text_features_no = self.encode_text(s_n_text, mode="no")

        s_graph_features = s_graph_features / s_graph_features.norm(dim=-1, keepdim=True)
        s_text_features = s_text_features / s_text_features.norm(dim=-1, keepdim=True)
        s_text_features_no = s_text_features_no / s_text_features_no.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        labels = torch.arange(s_graph_features.shape[0]).to(device)
        total_loss = 0.
        loss_list = self.args.loss.split("_")

        for item in loss_list:
            if item == "con":
                logits = logit_scale * s_graph_features @ s_text_features.t()
                loss_i = F.cross_entropy(logits, labels)
                loss_t = F.cross_entropy(logits.T, labels)
                node_loss = (loss_i + loss_t) / 2
                total_loss += node_loss
            elif item == "sum":
                t_text_features = self.encode_text(t_n_text)
                t_text_features = t_text_features.reshape(s_graph_features.shape[0], self.args.neigh_num,
                                                          self.args.gnn_output)
                t_text_features = torch.mean(t_text_features, dim=1, keepdim=False)
                t_text_features = t_text_features / t_text_features.norm(dim=-1, keepdim=True)

                logits_gt = logit_scale * s_graph_features @ t_text_features.t()
                loss_i = F.cross_entropy(logits_gt, labels)
                loss_t = F.cross_entropy(logits_gt.T, labels)
                loss_gt = (loss_i + loss_t) / 2

                logits_tt = logit_scale * s_text_features @ t_text_features.t()
                loss_i = F.cross_entropy(logits_tt, labels)
                loss_t = F.cross_entropy(logits_tt.T, labels)
                loss_tt = (loss_i + loss_t) / 2

                total_loss += (loss_gt + loss_tt) * self.edge_coef
            elif item == "mar":
                mar_loss = self.margin_loss(s_graph_features, s_text_features, s_text_features_no)
                loss_text = text_semantic_opposite_loss(s_text_features, s_text_features_no)
                total_loss += (mar_loss + loss_text) * 0.5
            elif item == "tm":
                text_features_nn = self.nn_text(s_text_features.detach(), s_graph_features.detach())
                text_features_nn = text_features_nn / text_features_nn.norm(dim=-1, keepdim=True)
                logits = logit_scale * s_graph_features @ text_features_nn.t()
                loss_i = F.cross_entropy(logits, labels)
                loss_t = F.cross_entropy(logits.T, labels)
                nn_loss = (loss_i + loss_t) / 2
                total_loss += nn_loss * self.args.nn_weight
            elif item == "np":
                s_graph_features_aug = self.encode_graph(x, adj_aug)[s_n]
                s_graph_features_aug = s_graph_features_aug / s_graph_features_aug.norm(dim=-1, keepdim=True)
                logits = logit_scale * s_graph_features_aug @ s_text_features.t()
                loss_i = F.cross_entropy(logits, labels)
                loss_t = F.cross_entropy(logits.T, labels)
                aug_loss = (loss_i + loss_t) / 2
                total_loss += aug_loss * self.args.aug_weight
            else:
                raise NotImplementedError

        if training == True:
            self.optim.zero_grad()
            torch.cuda.empty_cache()
            total_loss.backward()
            self.optim.step()

            return round((total_loss.detach().clone()).cpu().item(), 4)
        else:
            return logit_scale * s_graph_features @ s_text_features.t()


def tokenize(texts: Union[str, List[str]], context_length: int = 128, truncate: bool = True) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
