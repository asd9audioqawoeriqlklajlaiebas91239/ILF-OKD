import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math


class student_graph(nn.Module):
    def __init__(self, comps, seq_len, args, hidden_dim=100, output_dim=2, bias=True):
        super(student_graph, self).__init__()

        self.comps = comps
        self.num_comps = len(comps)
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.RankAttn_graph = AttentionLayer(ProbAttention(False, factor=4), len(comps), 4)
        self.batch = args.batch_size

    def forward(self, et, mp, co, vol, comp_idx):

        sz = et.size(0)
        et = et.view(self.seq_len, -1)
        co = co.view(self.seq_len, -1)
        mp = mp.view(self.seq_len, -1)
        vol = vol.view(self.seq_len, -1)
        comp_corr = torch.cat([et, co, mp, vol], dim=-1).view(sz, self.seq_len * 4, -1)
        comp_corr = self.RankAttn_graph(comp_corr, comp_corr, comp_corr, None)

        return comp_corr.view(self.seq_len, -1)


class student_ef(nn.Module):
    def __init__(self, comps, seq_len, args, hidden_dim=100, layer_dim=2,
                 output_dim=2, bias=True):
        super(student_ef, self).__init__()

        self.comps = comps
        self.num_comps = len(comps)
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        self.layer_dim = layer_dim
        self.prediction = nn.Linear(7, 2)

        self.ef_expand = nn.Linear(7, len(comps) * 4 * 7)
        self.R = nn.Parameter(torch.empty(size=(self.seq_len, 4, len(comps))))
        nn.init.xavier_uniform_(self.R.data, gain=1.414)

        self.RankAttn = AttentionLayer(ProbAttention(False, factor=10), 7 * len(comps), 4)

    def forward(self, ef, comp_corr):

        sz = ef.size(0)
        comp_corr = comp_corr.view(sz, self.seq_len, 4, -1)
        ef = ef.view(sz * self.seq_len, 7)

        expanded_ef = F.tanh(self.ef_expand(ef)).view(sz, self.seq_len, 4, 7, -1)

        expanded_ef = F.tanh(torch.einsum("blheg, blhg->blheg", expanded_ef, comp_corr))
        expanded_ef = expanded_ef.view(sz, self.seq_len * 4, -1)
        new_ef = self.RankAttn(expanded_ef, expanded_ef, expanded_ef, None)
        new_ef = new_ef.view(sz, 7, self.seq_len, 4, -1)
        new_ef = torch.einsum("bflhg, lhg->bf", new_ef, self.R)

        pred = F.softmax(self.prediction(new_ef))

        return pred, new_ef, self.R


class teacher_graph(nn.Module):
    def __init__(self, comps, seq_len, args, input_dim=16, hidden_dim=459, layer_dim=2, output_dim=2, bias=True):
        super(teacher_graph, self).__init__()

        self.comps = comps
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.seq_len = seq_len
        self.batch = args.batch_size

        self.company_embedding = nn.Embedding(len(comps), len(comps))
        self.et_gcn1 = nn.Sequential(nn.Linear(len(comps), hidden_dim),
                                       nn.Softmax())

        self.co_gcn1 = nn.Sequential(nn.Linear(len(comps), hidden_dim),
                                        nn.Softmax())

        self.mp_gcn1 = nn.Sequential(nn.Linear(len(comps), hidden_dim),
                                     nn.Softmax())

        self.vol_gcn1 = nn.Sequential(nn.Linear(len(comps), hidden_dim),
                                      nn.Softmax())

        self.RankAttn_graph = AttentionLayer(ProbAttention(False, factor=4), hidden_dim, 4)


    def forward(self, et, co, pt, vt, comp_idx):

        comp_weight = self.company_embedding(comp_idx)

        et = self.et_gcn1(comp_weight * et)
        co = self.co_gcn1(comp_weight * co)
        pt = self.mp_gcn1(comp_weight * pt)
        vt = self.vol_gcn1(comp_weight * vt)

        et = et.view(self.batch, self.seq_len, -1)
        co = co.view(self.batch, self.seq_len, -1)
        pt = pt.view(self.batch, self.seq_len, -1)
        vt = vt.view(self.batch, self.seq_len, -1)

        comp_corr = torch.cat([et, co, pt, vt], dim=-1)
        comp_corr = comp_corr.view(self.batch, self.seq_len * 4, -1)
        comp_corr = self.RankAttn_graph(comp_corr, comp_corr, comp_corr, None)

        return comp_corr.view(self.seq_len, -1)


class teacher_ef(nn.Module):
    def __init__(self, comps, seq_len, args, input_dim=16, hidden_dim=459, layer_dim=2,
                 output_dim=2, bias=True):
        super(teacher_ef, self).__init__()

        self.comps = comps
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.seq_len = seq_len
        self.batch = args.batch_size

        self.gru_cell1 = GRUCell(hidden_dim * 7, hidden_dim * 7, layer_dim)
        self.RankAttn = AttentionLayer(ProbAttention(False, factor=10), hidden_dim, 4)

        self.prediction = nn.Linear(7, output_dim)

        self.fusion1 = nn.Sequential(nn.Linear(4 * self.hidden_dim * 7, self.hidden_dim * 7),
                                     nn.Tanh())

        self.c2h = nn.Sequential(nn.Linear(seq_len * self.hidden_dim * 7, 7, bias=bias),
                                 nn.Softmax())

    def forward(self, ef, comp_corr):

        ef = ef.view(self.seq_len, 7)
        h0 = Variable(torch.zeros(self.layer_dim, 1, self.hidden_dim * 7)).to(self.device)

        outs = []
        hn = h0[0, :, :]

        for seq in range(ef.size(0)):
            new_ef = ef[seq, :].view(7, 1)
            corr = comp_corr[seq, :].view(self.batch, -1)
            expanded_ef = torch.matmul(new_ef, corr).view(self.batch, -1)
            expanded_ef = self.fusion1(expanded_ef)
            hn = self.gru_cell1(expanded_ef, hn)
            outs.append(hn)

        hc1 = torch.cat(outs, dim=0).unsqueeze(0)
        hc1 = hc1.view(self.batch, 70, -1)
        new_ef = self.RankAttn(hc1, hc1, hc1, None)
        new_ef = new_ef.view(self.batch, -1)
        new_ef = F.tanh(self.c2h(new_ef).view(self.batch, -1))
        pred = F.softmax(self.prediction(new_ef), dim=1)

        return pred, new_ef, None


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cuda"):
        _mask = torch.ones(L, scores.shape[-1], dytpe=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    def mask(self):
        return self._mask


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L, E = K.shape
        _, _, S, _ = Q.shape

        K_expand = K.unsqueeze(-3).expand(B, H, S, L, E)
        indx_sample = torch.randint(L, (S, sample_k))
        K_sample = K_expand[:, :, torch.arange(S).unsqueeze(1), indx_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.sum(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-1)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)
        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V)

        return context_in, attn

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, D = queries.shape
        _, S, _, _ = keys.shape

        queries = queries.view(B, H, L, -1)
        keys = keys.view(B, H, S, -1)
        values = values.view(B, H, S, -1)

        U = self.factor * np.ceil(np.log(S)).astype('int').item()
        u = self.factor * np.ceil(np.log(L)).astype('int').item()


        scores_top, index = self._prob_QK(queries, keys, u, U)
        # add scale factor
        scale = self.scale or 1. / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        context = self._get_initial_context(values, L)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L, attn_mask)

        return context.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        ).view(B, L, -1)

        return self.out_projection(out)


class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)

        return hy
