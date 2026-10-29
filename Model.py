"""Core MTHGNN architecture for cross-subject EEG emotion recognition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

def sinusoidal_positional_encoding(seq_len: int, dim: int, device: torch.device) -> torch.Tensor:
    """
    标准正弦位置编码，形状 [seq_len, dim]。
    """
    position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def build_batch_hypergraph_adjacency(H: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    动态批量归一化：G_b = Dv^{-1/2} H_b De^{-1} H_b^T Dv^{-1/2}
    - H: [B, N, K] 为每个样本的节点-超边隶属度（经 gumbel_softmax 的概率或近似 one-hot）。
    - 返回 G: [B, N, N]
    此实现等价于标准 HGNN 归一化，但不使用超边权重 W（认为 W=I）。
    """
    # 超边度与节点度
    d_e = torch.sum(H, dim=1) + eps            # [B, K]
    d_v = torch.sum(H, dim=2) + eps            # [B, N]

    # De^{-1} 缩放 H 的列（对每个样本分别进行）
    H_scaled = H / d_e.unsqueeze(1)            # [B, N, K]

    # M = H De^{-1} H^T
    M = torch.bmm(H_scaled, H.transpose(1, 2)) # [B, N, N]

    # Dv^{-1/2} 左右缩放
    sqrt_dv = torch.sqrt(d_v)                  # [B, N]
    denom = (sqrt_dv.unsqueeze(2) * sqrt_dv.unsqueeze(1)) + eps  # [B, N, N]
    G = M / denom
    return G

class HypergraphConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(HypergraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        nn.init.xavier_normal_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        # 兼容 2D (N,N) 与 3D (B,N,N) 的 G；torch.matmul 会自动进行批量乘法
        out = torch.matmul(G, x)                  # -> [B, N, C]
        out = torch.matmul(out, self.weight)      # -> [B, N, C_out]
        return out + self.bias if self.bias is not None else out

def generate_cheby_adj(G: torch.Tensor, num_layers: int) -> list:
    """
    支持 batched 的 G：
    - 若 G.shape = [N, N]，返回 [I, G, G^2, ...]
    - 若 G.shape = [B, N, N]，返回 [I_b, G, G@G, ...]
    """
    support = []
    if G.dim() == 2:
        N = G.shape[0]
        for i in range(num_layers):
            if i == 0:
                support.append(torch.eye(N, device=G.device))
            elif i == 1:
                support.append(G)
            else:
                support.append(torch.matmul(support[-1], G))
    else:
        B, N, _ = G.shape
        I = torch.eye(N, device=G.device).unsqueeze(0).expand(B, N, N)
        for i in range(num_layers):
            if i == 0:
                support.append(I)
            elif i == 1:
                support.append(G)
            else:
                support.append(torch.bmm(support[-1], G))
    return support

class HyperChebyNet(nn.Module):
    def __init__(self, in_channels: int, num_layers: int, out_channels: int):
        super(HyperChebyNet, self).__init__()
        self.num_layers = num_layers
        self.hg_layers = nn.ModuleList(HypergraphConvolution(in_channels, out_channels) for _ in range(num_layers))

    def forward(self, x: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        adj_list = generate_cheby_adj(G, self.num_layers)
        result = None
        for i, hg in enumerate(self.hg_layers):
            out_i = hg(x, adj_list[i])
            result = out_i if result is None else (result + out_i)
        result = F.relu(result)
        return result

class TimeAvgDHGNN(nn.Module):
    """Full MTHGNN with global-local hypergraph and temporal contrastive encoders.

    Input: EEG differential-entropy segments of shape [batch, time, electrodes, bands].
    The forward method returns emotion-class logits of shape [batch, num_classes].
    """
    def __init__(self,
                 in_channels: int = 5,
                 num_electrodes: int = 62,
                 num_layers: int = 2,
                 hid_channels: int = 32,
                 num_hyperedges: int = 10,
                 num_classes: int = 3,
                 dropout = 0.3):
        super(TimeAvgDHGNN, self).__init__()
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes
        self.num_layers = num_layers
        self.hid_channels = hid_channels
        self.num_hyperedges = num_hyperedges
        self.num_classes = num_classes
        self.d = dropout

        self.hcheby = HyperChebyNet(in_channels, num_layers, hid_channels)
        self.bn = nn.BatchNorm1d(in_channels)
        self.fc1 = nn.Linear(num_electrodes * hid_channels, hid_channels)
        # Transformer 时间聚合器（提高容量与稳定性）
        encoder_layer = nn.TransformerEncoderLayer(d_model=hid_channels, nhead=4, dim_feedforward=hid_channels, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 时序注意力打分头
        self.temp_attn = nn.Linear(hid_channels, 1)
        # Reliability-gating parameters of the full model.
        self.gate_alpha = 5.0
        self.gate_mix_lambda = 0.5
        self.w_t_floor = 1e-4
        # 注意力与均值池化混合以稳定聚合
        self.pool_mix_lambda = 0.5
        # 归一化层以稳定 Transformer 输入/输出
        self.in_ln = nn.LayerNorm(hid_channels)
        self.post_ln = nn.LayerNorm(hid_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hid_channels, num_classes)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.temp_attn.weight)

        # 视图对比学习的序列级投影头（仅用于对比损失，不影响分类路径）
        self.seq_proj = nn.Sequential(
            nn.Linear(hid_channels, hid_channels),
            nn.ReLU(),
            nn.Linear(hid_channels, hid_channels)
        )
        # 时序步特征对比的投影头（对每个时间步的 hid 做映射）
        self.time_proj = nn.Sequential(
            nn.Linear(hid_channels, hid_channels),
            nn.ReLU(),
            nn.Linear(hid_channels, hid_channels)
        )

        # Local hyperedge features and global-local fusion.
        self.local_attention = nn.MultiheadAttention(
            embed_dim=hid_channels,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.local_proj = nn.Linear(hid_channels, hid_channels)
        nn.init.xavier_normal_(self.local_proj.weight)
        nn.init.zeros_(self.local_proj.bias)
        self.local_hcheby = nn.ModuleList(
            HyperChebyNet(hid_channels, 1, hid_channels)
            for _ in range(num_hyperedges)
        )
        self.global_local_gate = nn.Sequential(
            nn.Linear(2 * hid_channels, hid_channels),
            nn.ReLU(),
            nn.Linear(hid_channels, 1),
            nn.Sigmoid()
        )
        for layer in self.global_local_gate:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        # 动态超边生成：超边原型嵌入 + 查询-键注意力 得到节点-超边打分
        self.hyperedges = num_hyperedges
        self.gumbel_tau = 0.8
        self.gumbel_hard = True
        # 超边原型（K, C_in）
        self.edge_embed = nn.Parameter(torch.FloatTensor(num_hyperedges, in_channels))
        nn.init.xavier_normal_(self.edge_embed)
        # 查询/键映射（保持维度为 C_in，便于与 bn 后的特征对齐）
        self.q_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.k_proj = nn.Linear(in_channels, in_channels, bias=False)
        nn.init.xavier_normal_(self.q_proj.weight)
        nn.init.xavier_normal_(self.k_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[0], x.shape[1]
        # 逐时刻处理图特征
        x = x.reshape(B * T, self.num_electrodes, self.in_channels)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)

        # 1) 节点特征作为 Query，超边原型作为 Key，计算注意力打分
        Q = self.q_proj(x)                                        # [B*T, N, C]
        K = self.k_proj(self.edge_embed)                          # [K, C]
        scores = torch.matmul(Q, K.transpose(0, 1))               # [B*T, N, K] 
        scores = scores / math.sqrt(K.shape[1] + 1e-10)
        # 通过 Gumbel-Softmax 得到节点-超边隶属度（概率分布，提升可解释性）
        H_soft = F.gumbel_softmax(scores, dim=2, tau=self.gumbel_tau, hard=self.gumbel_hard)

        # 2) 归一化得到批量超图邻接 G_b
        G_b = build_batch_hypergraph_adjacency(H_soft)            # [B*T, N, N]

        # 3) 超图卷积
        feat_nodes = self.hcheby(x, G_b)                          # [B*T, N, hid]
        feat_flat = feat_nodes.reshape(B * T, -1) # [B*T, N_nodes*hid]
        feat_hidden = F.relu(self.fc1(feat_flat)) # [B*T, 64]

        # 全局序列特征（每个时间步一个 hid 向量）
        seq = feat_hidden.view(B, T, -1)
        seq = self.dropout(seq)

        # Local hypergraph convolution and hyperedge-attention fusion.
        local_feat_list = []
        for k in range(self.num_hyperedges):
            hyperedge_k = H_soft[:, :, k]
            H_k = hyperedge_k.unsqueeze(-1)
            G_k = build_batch_hypergraph_adjacency(H_k)
            local_nodes_k = self.local_hcheby[k](feat_nodes, G_k)
            weighted_k = local_nodes_k * hyperedge_k.unsqueeze(-1)
            local_vec_k = weighted_k.sum(dim=1)
            local_feat_list.append(local_vec_k.unsqueeze(1))

        local_features = torch.cat(local_feat_list, dim=1)
        attn_out, _ = self.local_attention(local_features, local_features, local_features)
        l_feat = attn_out.mean(dim=1)
        l_feat = self.local_proj(l_feat)
        l_feat = self.dropout(l_feat)
        local_seq = l_feat.view(B, T, -1)
        gate_in = torch.cat([seq, local_seq], dim=-1)
        gate_w = self.global_local_gate(gate_in).squeeze(-1)
        seq = gate_w.unsqueeze(-1) * seq + (1.0 - gate_w).unsqueeze(-1) * local_seq
        
        # 计算基于 H_soft 的时间步可靠性权重 w_t（越低熵越可信）
        H_seq = H_soft.view(B, T, self.num_electrodes, self.num_hyperedges)  # [B,T,N,K]
        p = torch.clamp(H_seq, min=1e-10)
        ent_node = -(p * torch.log(p)).sum(dim=3)                             # [B,T,N]
        ent_t = ent_node.mean(dim=2)                                          # [B,T]
        w_raw = torch.exp(-self.gate_alpha * ent_t)                           # [B,T]
        w_t = w_raw / (w_raw.sum(dim=1, keepdim=True) + 1e-10)                # [B,T]

        # Noise-robust Transformer and reliability-guided temporal aggregation.
        pos = sinusoidal_positional_encoding(T, self.hid_channels, seq.device)
        seq = self.in_ln(seq + pos.unsqueeze(0))
        trans_out = self.transformer(seq)
        trans_out = self.dropout(trans_out)
        trans_out = self.post_ln(trans_out)
        scores_t = self.temp_attn(trans_out).squeeze(-1)
        a = torch.softmax(scores_t, dim=1)
        base_gate = self.gate_alpha * w_t
        uniform = torch.full_like(base_gate, 1.0 / T)
        mixed = self.gate_mix_lambda * base_gate + (1.0 - self.gate_mix_lambda) * uniform
        mixed = mixed + self.w_t_floor
        mixed = mixed / (mixed.sum(dim=1, keepdim=True) + 1e-10)
        a = a * mixed
        a = a / (a.sum(dim=1, keepdim=True) + 1e-10)
        attn_agg = torch.sum(trans_out * a.unsqueeze(-1), dim=1)
        mean_agg = torch.mean(trans_out, dim=1)
        agg = self.pool_mix_lambda * attn_agg + (1.0 - self.pool_mix_lambda) * mean_agg
        agg = F.dropout(agg, p=self.d, training=self.training)
        logits = self.fc2(agg)
        return logits

    def project_seq(self, agg: torch.Tensor) -> torch.Tensor:
        """
        将序列聚合嵌入映射到对比空间并进行 L2 归一化。
        """
        z = self.seq_proj(agg)
        z = F.normalize(z, dim=1, eps=1e-10)
        return z

    def encode_view(self, x: torch.Tensor, args) -> torch.Tensor:
        """
        对单路增强视图进行编码，返回序列聚合嵌入 agg。
        视图增强包括：时间步打乱、通道丢弃/打乱、超边丢弃。
        """
        B, T = x.shape[0], x.shape[1]
        N, C = self.num_electrodes, self.in_channels
        # 变形为 [B,T,N,C]
        X = x.view(B, T, N, C)

        # 通道丢弃
        drop_c = float(getattr(args, 'aug_channel_drop_prob', 0.0))
        if drop_c > 0.0:
            cmask = (torch.rand((1, 1, 1, C), device=X.device) > drop_c).float()
            X = X * cmask

        # 通道打乱（按批统一打乱）
        shuffle_c_prob = float(getattr(args, 'aug_channel_shuffle_prob', 0.0))
        if shuffle_c_prob > 0.0 and torch.rand((), device=X.device).item() < shuffle_c_prob:
            perm_c = torch.randperm(C, device=X.device)
            X = X[:, :, :, perm_c]

        # 时间步打乱（按批统一打乱）
        shuffle_t_prob = float(getattr(args, 'aug_time_shuffle_prob', 0.0))
        if shuffle_t_prob > 0.0 and torch.rand((), device=X.device).item() < shuffle_t_prob:
            perm_t = torch.randperm(T, device=X.device)
            X = X[:, perm_t, :, :]

        # 展平为 [B*T, N, C] 并批归一化
        x_nodes = X.view(B * T, N, C)
        x_nodes = self.bn(x_nodes.transpose(1, 2)).transpose(1, 2)

        # 动态超边打分
        Q = self.q_proj(x_nodes)
        K = self.k_proj(self.edge_embed)
        scores = torch.matmul(Q, K.transpose(0, 1))
        scores = scores / math.sqrt(K.shape[1] + 1e-10)
        # 超边丢弃：在打分阶段屏蔽被丢弃的超边，确保每个节点仍能选择到保留的超边
        edge_drop = float(getattr(args, 'aug_edge_drop_prob', 0.0))
        if edge_drop > 0.0:
            # 为每个样本（B*T）独立采样要保留的超边（Bernoulli keep），形状 [B*T, K]
            keep = (torch.rand((B * T, self.hyperedges), device=x_nodes.device) > edge_drop)
            # 确保每个样本至少保留一个超边，避免全部被屏蔽导致无效分配
            no_keep = (keep.sum(dim=1) == 0)
            if no_keep.any():
                idx_samples = torch.nonzero(no_keep, as_tuple=False).squeeze(1)
                rand_idx = torch.randint(low=0, high=self.hyperedges, size=(idx_samples.shape[0],), device=x_nodes.device)
                keep[idx_samples] = False
                keep[idx_samples, rand_idx] = True
            # 扩展到节点维度并对被丢弃的超边置为极小值，使 Gumbel-Softmax 不会选中它们
            keep_exp = keep.unsqueeze(1).expand(B * T, scores.shape[1], self.hyperedges)
            scores = scores.masked_fill(~keep_exp, -1e9)
        # 通过 Gumbel-Softmax 得到节点-超边隶属度（概率分布/one-hot）
        H_soft = F.gumbel_softmax(scores, dim=2, tau=self.gumbel_tau, hard=self.gumbel_hard)

        # 归一化批量超图邻接并超图卷积
        G_b = build_batch_hypergraph_adjacency(H_soft)
        feat_nodes = self.hcheby(x_nodes, G_b)
        feat_flat = feat_nodes.reshape(B * T, -1)
        feat_hidden = F.relu(self.fc1(feat_flat))

        # 全局序列特征
        seq = feat_hidden.view(B, T, -1)
        seq = self.dropout(seq)

        local_feat_list = []
        for k in range(self.hyperedges):
            hyperedge_k = H_soft[:, :, k]
            H_k = hyperedge_k.unsqueeze(-1)
            G_k = build_batch_hypergraph_adjacency(H_k)
            local_nodes_k = self.local_hcheby[k](feat_nodes, G_k)
            weighted_k = local_nodes_k * hyperedge_k.unsqueeze(-1)
            local_vec_k = weighted_k.sum(dim=1)
            local_feat_list.append(local_vec_k.unsqueeze(1))
        local_features = torch.cat(local_feat_list, dim=1)
        attn_out, _ = self.local_attention(local_features, local_features, local_features)
        l_feat = attn_out.mean(dim=1)
        l_feat = self.local_proj(l_feat)
        l_feat = self.dropout(l_feat)
        local_seq = l_feat.view(B, T, -1)
        gate_in = torch.cat([seq, local_seq], dim=-1)
        gate_w = self.global_local_gate(gate_in).squeeze(-1)
        seq = gate_w.unsqueeze(-1) * seq + (1.0 - gate_w).unsqueeze(-1) * local_seq

        # 基于 H_soft 的时间步权重
        H_seq = H_soft.view(B, T, N, self.hyperedges)
        p = torch.clamp(H_seq, min=1e-10)
        ent_node = -(p * torch.log(p)).sum(dim=3)
        ent_t = ent_node.mean(dim=2)
        w_raw = torch.exp(-self.gate_alpha * ent_t)
        w_t = w_raw / (w_raw.sum(dim=1, keepdim=True) + 1e-10)

        # Aggregate the sequence representation.
        pos = sinusoidal_positional_encoding(T, self.hid_channels, seq.device)
        seq = self.in_ln(seq + pos.unsqueeze(0))
        trans_out = self.transformer(seq)
        trans_out = self.dropout(trans_out)
        trans_out = self.post_ln(trans_out)
        scores_t = self.temp_attn(trans_out).squeeze(-1)
        a = torch.softmax(scores_t, dim=1)
        base_gate = self.gate_alpha * w_t
        uniform = torch.full_like(base_gate, 1.0 / T)
        mixed = self.gate_mix_lambda * base_gate + (1.0 - self.gate_mix_lambda) * uniform
        mixed = mixed + self.w_t_floor
        mixed = mixed / (mixed.sum(dim=1, keepdim=True) + 1e-10)
        a = a * mixed
        a = a / (a.sum(dim=1, keepdim=True) + 1e-10)
        attn_agg = torch.sum(trans_out * a.unsqueeze(-1), dim=1)
        mean_agg = torch.mean(trans_out, dim=1)
        agg = self.pool_mix_lambda * attn_agg + (1.0 - self.pool_mix_lambda) * mean_agg
        agg = F.dropout(agg, p=self.d, training=self.training)
        return agg

    def encode_views(self, x: torch.Tensor, args):
        """
        返回两路增强视图的投影向量 z1, z2（均为 L2 归一化）。
        """
        agg1 = self.encode_view(x, args)
        agg2 = self.encode_view(x, args)
        z1 = self.project_seq(agg1)
        z2 = self.project_seq(agg2)
        return z1, z2

    def encode_temporal_views(self, x: torch.Tensor, args):
        """
        在 Transformer 的时序特征上做两路增强，并输出每个时间步的对比投影向量。
        返回 z1_t, z2_t, pos_map_flat：形状分别为 [B*T, D], [B*T, D], [B*T]。
        其中 pos_map_flat 给出每个样本第 t 个时间步在第二视图中的正样本索引（在 BT 展平索引空间）。
        """
        B, T = x.shape[0], x.shape[1]
        N, C = self.num_electrodes, self.in_channels
        # 基础管线：节点 -> 动态超图 -> 超图卷积 -> 时序序列（不做结构视图增强）
        x_nodes = x.view(B * T, N, C)
        x_nodes = self.bn(x_nodes.transpose(1, 2)).transpose(1, 2)

        Q = self.q_proj(x_nodes)
        K = self.k_proj(self.edge_embed)
        scores = torch.matmul(Q, K.transpose(0, 1))
        scores = scores / math.sqrt(K.shape[1] + 1e-10)
        H_soft = F.gumbel_softmax(scores, dim=2, tau=self.gumbel_tau, hard=self.gumbel_hard)
        G_b = build_batch_hypergraph_adjacency(H_soft)
        feat_nodes = self.hcheby(x_nodes, G_b)
        feat_flat = feat_nodes.reshape(B * T, -1)
        feat_hidden = F.relu(self.fc1(feat_flat))

        seq = feat_hidden.view(B, T, -1)
        seq = self.dropout(seq)

        local_feat_list = []
        for k in range(self.hyperedges):
            hyperedge_k = H_soft[:, :, k]
            H_k = hyperedge_k.unsqueeze(-1)
            G_k = build_batch_hypergraph_adjacency(H_k)
            local_nodes_k = self.local_hcheby[k](feat_nodes, G_k)
            weighted_k = local_nodes_k * hyperedge_k.unsqueeze(-1)
            local_vec_k = weighted_k.sum(dim=1)
            local_feat_list.append(local_vec_k.unsqueeze(1))
        local_features = torch.cat(local_feat_list, dim=1)
        attn_out, _ = self.local_attention(local_features, local_features, local_features)
        l_feat = attn_out.mean(dim=1)
        l_feat = self.local_proj(l_feat)
        l_feat = self.dropout(l_feat)
        local_seq = l_feat.view(B, T, -1)
        gate_in = torch.cat([seq, local_seq], dim=-1)
        gate_w = self.global_local_gate(gate_in).squeeze(-1)
        seq = gate_w.unsqueeze(-1) * seq + (1.0 - gate_w).unsqueeze(-1) * local_seq

        # 视图增强在时序维进行：真正的时间结构增强（移位/分段打乱/反转）
        time_shift_max = int(getattr(args, 'tc_time_shift_max', 0))
        seg_shuffle_prob = float(getattr(args, 'tc_seg_shuffle_prob', 0.0))
        seg_len = int(getattr(args, 'tc_seg_len', 0))

        seq1 = seq.clone()
        seq2 = seq.clone()

        # 构建每个样本的时间索引映射：pos_map[b, t] = 第二视图中与第一视图 t 对齐的时间索引
        pos_map = torch.arange(T, device=seq.device).unsqueeze(0).repeat(B, 1)
        # 逐样本选择一种结构性时间增强（优先分段打乱，否则反转，否则移位）
        if seg_len > 0:
            # 允许最后一段为剩余长度
            seg_starts = [i for i in range(0, T, seg_len)]
        else:
            seg_starts = []
        for b in range(B):
            applied = False
            if seg_shuffle_prob > 0.0 and seg_len > 0 and len(seg_starts) >= 2 and random.random() < seg_shuffle_prob:
                # 构造分段索引列表并打乱段顺序
                segments = []
                for s in seg_starts:
                    e = min(s + seg_len, T)
                    segments.append(torch.arange(s, e, device=seq.device))
                order = torch.randperm(len(segments))
                new_idx = torch.cat([segments[i] for i in order], dim=0)
                pos_map[b, :new_idx.numel()] = new_idx
                if new_idx.numel() < T:
                    # 如果最后一段不足，补齐剩余（保持原顺序）
                    tail = torch.arange(new_idx.numel(), T, device=seq.device)
                    pos_map[b, new_idx.numel():] = tail
                applied = True
            if not applied and time_shift_max > 0:
                shift = random.randint(-time_shift_max, time_shift_max)
                pos_map[b] = (torch.arange(T, device=seq.device) + shift) % T

        # 根据 pos_map 重排第二视图的时间维（在加入位置编码之前进行）
        idx = pos_map.unsqueeze(-1).expand(B, T, seq2.size(-1))
        seq2 = torch.gather(seq2, dim=1, index=idx)

        # Transformer representations for temporal contrastive learning.
        pos = sinusoidal_positional_encoding(T, self.hid_channels, seq.device)
        seq1 = self.in_ln(seq1 + pos.unsqueeze(0))
        seq2 = self.in_ln(seq2 + pos.unsqueeze(0))
        trans1 = self.transformer(seq1)
        trans2 = self.transformer(seq2)
        trans1 = self.dropout(trans1)
        trans2 = self.dropout(trans2)
        time_feat1 = self.post_ln(trans1)
        time_feat2 = self.post_ln(trans2)

        # 投影并归一化到对比空间
        z1_t = self.time_proj(time_feat1).contiguous().view(B * T, -1)
        z2_t = self.time_proj(time_feat2).contiguous().view(B * T, -1)
        z1_t = F.normalize(z1_t, dim=1, eps=1e-10)
        z2_t = F.normalize(z2_t, dim=1, eps=1e-10)
        # 构造 BT 展平空间的索引映射：map[b, t] = b*T + pos_map[b, t]
        base = (torch.arange(B, device=seq.device) * T).unsqueeze(1).repeat(1, T)
        pos_map_flat = (base + pos_map).contiguous().view(-1)
        return z1_t, z2_t, pos_map_flat


MTHGNN = TimeAvgDHGNN
