"""Core MTHGNN architecture for cross-subject EEG emotion recognition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

def sinusoidal_positional_encoding(seq_len: int, dim: int, device: torch.device) -> torch.Tensor:
    """Return sinusoidal positional encodings of shape [seq_len, dim]."""
    position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def build_batch_hypergraph_adjacency(H: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Compute Dv^-1/2 H De^-1 H^T Dv^-1/2 for batched incidence matrices.

    H has shape [batch, nodes, hyperedges]; the result has shape
    [batch, nodes, nodes]. Hyperedge weights are the identity.
    """
    d_e = torch.sum(H, dim=1) + eps
    d_v = torch.sum(H, dim=2) + eps
    H_scaled = H / d_e.unsqueeze(1)
    M = torch.bmm(H_scaled, H.transpose(1, 2))
    sqrt_dv = torch.sqrt(d_v)
    denom = (sqrt_dv.unsqueeze(2) * sqrt_dv.unsqueeze(1)) + eps
    G = M / denom
    return G

class HypergraphConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(HypergraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        nn.init.xavier_normal_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        """Apply shared or batched hypergraph propagation to node features."""
        out = torch.matmul(G, x)
        out = torch.matmul(out, self.weight)
        return out + self.bias if self.bias is not None else out

def generate_cheby_adj(G: torch.Tensor, num_layers: int) -> list:
    """Return [I, G, G^2, ...] for a batched adjacency matrix."""
    support = []
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

class MTHGNN(nn.Module):
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
        super(MTHGNN, self).__init__()
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes
        self.hid_channels = hid_channels
        self.num_hyperedges = num_hyperedges
        self.d = dropout

        self.hcheby = HyperChebyNet(in_channels, num_layers, hid_channels)
        self.bn = nn.BatchNorm1d(in_channels)
        self.fc1 = nn.Linear(num_electrodes * hid_channels, hid_channels)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hid_channels, nhead=4, dim_feedforward=hid_channels, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.temp_attn = nn.Linear(hid_channels, 1)
        # Reliability-guided temporal aggregation.
        self.gate_alpha = 5.0
        self.gate_mix_lambda = 0.5
        self.w_t_floor = 1e-4
        self.pool_mix_lambda = 0.5
        self.in_ln = nn.LayerNorm(hid_channels)
        self.post_ln = nn.LayerNorm(hid_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hid_channels, num_classes)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.temp_attn.weight)

        # Sequence-level and time-step contrastive projections.
        self.seq_proj = nn.Sequential(
            nn.Linear(hid_channels, hid_channels),
            nn.ReLU(),
            nn.Linear(hid_channels, hid_channels)
        )
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

        # Learned hyperedge prototypes and query-key projections.
        self.hyperedges = num_hyperedges
        self.gumbel_tau = 0.8
        self.gumbel_hard = True
        self.edge_embed = nn.Parameter(torch.FloatTensor(num_hyperedges, in_channels))
        nn.init.xavier_normal_(self.edge_embed)
        self.q_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.k_proj = nn.Linear(in_channels, in_channels, bias=False)
        nn.init.xavier_normal_(self.q_proj.weight)
        nn.init.xavier_normal_(self.k_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[0], x.shape[1]
        x = x.reshape(B * T, self.num_electrodes, self.in_channels)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)

        # Assign electrodes to learned hyperedge prototypes.
        Q = self.q_proj(x)
        K = self.k_proj(self.edge_embed)
        scores = torch.matmul(Q, K.transpose(0, 1))
        scores = scores / math.sqrt(K.shape[1] + 1e-10)
        H_soft = F.gumbel_softmax(scores, dim=2, tau=self.gumbel_tau, hard=self.gumbel_hard)

        G_b = build_batch_hypergraph_adjacency(H_soft)
        feat_nodes = self.hcheby(x, G_b)
        feat_flat = feat_nodes.reshape(B * T, -1)
        feat_hidden = F.relu(self.fc1(feat_flat))

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
        
        # Estimate time-step reliability from hyperedge-assignment entropy.
        H_seq = H_soft.view(B, T, self.num_electrodes, self.num_hyperedges)
        p = torch.clamp(H_seq, min=1e-10)
        ent_node = -(p * torch.log(p)).sum(dim=3)
        ent_t = ent_node.mean(dim=2)
        w_raw = torch.exp(-self.gate_alpha * ent_t)
        w_t = w_raw / (w_raw.sum(dim=1, keepdim=True) + 1e-10)

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
        """Project sequence embeddings to the normalized contrastive space."""
        z = self.seq_proj(agg)
        z = F.normalize(z, dim=1, eps=1e-10)
        return z

    def encode_view(self, x: torch.Tensor, args) -> torch.Tensor:
        """Encode one structurally augmented view into a sequence embedding."""
        B, T = x.shape[0], x.shape[1]
        N, C = self.num_electrodes, self.in_channels
        X = x.view(B, T, N, C)

        drop_c = float(getattr(args, 'aug_channel_drop_prob', 0.0))
        if drop_c > 0.0:
            cmask = (torch.rand((1, 1, 1, C), device=X.device) > drop_c).float()
            X = X * cmask

        x_nodes = X.view(B * T, N, C)
        x_nodes = self.bn(x_nodes.transpose(1, 2)).transpose(1, 2)

        Q = self.q_proj(x_nodes)
        K = self.k_proj(self.edge_embed)
        scores = torch.matmul(Q, K.transpose(0, 1))
        scores = scores / math.sqrt(K.shape[1] + 1e-10)
        # Mask dropped prototypes before Gumbel-Softmax assignment.
        edge_drop = float(getattr(args, 'aug_edge_drop_prob', 0.0))
        if edge_drop > 0.0:
            keep = (torch.rand((B * T, self.hyperedges), device=x_nodes.device) > edge_drop)
            # Keep one prototype when a draw masks all hyperedges.
            no_keep = (keep.sum(dim=1) == 0)
            if no_keep.any():
                idx_samples = torch.nonzero(no_keep, as_tuple=False).squeeze(1)
                rand_idx = torch.randint(low=0, high=self.hyperedges, size=(idx_samples.shape[0],), device=x_nodes.device)
                keep[idx_samples] = False
                keep[idx_samples, rand_idx] = True
            keep_exp = keep.unsqueeze(1).expand(B * T, scores.shape[1], self.hyperedges)
            scores = scores.masked_fill(~keep_exp, -1e9)
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

        # Estimate time-step reliability from hyperedge-assignment entropy.
        H_seq = H_soft.view(B, T, N, self.hyperedges)
        p = torch.clamp(H_seq, min=1e-10)
        ent_node = -(p * torch.log(p)).sum(dim=3)
        ent_t = ent_node.mean(dim=2)
        w_raw = torch.exp(-self.gate_alpha * ent_t)
        w_t = w_raw / (w_raw.sum(dim=1, keepdim=True) + 1e-10)

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
        """Return two L2-normalized sequence-level contrastive embeddings."""
        agg1 = self.encode_view(x, args)
        agg2 = self.encode_view(x, args)
        z1 = self.project_seq(agg1)
        z2 = self.project_seq(agg2)
        return z1, z2

    def encode_temporal_views(self, x: torch.Tensor, args):
        """Return temporal contrastive embeddings and their flattened index map.

        The outputs have shapes [B*T, D], [B*T, D], and [B*T].
        """
        B, T = x.shape[0], x.shape[1]
        N, C = self.num_electrodes, self.in_channels
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

        # Temporal views use a random shift or segment shuffle.
        time_shift_max = int(getattr(args, 'tc_time_shift_max', 0))
        seg_shuffle_prob = float(getattr(args, 'tc_seg_shuffle_prob', 0.0))
        seg_len = int(getattr(args, 'tc_seg_len', 0))

        seq1 = seq.clone()
        seq2 = seq.clone()

        # Map each second-view position to its original time index.
        pos_map = torch.arange(T, device=seq.device).unsqueeze(0).repeat(B, 1)
        if seg_len > 0:
            seg_starts = [i for i in range(0, T, seg_len)]
        else:
            seg_starts = []
        for b in range(B):
            applied = False
            if seg_shuffle_prob > 0.0 and seg_len > 0 and len(seg_starts) >= 2 and random.random() < seg_shuffle_prob:
                segments = []
                for s in seg_starts:
                    e = min(s + seg_len, T)
                    segments.append(torch.arange(s, e, device=seq.device))
                order = torch.randperm(len(segments))
                new_idx = torch.cat([segments[i] for i in order], dim=0)
                pos_map[b, :new_idx.numel()] = new_idx
                if new_idx.numel() < T:
                    tail = torch.arange(new_idx.numel(), T, device=seq.device)
                    pos_map[b, new_idx.numel():] = tail
                applied = True
            if not applied and time_shift_max > 0:
                shift = random.randint(-time_shift_max, time_shift_max)
                pos_map[b] = (torch.arange(T, device=seq.device) + shift) % T

        # Reorder the second view before adding positional encodings.
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

        z1_t = self.time_proj(time_feat1).contiguous().view(B * T, -1)
        z2_t = self.time_proj(time_feat2).contiguous().view(B * T, -1)
        z1_t = F.normalize(z1_t, dim=1, eps=1e-10)
        z2_t = F.normalize(z2_t, dim=1, eps=1e-10)
        # Convert per-sample time indices to flattened batch-time indices.
        base = (torch.arange(B, device=seq.device) * T).unsqueeze(1).repeat(1, T)
        pos_map_flat = (base + pos_map).contiguous().view(-1)
        return z1_t, z2_t, pos_map_flat

