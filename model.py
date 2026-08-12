import torch
import numpy as np

class TimeAwareMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_size, head_num, dropout_rate, dev):
        super(TimeAwareMultiHeadAttention, self).__init__()
        self.Q_w = torch.nn.Linear(hidden_size, hidden_size)
        self.K_w = torch.nn.Linear(hidden_size, hidden_size)
        self.V_w = torch.nn.Linear(hidden_size, hidden_size)

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = hidden_size // head_num
        self.dropout_rate = dropout_rate
        self.dev = dev

    def forward(self, queries, keys, time_mask, attn_mask, time_matrix_K, time_matrix_V, abs_pos_K, abs_pos_V):
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)

        # head dim * batch dim for parallelization (h*N, T, C/h)
        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

        time_matrix_K_ = torch.cat(torch.split(time_matrix_K, self.head_size, dim=3), dim=0)
        time_matrix_V_ = torch.cat(torch.split(time_matrix_V, self.head_size, dim=3), dim=0)
        abs_pos_K_ = torch.cat(torch.split(abs_pos_K, self.head_size, dim=2), dim=0)
        abs_pos_V_ = torch.cat(torch.split(abs_pos_V, self.head_size, dim=2), dim=0)

        # batched channel wise matmul to gen attention weights
        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))
        attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))
        attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)

        # seq length adaptive scaling
        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)

        time_mask = time_mask.unsqueeze(-1).repeat(self.head_num, 1, 1)
        time_mask = time_mask.expand(-1, -1, attn_weights.shape[-1])
        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)
        paddings = torch.ones(attn_weights.shape) *  (-2**32+1) # -1e23 # float('-inf')
        paddings = paddings.to(self.dev)
        attn_weights = torch.where(time_mask, paddings, attn_weights) # True:pick padding
        attn_weights = torch.where(attn_mask, paddings, attn_weights) # enforcing causalit
        attn_weights = self.softmax(attn_weights) # code as below invalids pytorch backward rules
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(V_)
        outputs += attn_weights.matmul(abs_pos_V_)
        outputs += attn_weights.unsqueeze(2).matmul(time_matrix_V_).reshape(outputs.shape).squeeze(2)

        # (num_head * N, T, C / num_head) -> (N, T, C)
        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2) # div batch_size

        return outputs


class TemporalPatternAttention(torch.nn.Module):
    def __init__(self, num_filters, hidden_units, window_size, dropout_rate=0.2):
        super(TemporalPatternAttention, self).__init__()

        self.num_filters = num_filters
        self.hidden_units = hidden_units
        self.window_size = window_size
        self.conv1d = torch.nn.Conv1d(window_size, num_filters, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.W_a = torch.nn.Linear(num_filters, hidden_units)
        # self.W_v = torch.nn.Linear(num_filters, hidden_units)
        # self.W_h = torch.nn.Linear(hidden_units, hidden_units)

    def forward(self, query):
        last_state = query[:, -1, :]  # (batch, d)
        query_ = query[:, -self.window_size - 1:-1, :]  # (batch, window, d)

        # Apply Conv1D across features
        H_C = self.dropout(self.relu(self.conv1d(query_)))
        H_C = H_C.permute(0, 2, 1)  # (batch, d, k)

        # Apply scoring function
        f_h = self.relu(self.W_a(H_C))
        f_h = torch.matmul(f_h, last_state.unsqueeze(-1)).squeeze(-1)  # (batch, d)
        alpha = self.sigmoid(f_h)  # (batch, d)

        # v_t = torch.matmul(alpha.unsqueeze(-1).transpose(1, 2), H_C).squeeze(1)  # (batch, k)
        # z = self.relu(self.W_v(v_t)) + self.relu(self.W_h(last_state))  # (batch, d)

        # feats = torch.cat((query[:, -self.window_size:, :], z.unsqueeze(1)), dim=1)  # (batch, window+1, d)
        logits = query * alpha.unsqueeze(1)

        return logits


class TPASASRec(torch.nn.Module):  # similar to torch.nn.MultiheadAttention
    def __init__(self, user_num, item_num, time_num, batch_size=32, maxlen=30,
                 hidden_units=50, dropout_rate=0.2, time_span=256, num_blocks=2, num_heads=1,
                 num_filters=32, window_size=30, device='cpu'):
        super(TPASASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = device

        self.item_emb = torch.nn.Embedding(self.item_num + 1, hidden_units, padding_idx=0)
        self.item_emb_dropout = torch.nn.Dropout(p=dropout_rate)

        self.abs_pos_K_emb = torch.nn.Embedding(maxlen, hidden_units)
        self.abs_pos_V_emb = torch.nn.Embedding(maxlen, hidden_units)
        self.time_matrix_K_emb = torch.nn.Embedding(time_span + 1, hidden_units)
        self.time_matrix_V_emb = torch.nn.Embedding(time_span + 1, hidden_units)

        self.item_emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self.abs_pos_K_emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self.abs_pos_V_emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self.time_matrix_K_dropout = torch.nn.Dropout(p=dropout_rate)
        self.time_matrix_V_dropout = torch.nn.Dropout(p=dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            self.attention_layernorms.append(torch.nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(TimeAwareMultiHeadAttention(hidden_units, num_heads, dropout_rate, device))

            self.forward_layernorms.append(torch.nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(TemporalPatternAttention(num_filters, hidden_units, window_size, dropout_rate))

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def seq2feats(self, user_ids, log_seqs, time_matrices):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs = self.item_emb_dropout(seqs)

        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        positions = torch.LongTensor(positions).to(self.dev)
        abs_pos_K = self.abs_pos_K_emb(positions)
        abs_pos_V = self.abs_pos_V_emb(positions)
        abs_pos_K = self.abs_pos_K_emb_dropout(abs_pos_K)
        abs_pos_V = self.abs_pos_V_emb_dropout(abs_pos_V)

        time_matrices = torch.LongTensor(time_matrices).to(self.dev)
        time_matrix_K = self.time_matrix_K_emb(time_matrices)
        time_matrix_V = self.time_matrix_V_emb(time_matrices)
        time_matrix_K = self.time_matrix_K_dropout(time_matrix_K)
        time_matrix_V = self.time_matrix_V_dropout(time_matrix_V)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            # Self-attention, Q=layernorm(seqs), K=V=seqs
            # seqs = torch.transpose(seqs, 0, 1) # (N, T, C) -> (T, N, C)
            Q = self.attention_layernorms[i](seqs)  # PyTorch mha requires time first fmt
            mha_outputs = self.attention_layers[i](Q, seqs,
                                                   timeline_mask, attention_mask,
                                                   time_matrix_K, time_matrix_V,
                                                   abs_pos_K, abs_pos_V)
            seqs = Q + mha_outputs
            # seqs = torch.transpose(seqs, 0, 1) # (T, N, C) -> (N, T, C)

            # Point-wise Feed-forward, actually 2 Conv1D for channel wise fusion
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        attn_feats = self.last_layernorm(seqs)

        return attn_feats

    def forward(self, user_ids, log_seqs, time_matrices, pos_seqs, neg_seqs):  # for training
        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices)
        item_embs = self.item_emb(torch.LongTensor(torch.arange(self.item_num + 1)).to(self.dev))
        item_logits = log_feats.matmul(item_embs.T)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits, item_logits  # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, time_matrices, k=10, item_ids=None):  # for inference
        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices)
        final_feat = log_feats[:, -1, :]

        if item_ids is None:
            item_embs = self.item_emb(torch.LongTensor(torch.arange(self.item_num + 1)).to(self.dev))  # (U, I, C)
            pred = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
            recc = pred.argsort(dim=1, descending=True)[:, :k]
        else:
            item_embs = self.item_emb(torch.LongTensor(item_ids).to(self.dev))  # (U, I, C)
            pred = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
            recc = torch.gather(item_ids, dim=1, index=pred.argsort(dim=1, descending=True)[:, :k])

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return pred, recc  # preds # (U, I)
