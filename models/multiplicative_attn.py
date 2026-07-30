import torch
import torch.nn as nn
from models.seq2seq import Encoder

class MultiplicativeAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, keys, mask=None):
        query_proj = self.W(query)
        scores = torch.bmm(query_proj, keys.transpose(1, 2))
        if mask is not None:
            # mask: (batch, 1, src_len), True where PAD -> block attention there
            scores = scores.masked_fill(mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        return context, weights

class MultiplicativeAttentionSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=256, hidden_dim=512, pad_idx=0):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embed_dim, hidden_dim, pad_idx)
        self.attn = MultiplicativeAttention(hidden_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)
        self.pad_idx = pad_idx

    def forward(self, src, tgt):
        enc_outputs, hidden = self.encoder(src)
        # (batch, 1, src_len) True at PAD positions -> block attention there
        src_mask = (src == self.pad_idx).unsqueeze(1)
        outputs = []
        curr_hidden = hidden
        for t in range(tgt.size(1)):
            embed = self.tgt_embedding(tgt[:, t:t+1])
            context, _ = self.attn(curr_hidden.transpose(0, 1), enc_outputs, mask=src_mask)
            out, curr_hidden = self.gru(torch.cat([embed, context], dim=2), curr_hidden)
            outputs.append(self.fc(out))
        return torch.cat(outputs, dim=1)