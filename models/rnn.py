import torch
import torch.nn as nn

class VanillaRNNModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=256, hidden_dim=512, pad_idx=0):
        super().__init__()
        # Separate embedding layers for Source (Nepali) and Target (English)
        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=pad_idx)
        
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, src, tgt):
        # 1. Embed source and target independently
        src_embeds = self.src_embedding(src)  # (batch, src_len, embed_dim)
        tgt_embeds = self.tgt_embedding(tgt)  # (batch, tgt_len, embed_dim)
        
        # 2. Concatenate along sequence length (dimension 1)
        combined_embeds = torch.cat([src_embeds, tgt_embeds], dim=1)
        
        # 3. Pass concatenated sequence through RNN
        outputs, _ = self.rnn(combined_embeds)
        
        # 4. Slice only the target sequence timesteps for prediction
        tgt_len = tgt.size(1)
        logits = self.fc(outputs[:, -tgt_len:, :])
        return logits