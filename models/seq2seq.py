import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        embeds = self.embedding(src)
        outputs, hidden = self.gru(embeds)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt, hidden):
        embeds = self.embedding(tgt)
        outputs, hidden = self.gru(embeds, hidden)
        return self.fc(outputs), hidden

class Seq2SeqModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=256, hidden_dim=512, pad_idx=0):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embed_dim, hidden_dim, pad_idx)
        self.decoder = Decoder(tgt_vocab_size, embed_dim, hidden_dim, pad_idx)

    def forward(self, src, tgt):
        _, hidden = self.encoder(src)
        logits, _ = self.decoder(tgt, hidden)
        return logits