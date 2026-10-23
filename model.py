import os
import re
import unicodedata
import torch
import torch.nn as nn
import torch.nn.functional as F

class MeyTokenizer:
    def __init__(self):
        self.vocab = []
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0
        # Use regex to find words and punctuation, ignoring whitespace
        self.token_pattern = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

    def fit(self, text):
        # Normalize and lowercase
        normalized_text = ' '.join(text.split()).lower()
        
        # Add version without diacritics to make the vocabulary more robust for Polish
        def strip_diacritics(s):
            return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
        
        augmented_text = normalized_text + ' ' + strip_diacritics(normalized_text)
        tokens = self.token_pattern.findall(augmented_text)
        
        self.vocab = sorted(list(set(tokens)))
        if '<unk>' not in self.vocab:
            self.vocab.insert(0, '<unk>')
        if '<pad>' not in self.vocab:
            self.vocab.insert(0, '<pad>')
            
        self.stoi = {tok: i for i, tok in enumerate(self.vocab)}
        self.itos = {i: tok for tok, i in self.stoi.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        if not text:
            return []
        text = ' '.join(text.split()).lower()
        toks = self.token_pattern.findall(text)
        return [self.stoi.get(t, self.stoi.get('<unk>', 1)) for t in toks]

    def decode(self, token_ids):
        toks = [self.itos.get(i, '<unk>') for i in token_ids]
        # Simple detokenizer
        out = ''
        for t in toks:
            if t in ('<pad>', '<unk>'):
                continue
            if re.match(r'^[^\w\s]$', t, flags=re.UNICODE):
                out += t
            else:
                if out and not out.endswith(' '):
                    out += ' '
                out += t
        return out.strip()

    def save(self, filepath):
        import json
        data = {
            'vocab': self.vocab,
            'stoi': self.stoi,
            'itos': {str(k): v for k, v in self.itos.items()}  # JSON keys must be strings
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath):
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.stoi = data['stoi']
        self.itos = {int(k): v for k, v in data['itos'].items()}
        self.vocab_size = len(self.vocab)


class Head(nn.Module):
    def __init__(self, embed_dim, head_size, block_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # Compute attention scores ("affinities")
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        # Scale by the dimension of one attention head, as in the Transformer paper.
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x) # (B, T, head_size)
        return wei @ v # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, head_size, block_size, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([Head(embed_dim, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, dropout=0.1):
        super().__init__()
        head_size = embed_dim // num_heads
        self.sa = MultiHeadAttention(embed_dim, num_heads, head_size, block_size, dropout)
        self.ffwd = FeedForward(embed_dim, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MeyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=8, num_layers=6, block_size=64, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding_table = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, block_size, dropout) for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # Ensure we don't exceed the position embedding limit
        if T > self.block_size:
            idx = idx[:, -self.block_size:]
            B, T = idx.shape
            
        tok_emb = self.token_embedding_table(idx) # (B, T, embed_dim)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, embed_dim)
        x = tok_emb + pos_emb # (B, T, embed_dim)
        x = self.blocks(x) # (B, T, embed_dim)
        x = self.ln_final(x) # (B, T, embed_dim)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # If targets are shorter than predictions, truncate logits
            if targets.shape[1] < T:
                logits = logits[:, :targets.shape[1], :]
                B, T, C = logits.shape
            
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                min_top = v[:, -1].unsqueeze(1)
                logits = torch.where(logits < min_top, torch.full_like(logits, -1e10), logits)
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
