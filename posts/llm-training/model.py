import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, norm, ffn, prepost):
        super().__init__()
        if norm == "layer":
            self.n1 = nn.LayerNorm(n_embd)
            self.n2 = nn.LayerNorm(n_embd)
        else:
            self.n1 = nn.RMSNorm(n_embd)
            self.n2 = nn.RMSNorm(n_embd)
        
        self.prepost = prepost
        
        self.attn = nn.MultiHeadAttention(embed_dim=n_embd,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True,
            bias=False
        )
                
        if ffn == "mlp":
            self.ffn = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.GELU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(dropout),
            )
        else:            
            hidden = int(8 * n_embd / 3)  # ~param-match to 4*n_embd MLP
            self.ffn = SwiGLU(n_embd, hidden, dropout)
        
    def forward(self, x):
        if self.prepost == "pre":
            x = x + self.attn(self.n1(x))
            x = x + self.ffn(self.n2(x))
        else:
            x = self.n1(x+self.attn(x))
            x = self.n2(x+self.ffn(x))
        
        return x

class SwiGLU(nn.Module):
    def __init__(self, n_embd: int, hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 2 * hidden)
        self.fc2 = nn.Linear(hidden, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        u, v = self.fc1(x).chunk(2, dim=-1)      # [*, 2*hidden] -> two [*, hidden]
        x = F.silu(u) * v                         # SwiGLU: swish(u) ⊙ v
        x = self.fc2(x)
        return self.drop(x)

class LLM(nn.Module):
    def __init__(self, vocab_size, context_len, n_layer, n_head, n_embd, dropout, norm, ffn, prepost):
        super().__init__()
        self.context_len = context_len
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(context_len, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout, norm, ffn, prepost) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.prepost = prepost
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.context_len
        
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        
        for blk in self.blocks:
            x = blk(x)
            
        if self.prepost == "pre":    
            x = self.ln_f(x)
            
        logits = self.head(x)
        return logits