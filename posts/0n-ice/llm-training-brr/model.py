import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Block(nn.Module):
    def __init__(self, d_model, n_head, dropout, norm, ffn, prepost):
        super().__init__()
        if norm == "layer":
            self.n1 = nn.LayerNorm(d_model)
            self.n2 = nn.LayerNorm(d_model)
        else:
            self.n1 = nn.RMSNorm(d_model)
            self.n2 = nn.RMSNorm(d_model)
        
        self.prepost = prepost
        self.resid_drop = nn.Dropout(dropout)
        
        self.attn = nn.MultiheadAttention(embed_dim=d_model,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True,
            bias=False
        )
                
        if ffn == "mlp":
            self.ffn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
                nn.Dropout(dropout),
            )
        else:            
            hidden = int(8 * d_model / 3)  # ~param-match to 4*d_model MLP
            self.ffn = SwiGLU(d_model, hidden, dropout)
        
    def forward(self, x):
        T = x.size(1)
        attn_mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1)
        
        if self.prepost == "pre":
            xn = self.n1(x)
            y, _ = self.attn(xn, xn, xn, need_weights=False, is_causal=True, attn_mask=attn_mask)
            x = x + self.resid_drop(y)
            y = self.ffn(self.n2(x))
            x = x + self.resid_drop(y)
        else:
            y, _ = self.attn(x, x, x, need_weights=False, is_causal=True, attn_mask=attn_mask)
            x = self.n1(x + self.resid_drop(y))
            y = self.ffn(x)
            x = self.n2(x + self.resid_drop(y))
        
        return x

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 2 * hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        u, v = self.fc1(x).chunk(2, dim=-1)      # [*, 2*hidden] -> two [*, hidden]
        x = F.silu(u) * v                         # SwiGLU: swish(u) ⊙ v
        x = self.fc2(x)
        return self.drop(x)

class SinusoidalPositionalEmbedding(nn.Module):
	def __init__(self, num_positions, d_model):
		super(SinusoidalPositionalEmbedding, self).__init__()
		self.num_positions = num_positions
		self.d_model = d_model
		self.register_buffer("pos_embedding", self.create_positional_embedding())

	def create_positional_embedding(self):
		position = torch.arange(0, self.num_positions).unsqueeze(1)
		div = torch.exp(
			torch.arange(0, self.d_model, 2) * -(math.log(10000.0) / self.d_model)
		)

		pos_embedding = torch.zeros(self.num_positions, self.d_model)
		pos_embedding[:, 0::2] = torch.sin(position * div)
		pos_embedding[:, 1::2] = torch.cos(position * div)

		return pos_embedding.unsqueeze(0)

	def forward(self, positions: torch.Tensor):
		T = positions.size(-1)
		return self.pos_embedding[:, :T, :]

class LLM(nn.Module):
    def __init__(self, vocab_size, context_len, n_layer, n_head, d_model, dropout, norm, ffn, prepost, pos_emb):
        super().__init__()
        assert d_model % n_head == 0
        
        self.context_len = context_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_len, d_model) if pos_emb == "learned" else SinusoidalPositionalEmbedding(context_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_head, dropout, norm, ffn, prepost) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model) if norm == "layer" else nn.RMSNorm(d_model)
        self.prepost = prepost
        self.head = nn.Linear(d_model, vocab_size, bias=False)
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
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, context_len=None):
        self.eval()
        device = idx.device
        if context_len is None:
            context_len = self.context_len
        x = idx
        for _ in range(max_new_tokens):
            logits = self(x[:, -context_len:])[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None and 0 < top_k < logits.size(-1):
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_id], dim=1)
        return x