import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#import matplotlib as mpl
#import matplotlib.pyplot as plt

TT = torch.Tensor



class Block(nn.Module):
    """ Encoder block """
    def __init__(self, d_model:int, n_heads:int, d_ff:int | None, dropout: float = 0.1):
        super.__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MHAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: TT, mask=None):
        a, attn_weights = self.attn(self.ln1(x), mask)
        x = x + self.drop(a)   
        x = x + self.drop(self.ff(self.ln2(x)))
        return x, attn_weights   

    pass



class Transformer(nn.Module):
    """ Encoder-only transformer """
    def __init__(self, d_in: int, d_model: int, n_heads: int, n_layers: int, 
                max_len: int = 512, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)] 
        )
        self.ln_f = nn.LayerNorm(d_model)


    def forward(self, x: TT, mask=None):
        B, T, _ = x.shape
        x = self.drop(self.in_proj(x) + self.pos[:, :T])

        if mask is not None and mask.dim() == 2:
            mask = mask[:, None, None, :]

        attn_maps = []
        for block in self.blocks:
            x, w = block(x, mask)
            attn_maps.append(w)

        return self.ln_f(x), attn_maps


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),   # 64 -> 256
            nn.GELU(),                  # nonlinear
            nn.Dropout(dropout),        # regularization, only during training
            nn.Linear(d_ff, d_model),   # 256 -> 64
        )

    def forward(self, x: TT) -> TT:
        return self.net(x)


# Multi Head Attention
class MHAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be dividable by n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn = ScaledDPAttention()

    def _split(self, x:TT) -> TT:
        B, T, _  = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, x:TT, mask=None):
        B, T, d_model = x.shape
        q = self._split(self.w_q(x))
        k = self._split(self.w_k(x))
        v = self._split(self.w_v(x))

        out, attn_weights = self.attn(q, k, v, mask)

        out = out.transpose(1, 2).contiguous().view(B, T, d_model)
        
        return self.out_proj(out), attn_weights



class ScaledDPAttention(nn.Module):
    """
        Q, K, ->MatMul -> Scale -> Mask -> SoftMax -- > MatMul -> OUT
                                        V -/
    """

    def forward(self, q: TT, k: TT, v: TT, mask=None):

        d_k = q.size(-1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        out = attn_weights @ v
        
        return (out, attn_weights)


def main():
    q, k, v = (torch.randn(2, 4, 10, 16) for _ in range(3)) 
    out, _ = ScaledDPAttention()(q, k, v)

    ref = F.scaled_dot_product_attention(q, k, v)

    print(torch.allclose(out, ref, atol=1e-6))

if __name__ == "__main__":
    main()