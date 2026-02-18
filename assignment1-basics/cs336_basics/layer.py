import torch
from torch import nn
from torch.nn import init
from einops import rearrange, einsum
from torch import Tensor
from jaxtyping import Float, Int, Bool
import math


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.weight = nn.Parameter(torch.empty((out_features,in_features),**factory_kwargs))
        # Initialize weights using truncated normal
        std = (2 / (in_features + out_features)) ** 0.5
        init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x):
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
    
    
class Embedding(nn.Module):
    def __init__(self, 
                 num_embeddings, # vocabulary size
                 embedding_dim, # embedding dimension
                 device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        factory_kwargs = {'device': device, 'dtype': dtype}

        # Weight shape is (num_embeddings, embedding_dim)
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))

        # Initialize weights using truncated normal
        std = 1
        init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids] 
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}

        # Initialize weights to 1
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))
    
    def forward(self, x: Tensor) -> Tensor:

        # Prevent overflow in mean/sqrt calculations
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # Perform RMSNorm calculation 

        RMS = (x.pow(2).mean(dim=-1, keepdim=True)+ self.eps).sqrt() 
        normalized_x = x / RMS
        
        results = normalized_x * self.weight # W will automatically broadcast to ..., d_model

        # Return the result in the original dtype
        return results.to(in_dtype)


def sigmoid(x: Tensor): return 1 / (1 + torch.exp(-x)) # sigmoid activation function

def silu(x: Tensor): return x * torch.sigmoid(x) # SiLU activation function

def glu(a:Tensor, b:Tensor): return a * b # element-wise multiplication

def swiglu_fn(a: Tensor, b: Tensor): return glu(silu(a), b) # SwiGLU activation function

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.linear1 = Linear(d_model, d_ff, **factory_kwargs)  # W1
        self.linear2 = Linear(d_ff, d_model, **factory_kwargs)  # W2
        self.linear3 = Linear(d_model, d_ff, **factory_kwargs)  # W3

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        w1x = self.linear1(x)
        w3x = self.linear3(x)
        h = swiglu_fn(w1x, w3x)
        return self.linear2(h)
    

class RotaryPositionalEmbedding(nn.Module):
    """
    Args：
    theta : float
        Base used to generate inverse frequencies (e.g. 10_000).
    d_k : int
        Dimension of the key / query vectors (must be even).
    max_seq_len : int
        Maximum sequence length expected at inference / training time.
    device : torch.device | None
        Where to place the pre-computed sine / cosine tables.
    """
    def __init__(self,
                 theta: float,
                 d_k: int,
                 max_seq_len: int,
                 device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RoPE.")
        self.d_k = d_k
        # ---- pre-compute inverse frequencies ----
        # freq[k] = 1 / theta ** (2k / d_k)          (k = 0,1,…,d_k/2-1)
        freq = 1.0 / (theta ** (torch.arange(0,d_k,2, device=device).float() / d_k))

        # shape: (max_seq_len, d_k // 2)
        positions = torch.arange(max_seq_len, device=device).float()
        freqs = torch.outer(positions, freq)

        # cache cos/sin; no gradients needed → persistent=False
        self.register_buffer('cos_cached', torch.cos(freqs),persistent=False) # persistent=False does not save to state_dict
        self.register_buffer('sin_cached', torch.sin(freqs), persistent=False)
    
    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"]
        ) -> Float[Tensor, "... seq_len d_k"]:
        """
        Apply RoPE to `x`.  Works with any batch shape prefix.
        """
        # Check if the last dimension matches d_k
        if x.size(-1) != self.d_k:
            raise ValueError(f"Last dim of x ({x.size(-1)}) ≠ d_k ({self.d_k}).")
        
        # Gather the cached tables for the required positions
        cos_pos = self.cos_cached[token_positions]
        sin_pos = self.sin_cached[token_positions]

        # Split even / odd channels
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        # Apply the 2-D rotation to each pair
        out_even = x_even * cos_pos - x_odd * sin_pos
        out_odd = x_even * sin_pos + x_odd * cos_pos

        # Re-interleave
        out = torch.empty_like(x)
        out[..., ::2] = out_even
        out[..., 1::2] = out_odd
        return out
    
def softmax_stable(x: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable softmax."""
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.scale = 1.0 / math.sqrt(d_k)

    def forward(
        self,
        query: Float[Tensor, "... seq_len_q d_k"],
        key: Float[Tensor, "... seq_len_k d_k"],
        value: Float[Tensor, "... seq_len_k d_v"],
        mask: Bool[Tensor, "seq_len_q seq_len_k"] = None
    ) -> Float[Tensor, "... seq_len_q d_v"]:
        # Compute scaled dot product attention scores using einsum
        attn_scores = einsum(query, key, "... q d, ... k d -> ... q k") * self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_probs = softmax_stable(attn_scores, dim=-1)

        # Compute attention output using einsum again
        output = einsum(attn_probs, value, "... q k, ... k d -> ... q d")

        return output
    
    
class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        use_rope: bool = True,
        device=None,
        dtype=None
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k  # match d_k for simplicity
        self.use_rope = use_rope

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = [Linear(d_model, d_model, **factory_kwargs)
                                                              for _ in range(4)]
        self.attn = ScaledDotProductAttention(self.d_k)

        # Create a causal mask for the attention mechanism
        # Shape: (1, 1, max_seq_len, max_seq_len)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device))
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0), persistent=False)

        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                theta=rope_theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)

    def forward(
        self,
        x: Float[Tensor, "batch seq_len d_model"],
        token_positions: Int[Tensor, "batch seq_len"]| None = None,
    ) -> Float[Tensor, "batch seq_len d_model"]:
        B, S, _ = x.shape

        # Project to multi-head Q, K, V
        q,k,v = [rearrange(proj(x), "b s (h d) -> b h s d", h=self.num_heads) 
                 for proj in [self.q_proj, self.k_proj, self.v_proj]]

        # Apply RoPE to Q and K if enabled
        if self.use_rope: q,k = self.rope(q, token_positions),self.rope(k, token_positions)

        # Compute attention
        out = self.attn(q, k, v, mask=self.causal_mask[..., :S, :S])

        # Merge heads and project
        out = rearrange(out, "b h s d -> b s (h d)")
        return self.o_proj(out)
    
    
class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with two sub-layers:

       x ──► RMSNorm ──► MHA ──► + ──►
         │                     ▲
         └─────────────────────┘     (sublayer-1)

       y ──► RMSNorm ──► FF  ──► + ──► out
         │                     ▲
         └─────────────────────┘     (sublayer-2)
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float = 10_000.0,
        use_rope: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        kwargs = {"device": device, "dtype": dtype}

        # sub-layer 1: (RMSNorm → causal MHA)
        self.norm1 = RMSNorm(d_model, **kwargs)
        self.attn  = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            use_rope=use_rope,
            **kwargs,
        )

        #sub-layer 2: (RMSNorm → feed-forward) 
        self.norm2 = RMSNorm(d_model, **kwargs)
        self.ff    = SwiGLU(d_model=d_model, d_ff=d_ff, **kwargs)

    
    def forward(
        self,
        x: torch.Tensor,               # (batch, seq_len, d_model)
        token_positions: torch.Tensor | None = None,  # (batch, seq_len)
    ) -> torch.Tensor:
        b, s, _ = x.shape

        # sub-layer-1: RMSNorm → MHA → residual 
        attn_out = self.attn(self.norm1(x), token_positions=token_positions)
        x = x + attn_out                       # residual connection

        # sub-layer-2: RMSNorm → FF → residual 
        ff_out   = self.ff(self.norm2(x))
        x        = x + ff_out                  # residual connection
        return x
    
    
def _copy_param(target: torch.Tensor, source: torch.Tensor) -> None:
    """
    Copy `source` into `target` in-place, transposing `source` if that
    is what makes the shapes line up.
    """
    if source.shape == target.shape:
        target.data.copy_(source)
    elif source.T.shape == target.shape:
        target.data.copy_(source.T)
    else:
        raise ValueError(f"Shape mismatch: cannot load parameter of shape {source.shape} "
                         f"into tensor of shape {target.shape}")
    

class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float,
                 device=None,
                 dtype=None):
        super().__init__()
        kw = dict(device=device, dtype=dtype)

        # token embedding  (no separate pos-emb: RoPE lives inside blocks)
        self.tok_emb = Embedding(vocab_size, d_model, **kw)

        # L Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                rope_theta=rope_theta,
                use_rope=True,
                **kw,
            )
            for _ in range(num_layers)
        ])

        # final norm
        self.ln_final = RMSNorm(d_model, **kw)
        self.lm_head = Linear(d_model, vocab_size, **kw)

        self.context_length = context_length

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        b, s = token_ids.shape
        if s > self.context_length:
            raise ValueError(f"seq_len {s} exceeds context_length {self.context_length}")

        # token embeddings
        x = self.tok_emb(token_ids)                         # (b, s, d)

        # token positions for RoPE
        pos = torch.arange(s, device=token_ids.device).unsqueeze(0).expand(b, s)

        # transformer stack
        for blk in self.blocks:
            x = blk(x, token_positions=pos)                # (b, s, d)

        # final norm → tied linear projection (logits)
        x = self.ln_final(x)                                 # (b, s, d)

        logits = self.lm_head(x)  # (b, s, vocab_size)
        return logits