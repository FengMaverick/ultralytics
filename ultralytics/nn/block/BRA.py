import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# =================================================================================================
# Part 1: Internal Helper Functions
# Source: BiFormer/ops/torch/rrsda.py
# =================================================================================================

def _grid2seq(x: torch.Tensor, region_size: Tuple[int, int], num_heads: int):
    """
    [Original] Rearranges grid layout to sequence layout for attention.
    Args:
        x: BCHW tensor
        region_size: (rh, rw)
        num_heads: number of attention heads
    Return:
        out: rearranged x, has a shape of (bs, nhead, nregion, reg_size, head_dim)
        region_h, region_w: number of regions per col/row
    """
    B, C, H, W = x.size()
    region_h, region_w = H // region_size[0], W // region_size[1]
    
    # [Original] Reshape logic
    x = x.view(B, num_heads, C // num_heads, region_h, region_size[0], region_w, region_size[1])
    
    # [Original] Permute to group regions and tokens within regions
    # b: batch, m: head, d: dim/head, h: reg_h, p: reg_sz_h, w: reg_w, q: reg_sz_w
    # Target: b, m, h, w, p, q, d -> flatten h,w to nregion; flatten p,q to reg_size
    x = torch.einsum('bmdhpwq->bmhwpqd', x).flatten(2, 3).flatten(-3, -2)
    return x, region_h, region_w

def _seq2grid(x: torch.Tensor, region_h: int, region_w: int, region_size: Tuple[int, int]):
    """
    [Original] Rearranges sequence layout back to grid layout.
    Args: 
        x: (bs, nhead, nregion, reg_size^2, head_dim)
    Return:
        x: (bs, C, H, W)
    """
    bs, nhead, nregion, reg_size_square, head_dim = x.size()
    
    # [Original] Reshape back: (bs, nhead, region_h, region_w, rh, rw, head_dim)
    x = x.view(bs, nhead, region_h, region_w, region_size[0], region_size[1], head_dim)
    
    # [Original] bmhwpqd -> bmdhpwq
    x = torch.einsum('bmhwpqd->bmdhpwq', x).reshape(bs, nhead * head_dim,
                                                    region_h * region_size[0], 
                                                    region_w * region_size[1])
    return x

def regional_routing_attention_torch(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float,
    region_graph: torch.LongTensor, region_size: Tuple[int, int],
    kv_region_size: Optional[Tuple[int, int]] = None,
    auto_pad=True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    [Original] Regional Routing Scaled Dot-product Attention
    This function performs the core token-to-token attention based on the routing graph.
    """
    kv_region_size = kv_region_size or region_size
    bs, nhead, q_nregion, topk = region_graph.size()
    
    # [Modified] Enhanced Auto-pad logic for robustness
    # The original implementation assumes consistent H, W extraction.
    # We added explicit shape extraction to ensure it works with dynamic input sizes.
    q_pad_b, q_pad_r, kv_pad_b, kv_pad_r = 0, 0, 0, 0
    Hq, Wq = query.shape[2], query.shape[3]
    if auto_pad:
        q_pad_b = (region_size[0] - Hq % region_size[0]) % region_size[0]
        q_pad_r = (region_size[1] - Wq % region_size[1]) % region_size[1]
        if (q_pad_b > 0 or q_pad_r > 0):
            query = F.pad(query, (0, q_pad_r, 0, q_pad_b)) # zero padding

        Hk, Wk = key.shape[2], key.shape[3]
        kv_pad_b = (kv_region_size[0] - Hk % kv_region_size[0]) % kv_region_size[0]
        kv_pad_r = (kv_region_size[1] - Wk % kv_region_size[1]) % kv_region_size[1]
        if (kv_pad_r > 0 or kv_pad_b > 0):
            key = F.pad(key, (0, kv_pad_r, 0, kv_pad_b)) # zero padding
            value = F.pad(value, (0, kv_pad_r, 0, kv_pad_b)) # zero padding
    
    # [Original] Convert to sequence format
    query_seq, q_region_h, q_region_w = _grid2seq(query, region_size=region_size, num_heads=nhead)
    key_seq, _, _ = _grid2seq(key, region_size=kv_region_size, num_heads=nhead)
    value_seq, _, _ = _grid2seq(value, region_size=kv_region_size, num_heads=nhead)

    # [Original] Gather key and values based on region graph
    head_dim = key_seq.size(-1)
    kv_region_size_sq = key_seq.size(-2) # reg_size^2
    
    # [Original] Broadcasting logic for gather
    broadcasted_region_graph = region_graph.view(bs, nhead, q_nregion, topk, 1, 1).expand(-1, -1, -1, -1, kv_region_size_sq, head_dim)
    
    # [Original] Memory-intensive expand + gather (Standard BRA implementation)
    # Note: This is computationally efficient for small topk but memory intensive for large feature maps.
    # We kept it as-is for correctness.
    key_expanded = key_seq.view(bs, nhead, 1, -1, kv_region_size_sq, head_dim).expand(-1, -1, q_nregion, -1, -1, -1)
    value_expanded = value_seq.view(bs, nhead, 1, -1, kv_region_size_sq, head_dim).expand(-1, -1, q_nregion, -1, -1, -1)
    
    key_g = torch.gather(key_expanded, dim=3, index=broadcasted_region_graph) 
    value_g = torch.gather(value_expanded, dim=3, index=broadcasted_region_graph)

    # [Original] Token-to-token attention mechanism
    key_g_flat = key_g.flatten(3, 4)
    value_g_flat = value_g.flatten(3, 4)
    
    attn = (query_seq * scale) @ key_g_flat.transpose(-1, -2)
    attn = torch.softmax(attn, dim=-1)
    output = attn @ value_g_flat
    
    # [Original] Convert back to grid format
    output = _seq2grid(output, region_h=q_region_h, region_w=q_region_w, region_size=region_size)
    
    # [Original] Remove padding
    if auto_pad and (q_pad_b > 0 or q_pad_r > 0):
        output = output[:, :, :Hq, :Wq]
        
    return output, attn


# =================================================================================================
# Part 2: Main Class
# Source: BiFormer/ops/bra_nchw.py
# =================================================================================================

class BiLevelRoutingAttention(nn.Module):
    """
    [Original] Bi-Level Routing Attention (BRA) module.
    CVPR 2023: BiFormer: Vision Transformers with Bi-Level Routing Attention.
    """
    def __init__(self, dim, num_heads=8, n_win=7, qk_scale=None, topk=4, side_dwconv=3, auto_pad=True):
        super().__init__()
        # [Modified] Removed unused arguments and simplified init for YOLO integration
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim must be divisible by num_heads!'
        self.head_dim = self.dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        
        # [Original] Side DWConv (LePE) - Local Context Enhancement
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else nn.Identity()
        
        self.topk = topk
        self.n_win = n_win  # number of windows per row/col (default 7)
        self.auto_pad = auto_pad

        # [Original] Linear Projections
        self.qkv_linear = nn.Conv2d(self.dim, 3*self.dim, kernel_size=1)
        self.output_linear = nn.Conv2d(self.dim, self.dim, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: NCHW tensor
        Return:
            NCHW tensor
        """
        N, C, H, W = x.size()
        
        # [Modified] Dynamic Region Size Calculation
        # Original BRA assumes fixed region size or n_win.
        # Here we dynamically calculate region_size based on input H/W to be robust against different scales.
        # This ensures n_win partitions are roughly maintained.
        region_size = (max(1, H // self.n_win), max(1, W // self.n_win))
        # Note: If H < n_win, region_size becomes 1, effectively degenerating to full attention (which is fine for small maps).
        
        # [Original] STEP 1: Linear projection
        qkv = self.qkv_linear(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # [Original] STEP 2: Region-to-region routing
        # Coarse grained routing using Average Pooling
        q_r = F.avg_pool2d(q.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)
        k_r = F.avg_pool2d(k.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)
        
        # [Original] Flatten for routing calculation
        q_r = q_r.permute(0, 2, 3, 1).flatten(1, 2)
        k_r = k_r.flatten(2, 3) 
        
        # [Original] Adjacency matrix of regional graph
        a_r = q_r @ k_r 
        
        # [Original] Top-k routing
        # We want topk most relevant regions for each query region
        k_fixed = min(self.topk, a_r.size(-1)) # Safety check
        _, idx_r = torch.topk(a_r, k=k_fixed, dim=-1)
        
        # [Original] Expand for multi-head usage (Shared routing across heads)
        idx_r = idx_r.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # [Original] STEP 3: Token-to-token attention
        # Calls the helper function defined above
        output, _ = regional_routing_attention_torch(
            query=q, key=k, value=v, scale=self.scale,
            region_graph=idx_r, region_size=region_size,
            auto_pad=self.auto_pad
        )
        
        # [Original] Add Local Context Enhancement (LePE) and Output Projection
        output = output + self.lepe(v)
        output = self.output_linear(output) 

        return output
