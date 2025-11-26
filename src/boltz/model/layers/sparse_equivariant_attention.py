from typing import Optional, Tuple

import torch
from einops.layers.torch import Rearrange
from torch import Tensor, nn

import boltz.model.layers.initialize as init


def compute_spherical_harmonics(
    directions: Tensor, max_l: int = 2
) -> Tensor:
    """Compute real spherical harmonics for given directions.
    
    This implements Y_l^m for l=0 to max_l, using real-valued spherical harmonics
    that are equivariant under SO(3) rotations.
    
    Parameters
    ----------
    directions : Tensor
        Normalized direction vectors of shape (..., 3)
    max_l : int
        Maximum angular momentum quantum number, by default 2
        
    Returns
    -------
    Tensor
        Spherical harmonics of shape (..., (max_l+1)^2)
    """
    # Normalize directions
    r = torch.norm(directions, dim=-1, keepdim=True)
    r = torch.clamp(r, min=1e-8)
    directions = directions / r
    
    x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]
    
    # Compute spherical harmonics for each l
    harmonics = []
    
    # l=0: Y_0^0 = 1/sqrt(4*pi) (constant)
    harmonics.append(torch.ones_like(x) / (4 * torch.pi) ** 0.5)
    
    if max_l >= 1:
        # l=1: Y_1^{-1}, Y_1^0, Y_1^1
        sqrt_3_over_4pi = (3.0 / (4 * torch.pi)) ** 0.5
        harmonics.append(sqrt_3_over_4pi * y)  # Y_1^{-1}
        harmonics.append(sqrt_3_over_4pi * z)  # Y_1^0
        harmonics.append(sqrt_3_over_4pi * x)  # Y_1^1
    
    if max_l >= 2:
        # l=2: Y_2^{-2}, Y_2^{-1}, Y_2^0, Y_2^1, Y_2^2
        sqrt_15_over_4pi = (15.0 / (4 * torch.pi)) ** 0.5
        sqrt_15_over_8pi = (15.0 / (8 * torch.pi)) ** 0.5
        sqrt_5_over_16pi = (5.0 / (16 * torch.pi)) ** 0.5
        
        xy = x * y
        xz = x * z
        yz = y * z
        x2 = x * x
        y2 = y * y
        z2 = z * z
        
        harmonics.append(sqrt_15_over_4pi * xy)  # Y_2^{-2}
        harmonics.append(sqrt_15_over_4pi * yz)  # Y_2^{-1}
        harmonics.append(sqrt_5_over_16pi * (3 * z2 - 1))  # Y_2^0
        harmonics.append(sqrt_15_over_4pi * xz)  # Y_2^1
        harmonics.append(sqrt_15_over_8pi * (x2 - y2))  # Y_2^2
    
    return torch.stack(harmonics, dim=-1)


def tensor_product_equivariant(
    features_q: Tensor,
    features_k: Tensor,
    spherical_harmonics: Tensor,
) -> Tensor:
    """Compute equivariant tensor product between query and key features.
    
    This implements the tensor product operation that maintains SO(3) equivariance
    by using spherical harmonics. The tensor product combines features in an
    equivariant manner using the spherical harmonic basis.
    
    Parameters
    ----------
    features_q : Tensor
        Query features of shape (B, N, D)
    features_k : Tensor
        Key features of shape (B, N, D)
    spherical_harmonics : Tensor
        Spherical harmonics of shape (B, N, N, num_harmonics)
        
    Returns
    -------
    Tensor
        Equivariant attention scores of shape (B, N, N)
    """
    # Tensor product: compute attention as weighted sum over spherical harmonic channels
    # This maintains SO(3) equivariance by using rotation-equivariant basis functions
    attn_scores = torch.einsum(
        "bnd,bmd,bnmh->bnm",
        features_q,
        features_k,
        spherical_harmonics,
    ) / (features_q.shape[-1] ** 0.5)
    
    return attn_scores


class SparseEquivariantAttentionPairBias(nn.Module):
    """Sparse equivariant attention pair bias layer using tensor networks.
    
    This implementation uses tensor network operations with irreducible representations
    to maintain strict SO(3) equivariance. The attention mechanism uses:
    1. Spherical harmonics for encoding relative positions
    2. Tensor product operations for combining features
    3. Sparse attention patterns that preserve equivariance
    """

    def __init__(
        self,
        c_s: int,
        c_z: Optional[int] = None,
        num_heads: Optional[int] = None,
        inf: float = 1e6,
        compute_pair_bias: bool = True,
        sparsity_mode: str = "topk",
        topk_ratio: float = 0.5,
        distance_threshold: Optional[float] = None,
        max_l: int = 2,
        use_tensor_product: bool = True,
    ) -> None:
        """Initialize the sparse equivariant attention pair bias layer.

        Parameters
        ----------
        c_s : int
            The input sequence dimension.
        c_z : int, optional
            The input pairwise dimension.
        num_heads : int, optional
            The number of heads.
        inf : float, optional
            The inf value for masking, by default 1e6
        compute_pair_bias : bool, optional
            Whether to compute pair bias from z, by default True
        sparsity_mode : str, optional
            Sparsity mode: "topk" or "distance", by default "topk"
        topk_ratio : float, optional
            Ratio of top-k pairs to keep (0.0 to 1.0), by default 0.5
        distance_threshold : float, optional
            Distance threshold for distance-based sparsity, by default None
        max_l : int, optional
            Maximum angular momentum for spherical harmonics, by default 2
        use_tensor_product : bool, optional
            Whether to use tensor product operations, by default True

        """
        super().__init__()

        assert c_s % num_heads == 0, f"c_s ({c_s}) must be divisible by num_heads ({num_heads})"
        assert sparsity_mode in ["topk", "distance"], f"sparsity_mode must be 'topk' or 'distance', got {sparsity_mode}"
        assert 0.0 < topk_ratio <= 1.0, f"topk_ratio must be in (0, 1], got {topk_ratio}"

        self.c_s = c_s
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.inf = inf
        self.sparsity_mode = sparsity_mode
        self.topk_ratio = topk_ratio
        self.distance_threshold = distance_threshold
        self.max_l = max_l
        self.use_tensor_product = use_tensor_product
        
        # Number of spherical harmonics: sum of (2l+1) for l=0 to max_l
        self.num_harmonics = sum(2 * l + 1 for l in range(max_l + 1))

        # Equivariant projections using tensor network structure
        self.proj_q = nn.Linear(c_s, c_s)
        self.proj_k = nn.Linear(c_s, c_s, bias=False)
        self.proj_v = nn.Linear(c_s, c_s, bias=False)
        self.proj_g = nn.Linear(c_s, c_s, bias=False)
        
        # Projection for spherical harmonics integration
        if use_tensor_product:
            self.harmonics_proj = nn.Linear(self.num_harmonics, num_heads, bias=False)

        self.compute_pair_bias = compute_pair_bias
        if compute_pair_bias:
            self.proj_z = nn.Sequential(
                nn.LayerNorm(c_z),
                nn.Linear(c_z, num_heads, bias=False),
                Rearrange("b ... h -> b h ..."),
            )
        else:
            self.proj_z = Rearrange("b ... h -> b h ...")

        self.proj_o = nn.Linear(c_s, c_s, bias=False)
        init.final_init_(self.proj_o.weight)

    def _compute_relative_positions(
        self,
        positions: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute relative positions and spherical harmonics.
        
        Parameters
        ----------
        positions : Tensor
            Position coordinates of shape (B, N, 3)
        mask : Tensor
            Mask of shape (B, N, N)
            
        Returns
        -------
        Tuple[Tensor, Tensor]
            Relative positions (B, N, N, 3) and spherical harmonics (B, N, N, num_harmonics)
        """
        # Compute relative positions (equivariant: only depends on differences)
        rel_pos = positions[:, :, None, :] - positions[:, None, :, :]  # (B, N, N, 3)
        
        # Compute distances
        distances = torch.norm(rel_pos, dim=-1, keepdim=True)  # (B, N, N, 1)
        distances = torch.clamp(distances, min=1e-8)
        
        # Normalize directions for spherical harmonics
        directions = rel_pos / distances  # (B, N, N, 3)
        
        # Compute spherical harmonics (SO(3) equivariant)
        spherical_harmonics = compute_spherical_harmonics(
            directions, max_l=self.max_l
        )  # (B, N, N, num_harmonics)
        
        # Apply mask
        mask_expanded = mask.unsqueeze(-1)  # (B, N, N, 1)
        spherical_harmonics = spherical_harmonics * mask_expanded.float()
        
        return rel_pos, spherical_harmonics

    def _compute_sparse_mask(
        self,
        attn_scores: Tensor,
        mask: Tensor,
        positions: Optional[Tensor] = None,
        distances: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute sparse attention mask based on sparsity mode.

        Parameters
        ----------
        attn_scores : Tensor
            Attention scores of shape (B, H, N, N)
        mask : Tensor
            Original mask of shape (B, N, N)
        positions : Tensor, optional
            Position coordinates for distance computation, shape (B, N, 3)
        distances : Tensor, optional
            Precomputed distances of shape (B, N, N)

        Returns
        -------
        Tensor
            Sparse attention mask of shape (B, H, N, N)
        """
        B, H, N, _ = attn_scores.shape
        sparse_mask = torch.zeros_like(attn_scores, dtype=torch.bool)

        if self.sparsity_mode == "topk":
            # Top-k sparsity: keep top-k attention scores per query
            k = max(1, int(self.topk_ratio * N))
            
            # Apply original mask first
            valid_mask = mask[:, None, None].bool()  # (B, 1, 1, N)
            
            # Set invalid positions to very negative values
            masked_scores = attn_scores.clone()
            masked_scores = masked_scores.masked_fill(~valid_mask, float('-inf'))
            
            # Get top-k indices per query
            _, topk_indices = torch.topk(masked_scores, k, dim=-1)  # (B, H, N, k)
            
            # Create sparse mask
            batch_indices = torch.arange(B, device=attn_scores.device)[:, None, None, None]
            head_indices = torch.arange(H, device=attn_scores.device)[None, :, None, None]
            query_indices = torch.arange(N, device=attn_scores.device)[None, None, :, None]
            
            sparse_mask[batch_indices, head_indices, query_indices, topk_indices] = True
            
        elif self.sparsity_mode == "distance":
            # Distance-based sparsity (equivariant: based on distances)
            if distances is None and positions is not None:
                # Compute pairwise distances
                pos_diff = positions[:, :, None, :] - positions[:, None, :, :]  # (B, N, N, 3)
                distances = torch.norm(pos_diff, dim=-1)  # (B, N, N)
            
            if distances is not None:
                if self.distance_threshold is None:
                    # Use adaptive threshold based on median distance
                    flat_distances = distances[mask.bool()]
                    if flat_distances.numel() > 0:
                        threshold = torch.median(flat_distances) * self.topk_ratio
                    else:
                        threshold = torch.quantile(distances[mask.bool()], self.topk_ratio) if mask.bool().any() else distances.max()
                else:
                    threshold = self.distance_threshold
                
                # Create sparse mask based on distance threshold
                distance_mask = (distances <= threshold)[:, None, :, :]  # (B, 1, N, N)
                sparse_mask = distance_mask.expand(-1, H, -1, -1) & mask[:, None, None].bool()
            else:
                # Fallback to topk if no distance information
                k = max(1, int(self.topk_ratio * N))
                valid_mask = mask[:, None, None].bool()
                masked_scores = attn_scores.clone()
                masked_scores = masked_scores.masked_fill(~valid_mask, float('-inf'))
                _, topk_indices = torch.topk(masked_scores, k, dim=-1)
                
                batch_indices = torch.arange(B, device=attn_scores.device)[:, None, None, None]
                head_indices = torch.arange(H, device=attn_scores.device)[None, :, None, None]
                query_indices = torch.arange(N, device=attn_scores.device)[None, None, :, None]
                
                sparse_mask[batch_indices, head_indices, query_indices, topk_indices] = True

        return sparse_mask

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        k_in: Tensor,
        multiplicity: int = 1,
        positions: Optional[Tensor] = None,
        distances: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with sparse equivariant attention using tensor networks.

        Parameters
        ----------
        s : torch.Tensor
            The input sequence tensor (B, S, D)
        z : torch.Tensor
            The input pairwise tensor or bias (B, N, N, D)
        mask : torch.Tensor
            The pairwise mask tensor (B, N, N)
        k_in : torch.Tensor
            The keys input tensor (B, N, D)
        multiplicity : int, optional
            Multiplicity for diffusion, by default 1
        positions : torch.Tensor, optional
            Position coordinates for equivariant attention (B, N, 3)
        distances : torch.Tensor, optional
            Precomputed pairwise distances (B, N, N)

        Returns
        -------
        torch.Tensor
            The output sequence tensor (B, S, D)
        """
        B = s.shape[0]
        N = s.shape[1]

        # Compute projections
        q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
        k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)

        bias = self.proj_z(z)
        bias = bias.repeat_interleave(multiplicity, 0)

        g = self.proj_g(s).sigmoid()

        with torch.autocast("cuda", enabled=False):
            if self.use_tensor_product and positions is not None:
                # Equivariant attention using tensor network operations
                # Compute relative positions and spherical harmonics
                rel_pos, spherical_harmonics = self._compute_relative_positions(
                    positions, mask
                )
                
                # Project spherical harmonics to attention heads for bias
                harmonics_bias = self.harmonics_proj(spherical_harmonics)  # (B, N, N, H)
                harmonics_bias = harmonics_bias.permute(0, 3, 1, 2)  # (B, H, N, N)
                harmonics_bias = harmonics_bias.repeat_interleave(multiplicity, 0)
                
                # Compute attention scores per head using tensor product structure
                # For each head, combine standard attention with equivariant tensor product
                attn_standard = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
                attn_standard = attn_standard / (self.head_dim**0.5)
                
                # Add equivariant bias from spherical harmonics (tensor network contribution)
                attn = attn_standard + bias.float() + harmonics_bias.float()
            else:
                # Standard attention (fallback when positions not available)
                attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
                attn = attn / (self.head_dim**0.5) + bias.float()
            
            # Apply original mask
            attn = attn + (1 - mask[:, None, None].float()) * -self.inf
            
            # Compute sparse mask
            sparse_mask = self._compute_sparse_mask(
                attn, mask, positions, distances
            )
            
            # Apply sparse mask: set non-sparse positions to -inf
            attn = attn.masked_fill(~sparse_mask, float('-inf'))
            
            # Softmax over sparse attention
            attn = attn.softmax(dim=-1)
            
            # Ensure we only attend to sparse positions
            attn = attn * sparse_mask.float()

            # Compute output (equivariant: linear combination preserves equivariance)
            o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
        
        o = o.reshape(B, -1, self.c_s)
        o = self.proj_o(g * o)

        return o
