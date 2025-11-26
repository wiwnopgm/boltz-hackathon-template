"""Discrete diffusion for interaction matrix embeddings.

This module implements discrete diffusion for pairwise interaction embeddings I_{ij},
where interactions are masked and denoised in a sequence-based manner similar to BERT-style masking.
"""

from __future__ import annotations

from math import sqrt

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn import Module

import boltz.model.layers.initialize as init
from boltz.model.modules.transformersv2 import DiffusionTransformer
from boltz.model.modules.utils import LinearNoBias, default, log


class InteractionDiffusionModule(Module):
    """Diffusion module for pairwise interaction embeddings.
    
    Processes the interaction matrix I_{ij} through transformer layers
    with conditioning information.
    """

    def __init__(
        self,
        token_s: int,
        interaction_dim: int,
        token_transformer_depth: int = 12,
        token_transformer_heads: int = 8,
        sigma_data: float = 1.0,
        dim_fourier: int = 256,
        conditioning_transition_layers: int = 2,
        activation_checkpointing: bool = False,
        transformer_post_ln: bool = False,
    ) -> None:
        super().__init__()

        self.interaction_dim = interaction_dim
        self.sigma_data = sigma_data
        self.activation_checkpointing = activation_checkpointing

        # Fourier time embedding
        self.fourier_dim = dim_fourier
        self.register_buffer(
            "freqs", 2 * torch.pi * torch.randn(dim_fourier // 2), persistent=False
        )

        # Time conditioning projection
        self.time_mlp = nn.Sequential(
            nn.Linear(dim_fourier, token_s),
            nn.SiLU(),
            nn.Linear(token_s, token_s),
        )

        # Project interaction embeddings to working dimension
        self.interaction_proj_in = nn.Sequential(
            nn.LayerNorm(interaction_dim),
            LinearNoBias(interaction_dim, token_s),
        )

        # Conditioning from single representation
        self.single_conditioner = nn.Sequential(
            nn.LayerNorm(token_s),
            LinearNoBias(token_s, token_s),
        )

        # Main transformer for processing interactions
        # Interactions are flattened from [b, n, n, d] to [b, n*n, d]
        self.interaction_transformer = DiffusionTransformer(
            dim=token_s,
            dim_single_cond=token_s,
            depth=token_transformer_depth,
            heads=token_transformer_heads,
            activation_checkpointing=activation_checkpointing,
        )

        self.interaction_norm = nn.LayerNorm(token_s)

        # Project back to interaction dimension
        self.interaction_proj_out = nn.Sequential(
            nn.LayerNorm(token_s),
            LinearNoBias(token_s, interaction_dim),
        )
        init.final_init_(self.interaction_proj_out[1].weight)

    def fourier_encode_time(self, times):
        """Encode timesteps with Fourier features."""
        times = rearrange(times, "b -> b 1")
        freqs = rearrange(self.freqs, "d -> 1 d")
        fourier_feats = times * freqs
        fourier_feats = torch.cat([fourier_feats.sin(), fourier_feats.cos()], dim=-1)
        return fourier_feats

    def forward(
        self,
        I_noisy,  # Float['b n n d'] - noisy interaction embeddings
        s_trunk,  # Float['b n ts'] - single token representations
        times,  # Float['b'] - noise level times
        mask=None,  # Float['b n n'] - mask for valid interactions
    ):
        """
        Args:
            I_noisy: Noisy interaction embeddings [batch, n_tokens, n_tokens, interaction_dim]
            s_trunk: Single token representations [batch, n_tokens, token_s]
            times: Noise times [batch]
            mask: Optional mask for valid token pairs [batch, n_tokens, n_tokens]
            
        Returns:
            I_update: Predicted interaction embeddings [batch, n_tokens, n_tokens, interaction_dim]
        """
        batch, n_tokens, _, _ = I_noisy.shape
        device = I_noisy.device

        # Fourier encode time and project
        fourier_feats = self.fourier_encode_time(times)
        time_embed = self.time_mlp(fourier_feats)  # [b, token_s]

        # Project interaction embeddings to working dimension
        I_working = self.interaction_proj_in(I_noisy)  # [b, n, n, token_s]

        # Condition single representations with time
        s_cond = self.single_conditioner(s_trunk)  # [b, n, token_s]
        s_cond = s_cond + time_embed.unsqueeze(1)  # Add time to each token

        # Broadcast single conditioning to pairwise
        # Create pairwise conditioning by combining i and j token features
        s_i = s_cond.unsqueeze(2).expand(-1, -1, n_tokens, -1)  # [b, n, n, token_s]
        s_j = s_cond.unsqueeze(1).expand(-1, n_tokens, -1, -1)  # [b, n, n, token_s]
        pairwise_cond = s_i + s_j  # [b, n, n, token_s]

        # Add conditioning to interaction features
        I_working = I_working + pairwise_cond

        # Flatten pairwise interactions for transformer processing
        I_flat = rearrange(I_working, "b n m d -> b (n m) d")  # [b, n*n, token_s]
        
        # Create flat mask if provided
        if mask is not None:
            mask_flat = rearrange(mask, "b n m -> b (n m)")  # [b, n*n]
        else:
            mask_flat = None

        # Process through transformer
        if self.activation_checkpointing and self.training:
            I_transformed = torch.utils.checkpoint.checkpoint(
                self.interaction_transformer,
                I_flat,
                mask_flat.float() if mask_flat is not None else None,
                s_cond,
                None,  # bias
                1,  # multiplicity
            )
        else:
            I_transformed = self.interaction_transformer(
                I_flat,
                mask=mask_flat.float() if mask_flat is not None else None,
                s=s_cond,
                bias=None,
                multiplicity=1,
            )

        I_transformed = self.interaction_norm(I_transformed)

        # Reshape back to pairwise
        I_transformed = rearrange(
            I_transformed, "b (n m) d -> b n m d", n=n_tokens, m=n_tokens
        )

        # Project back to interaction dimension
        I_update = self.interaction_proj_out(I_transformed)

        return I_update


class DiscreteInteractionDiffusion(Module):
    """Discrete diffusion for interaction matrix embeddings.
    
    Implements masked denoising of pairwise interaction embeddings,
    similar to masked language modeling but for interaction matrices.
    """

    def __init__(
        self,
        score_model_args,
        mask_token_id: int = 0,  # ID for mask token
        num_sampling_steps: int = 10,
        mask_ratio_min: float = 0.15,  # Minimum masking ratio during training
        mask_ratio_max: float = 0.70,  # Maximum masking ratio during training
        sigma_data: float = 1.0,
        temperature: float = 1.0,  # Temperature for sampling
        compile_score: bool = False,
    ):
        """
        Args:
            score_model_args: Arguments for InteractionDiffusionModule
            mask_token_id: Token ID representing masked interactions
            num_sampling_steps: Number of denoising steps during inference
            mask_ratio_min: Minimum fraction of interactions to mask during training
            mask_ratio_max: Maximum fraction of interactions to mask during training
            sigma_data: Standard deviation of data distribution
            temperature: Sampling temperature (lower = more confident predictions)
            compile_score: Whether to compile the score model
        """
        super().__init__()
        
        self.score_model = InteractionDiffusionModule(**score_model_args)
        
        if compile_score:
            self.score_model = torch.compile(
                self.score_model, dynamic=False, fullgraph=False
            )

        self.mask_token_id = mask_token_id
        self.num_sampling_steps = num_sampling_steps
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.sigma_data = sigma_data
        self.temperature = temperature
        
        self.interaction_dim = score_model_args["interaction_dim"]
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

    @property
    def device(self):
        return next(self.score_model.parameters()).device

    def get_mask_schedule(self, num_steps=None):
        """Get masking schedule from high to low mask ratio."""
        num_steps = default(num_steps, self.num_sampling_steps)
        # Linear schedule from max to min masking
        mask_ratios = torch.linspace(
            self.mask_ratio_max, self.mask_ratio_min, num_steps, device=self.device
        )
        return mask_ratios

    def create_random_mask(self, shape, mask_ratio, valid_mask=None):
        """Create random mask for interactions.
        
        Args:
            shape: (batch, n_tokens, n_tokens)
            mask_ratio: Fraction of interactions to mask
            valid_mask: Optional mask indicating which positions are valid
            
        Returns:
            Binary mask where 1 = masked, 0 = visible
        """
        batch, n, m = shape
        device = self.device

        # Start with all unmasked
        mask = torch.zeros((batch, n, m), device=device)

        # Apply valid_mask if provided
        if valid_mask is not None:
            maskable_positions = valid_mask
        else:
            maskable_positions = torch.ones((batch, n, m), device=device)

        # For each sample in batch, randomly mask positions
        for b in range(batch):
            valid_positions = maskable_positions[b].bool()
            n_valid = valid_positions.sum().item()
            n_to_mask = int(n_valid * mask_ratio)

            if n_to_mask > 0:
                # Get indices of valid positions
                valid_indices = torch.nonzero(valid_positions, as_tuple=False)
                # Randomly select positions to mask
                perm = torch.randperm(n_valid, device=device)[:n_to_mask]
                mask_indices = valid_indices[perm]
                # Set mask
                mask[b, mask_indices[:, 0], mask_indices[:, 1]] = 1.0

        return mask

    def mask_interactions(self, I, mask):
        """Apply mask to interaction embeddings.
        
        Args:
            I: Interaction embeddings [batch, n, n, d]
            mask: Binary mask [batch, n, n] where 1 = masked
            
        Returns:
            Masked interaction embeddings
        """
        # Zero out masked positions
        I_masked = I * (1 - mask.unsqueeze(-1))
        return I_masked

    def sample(
        self,
        s_trunk,  # Single token representations
        interaction_shape,  # (batch, n_tokens, n_tokens, interaction_dim)
        num_sampling_steps=None,
        valid_mask=None,  # Which interactions are valid
        initial_I=None,  # Optional initial interaction embeddings
    ):
        """Sample interaction embeddings via iterative demasking.
        
        Args:
            s_trunk: Single token representations [batch, n_tokens, token_s]
            interaction_shape: Shape of interaction matrix to generate
            num_sampling_steps: Number of denoising steps
            valid_mask: Mask indicating valid token pairs [batch, n_tokens, n_tokens]
            initial_I: Optional initial interaction embeddings
            
        Returns:
            Generated interaction embeddings [batch, n_tokens, n_tokens, interaction_dim]
        """
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        batch, n_tokens, _, interaction_dim = interaction_shape
        device = self.device

        # Initialize with zeros or provided initial embeddings
        if initial_I is None:
            I = torch.zeros(interaction_shape, device=device)
        else:
            I = initial_I.clone()

        # Get mask schedule (from high to low masking)
        mask_ratios = self.get_mask_schedule(num_sampling_steps)

        # Start fully masked
        current_mask = torch.ones((batch, n_tokens, n_tokens), device=device)
        if valid_mask is not None:
            current_mask = current_mask * valid_mask

        # Iteratively unmask
        for step_idx in range(num_sampling_steps):
            target_mask_ratio = mask_ratios[step_idx].item()

            # Apply current mask
            I_masked = self.mask_interactions(I, current_mask)

            # Predict unmasked interactions
            with torch.no_grad():
                # Noise level decreases over time
                times = torch.full(
                    (batch,), step_idx / num_sampling_steps, device=device
                )
                I_pred = self.score_model(
                    I_masked, s_trunk, times, mask=valid_mask
                )

            # Update masked positions with predictions
            I = torch.where(
                current_mask.unsqueeze(-1).bool(),
                I_pred,
                I
            )

            # Update mask for next iteration (gradually unmask)
            if step_idx < num_sampling_steps - 1:
                next_mask_ratio = mask_ratios[step_idx + 1].item()
                current_mask = self.create_random_mask(
                    (batch, n_tokens, n_tokens), next_mask_ratio, valid_mask
                )

        return I

    def forward(
        self,
        I_true,  # Ground truth interaction embeddings
        s_trunk,  # Single token representations
        valid_mask=None,  # Which interactions are valid
    ):
        """Training forward pass with random masking.
        
        Args:
            I_true: Ground truth interaction embeddings [batch, n, n, interaction_dim]
            s_trunk: Single token representations [batch, n, token_s]
            valid_mask: Optional mask for valid interactions [batch, n, n]
            
        Returns:
            Dictionary with predictions and losses
        """
        batch, n_tokens, _, interaction_dim = I_true.shape
        device = I_true.device

        # Sample random mask ratio for this batch
        mask_ratio = (
            self.mask_ratio_min
            + (self.mask_ratio_max - self.mask_ratio_min)
            * torch.rand(1, device=device).item()
        )

        # Create random mask
        mask = self.create_random_mask(
            (batch, n_tokens, n_tokens), mask_ratio, valid_mask
        )

        # Mask interactions
        I_masked = self.mask_interactions(I_true, mask)

        # Create time embedding (random during training)
        times = torch.rand((batch,), device=device)

        # Predict interactions
        I_pred = self.score_model(I_masked, s_trunk, times, mask=valid_mask)

        return {
            "I_pred": I_pred,
            "I_true": I_true,
            "mask": mask,
            "mask_ratio": mask_ratio,
        }

    def compute_loss(
        self,
        out_dict,
        valid_mask=None,
        reduction="mean",
    ):
        """Compute reconstruction loss on masked positions.
        
        Args:
            out_dict: Output from forward pass
            valid_mask: Optional mask for valid interactions
            reduction: Loss reduction method
            
        Returns:
            Dictionary with loss and breakdown
        """
        I_pred = out_dict["I_pred"]
        I_true = out_dict["I_true"]
        mask = out_dict["mask"]

        # Compute MSE loss only on masked positions
        mse_loss = F.mse_loss(I_pred, I_true, reduction="none")  # [b, n, n, d]
        mse_loss = mse_loss.mean(dim=-1)  # Average over embedding dim [b, n, n]

        # Apply mask - only compute loss on masked positions
        masked_loss = mse_loss * mask

        # Apply valid_mask if provided
        if valid_mask is not None:
            masked_loss = masked_loss * valid_mask

        # Compute mean loss
        n_masked = mask.sum() + 1e-8
        if valid_mask is not None:
            n_masked = (mask * valid_mask).sum() + 1e-8

        if reduction == "mean":
            total_loss = masked_loss.sum() / n_masked
        elif reduction == "sum":
            total_loss = masked_loss.sum()
        else:
            total_loss = masked_loss

        loss_breakdown = {
            "mse_loss": total_loss,
            "mask_ratio": out_dict["mask_ratio"],
        }

        return {"loss": total_loss, "loss_breakdown": loss_breakdown}


def create_discrete_interaction_diffusion(
    token_s: int = 384,
    interaction_dim: int = 128,
    token_transformer_depth: int = 12,
    token_transformer_heads: int = 8,
    **kwargs,
):
    """Factory function to create discrete interaction diffusion model."""
    score_model_args = {
        "token_s": token_s,
        "interaction_dim": interaction_dim,
        "token_transformer_depth": token_transformer_depth,
        "token_transformer_heads": token_transformer_heads,
    }
    
    return DiscreteInteractionDiffusion(
        score_model_args=score_model_args,
        **kwargs,
    )

