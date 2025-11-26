"""Tests for discrete interaction diffusion module."""

import pytest
import torch
import torch.nn as nn

from boltz.model.modules.discrete_interaction_diffusion import (
    DiscreteInteractionDiffusion,
    InteractionDiffusionModule,
    create_discrete_interaction_diffusion,
)


class TestInteractionDiffusionModule:
    """Test InteractionDiffusionModule."""

    @pytest.fixture
    def model_args(self):
        """Get model arguments for testing."""
        return {
            "token_s": 64,
            "interaction_dim": 32,
            "token_transformer_depth": 2,
            "token_transformer_heads": 4,
            "sigma_data": 1.0,
            "dim_fourier": 64,
            "conditioning_transition_layers": 1,
            "activation_checkpointing": False,
            "transformer_post_ln": False,
        }

    @pytest.fixture
    def model(self, model_args):
        """Create model for testing."""
        return InteractionDiffusionModule(**model_args)

    @pytest.fixture
    def sample_data(self):
        """Create sample input data."""
        batch = 2
        n_tokens = 8
        interaction_dim = 32
        token_s = 64

        I_noisy = torch.randn(batch, n_tokens, n_tokens, interaction_dim)
        s_trunk = torch.randn(batch, n_tokens, token_s)
        times = torch.rand(batch)
        mask = torch.ones(batch, n_tokens, n_tokens)

        return {
            "I_noisy": I_noisy,
            "s_trunk": s_trunk,
            "times": times,
            "mask": mask,
        }

    def test_model_initialization(self, model_args):
        """Test that model initializes correctly."""
        model = InteractionDiffusionModule(**model_args)
        assert isinstance(model, nn.Module)
        assert model.interaction_dim == model_args["interaction_dim"]
        assert model.sigma_data == model_args["sigma_data"]

    def test_forward_pass(self, model, sample_data):
        """Test forward pass produces correct output shape."""
        I_update = model(
            sample_data["I_noisy"],
            sample_data["s_trunk"],
            sample_data["times"],
            sample_data["mask"],
        )

        # Check output shape matches input shape
        assert I_update.shape == sample_data["I_noisy"].shape

    def test_forward_without_mask(self, model, sample_data):
        """Test forward pass works without mask."""
        I_update = model(
            sample_data["I_noisy"],
            sample_data["s_trunk"],
            sample_data["times"],
            mask=None,
        )

        assert I_update.shape == sample_data["I_noisy"].shape

    def test_fourier_encoding(self, model, sample_data):
        """Test Fourier time encoding."""
        times = sample_data["times"]
        fourier_feats = model.fourier_encode_time(times)

        # Check output shape
        assert fourier_feats.shape == (times.shape[0], model.fourier_dim)

    def test_gradient_flow(self, model, sample_data):
        """Test that gradients flow through the model."""
        model.train()
        
        I_update = model(
            sample_data["I_noisy"],
            sample_data["s_trunk"],
            sample_data["times"],
            sample_data["mask"],
        )

        # Compute dummy loss and backprop
        loss = I_update.sum()
        loss.backward()

        # Check that some parameters have gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients found in model parameters"

    def test_different_batch_sizes(self, model, model_args):
        """Test model works with different batch sizes."""
        for batch_size in [1, 2, 4]:
            n_tokens = 8
            I_noisy = torch.randn(
                batch_size, n_tokens, n_tokens, model_args["interaction_dim"]
            )
            s_trunk = torch.randn(batch_size, n_tokens, model_args["token_s"])
            times = torch.rand(batch_size)

            I_update = model(I_noisy, s_trunk, times)
            assert I_update.shape == I_noisy.shape


class TestDiscreteInteractionDiffusion:
    """Test DiscreteInteractionDiffusion."""

    @pytest.fixture
    def diffusion_args(self):
        """Get diffusion model arguments."""
        score_model_args = {
            "token_s": 64,
            "interaction_dim": 32,
            "token_transformer_depth": 2,
            "token_transformer_heads": 4,
            "sigma_data": 1.0,
            "dim_fourier": 64,
        }
        
        return {
            "score_model_args": score_model_args,
            "mask_token_id": 0,
            "num_sampling_steps": 5,
            "mask_ratio_min": 0.15,
            "mask_ratio_max": 0.70,
            "sigma_data": 1.0,
            "temperature": 1.0,
            "compile_score": False,
        }

    @pytest.fixture
    def diffusion_model(self, diffusion_args):
        """Create diffusion model for testing."""
        return DiscreteInteractionDiffusion(**diffusion_args)

    @pytest.fixture
    def sample_data(self, diffusion_args):
        """Create sample data for diffusion testing."""
        batch = 2
        n_tokens = 8
        interaction_dim = diffusion_args["score_model_args"]["interaction_dim"]
        token_s = diffusion_args["score_model_args"]["token_s"]

        I_true = torch.randn(batch, n_tokens, n_tokens, interaction_dim)
        s_trunk = torch.randn(batch, n_tokens, token_s)
        valid_mask = torch.ones(batch, n_tokens, n_tokens)

        return {
            "I_true": I_true,
            "s_trunk": s_trunk,
            "valid_mask": valid_mask,
            "batch": batch,
            "n_tokens": n_tokens,
            "interaction_dim": interaction_dim,
        }

    def test_diffusion_initialization(self, diffusion_args):
        """Test diffusion model initializes correctly."""
        model = DiscreteInteractionDiffusion(**diffusion_args)
        assert isinstance(model, nn.Module)
        assert model.num_sampling_steps == diffusion_args["num_sampling_steps"]
        assert model.mask_ratio_min == diffusion_args["mask_ratio_min"]
        assert model.mask_ratio_max == diffusion_args["mask_ratio_max"]

    def test_create_random_mask(self, diffusion_model, sample_data):
        """Test random mask creation."""
        shape = (
            sample_data["batch"],
            sample_data["n_tokens"],
            sample_data["n_tokens"],
        )
        mask_ratio = 0.5

        mask = diffusion_model.create_random_mask(shape, mask_ratio)

        # Check shape
        assert mask.shape == shape

        # Check mask ratio is approximately correct
        actual_ratio = mask.mean().item()
        assert 0.3 < actual_ratio < 0.7, f"Mask ratio {actual_ratio} not close to {mask_ratio}"

    def test_create_random_mask_with_valid_mask(self, diffusion_model, sample_data):
        """Test random mask creation with valid mask."""
        shape = (
            sample_data["batch"],
            sample_data["n_tokens"],
            sample_data["n_tokens"],
        )
        mask_ratio = 0.5
        
        # Create valid mask that only allows upper triangle
        valid_mask = torch.triu(torch.ones(shape))

        mask = diffusion_model.create_random_mask(shape, mask_ratio, valid_mask)

        # Check that mask respects valid_mask
        assert (mask * (1 - valid_mask)).sum() == 0, "Mask extends outside valid positions"

    def test_mask_interactions(self, diffusion_model, sample_data):
        """Test masking of interaction embeddings."""
        I = sample_data["I_true"]
        mask = torch.zeros_like(I[..., 0])
        mask[0, 0, 0] = 1.0  # Mask one position

        I_masked = diffusion_model.mask_interactions(I, mask)

        # Check that masked position is zero
        assert torch.allclose(I_masked[0, 0, 0], torch.zeros_like(I_masked[0, 0, 0]))
        
        # Check that unmasked positions are unchanged
        assert torch.allclose(I_masked[0, 1, 1], I[0, 1, 1])

    def test_forward_pass(self, diffusion_model, sample_data):
        """Test forward pass during training."""
        out_dict = diffusion_model(
            sample_data["I_true"],
            sample_data["s_trunk"],
            sample_data["valid_mask"],
        )

        # Check output dictionary
        assert "I_pred" in out_dict
        assert "I_true" in out_dict
        assert "mask" in out_dict
        assert "mask_ratio" in out_dict

        # Check shapes
        assert out_dict["I_pred"].shape == sample_data["I_true"].shape
        assert out_dict["mask"].shape == sample_data["I_true"].shape[:3]

    def test_compute_loss(self, diffusion_model, sample_data):
        """Test loss computation."""
        out_dict = diffusion_model(
            sample_data["I_true"],
            sample_data["s_trunk"],
            sample_data["valid_mask"],
        )

        loss_dict = diffusion_model.compute_loss(out_dict, sample_data["valid_mask"])

        # Check loss dictionary
        assert "loss" in loss_dict
        assert "loss_breakdown" in loss_dict
        assert "mse_loss" in loss_dict["loss_breakdown"]

        # Check loss is a scalar
        assert loss_dict["loss"].dim() == 0

        # Check loss is non-negative
        assert loss_dict["loss"].item() >= 0

    def test_gradient_flow_through_loss(self, diffusion_model, sample_data):
        """Test gradients flow through forward and loss."""
        diffusion_model.train()

        out_dict = diffusion_model(
            sample_data["I_true"],
            sample_data["s_trunk"],
            sample_data["valid_mask"],
        )

        loss_dict = diffusion_model.compute_loss(out_dict, sample_data["valid_mask"])
        loss = loss_dict["loss"]

        # Backprop
        loss.backward()

        # Check gradients exist
        has_grad = False
        for param in diffusion_model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients found after backprop"

    def test_sampling(self, diffusion_model, sample_data):
        """Test sampling process."""
        interaction_shape = (
            sample_data["batch"],
            sample_data["n_tokens"],
            sample_data["n_tokens"],
            sample_data["interaction_dim"],
        )

        I_sampled = diffusion_model.sample(
            sample_data["s_trunk"],
            interaction_shape,
            num_sampling_steps=3,
            valid_mask=sample_data["valid_mask"],
        )

        # Check output shape
        assert I_sampled.shape == interaction_shape

    def test_sampling_with_initial(self, diffusion_model, sample_data):
        """Test sampling with initial interaction embeddings."""
        interaction_shape = (
            sample_data["batch"],
            sample_data["n_tokens"],
            sample_data["n_tokens"],
            sample_data["interaction_dim"],
        )

        initial_I = torch.randn(interaction_shape)

        I_sampled = diffusion_model.sample(
            sample_data["s_trunk"],
            interaction_shape,
            num_sampling_steps=3,
            initial_I=initial_I,
        )

        # Check output shape
        assert I_sampled.shape == interaction_shape

    def test_mask_schedule(self, diffusion_model):
        """Test masking schedule."""
        mask_ratios = diffusion_model.get_mask_schedule(num_steps=10)

        # Check length
        assert len(mask_ratios) == 10

        # Check decreasing order
        for i in range(len(mask_ratios) - 1):
            assert mask_ratios[i] >= mask_ratios[i + 1]

        # Check bounds
        assert mask_ratios[0] <= diffusion_model.mask_ratio_max
        assert mask_ratios[-1] >= diffusion_model.mask_ratio_min


class TestFactoryFunction:
    """Test factory function."""

    def test_create_discrete_interaction_diffusion(self):
        """Test factory function creates model correctly."""
        model = create_discrete_interaction_diffusion(
            token_s=64,
            interaction_dim=32,
            token_transformer_depth=2,
            token_transformer_heads=4,
            num_sampling_steps=5,
        )

        assert isinstance(model, DiscreteInteractionDiffusion)
        assert model.interaction_dim == 32
        assert model.num_sampling_steps == 5


class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.fixture
    def setup(self):
        """Setup for integration tests."""
        model = create_discrete_interaction_diffusion(
            token_s=64,
            interaction_dim=32,
            token_transformer_depth=2,
            token_transformer_heads=4,
            num_sampling_steps=5,
        )

        batch = 2
        n_tokens = 8
        interaction_dim = 32
        token_s = 64

        I_true = torch.randn(batch, n_tokens, n_tokens, interaction_dim)
        s_trunk = torch.randn(batch, n_tokens, token_s)
        valid_mask = torch.ones(batch, n_tokens, n_tokens)

        return {
            "model": model,
            "I_true": I_true,
            "s_trunk": s_trunk,
            "valid_mask": valid_mask,
            "interaction_shape": (batch, n_tokens, n_tokens, interaction_dim),
        }

    def test_train_and_sample(self, setup):
        """Test full training and sampling pipeline."""
        model = setup["model"]
        model.train()

        # Training step
        out_dict = model(
            setup["I_true"],
            setup["s_trunk"],
            setup["valid_mask"],
        )

        loss_dict = model.compute_loss(out_dict, setup["valid_mask"])
        loss = loss_dict["loss"]

        # Backward
        loss.backward()

        # Check loss is reasonable
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        # Sampling step
        model.eval()
        with torch.no_grad():
            I_sampled = model.sample(
                setup["s_trunk"],
                setup["interaction_shape"],
                num_sampling_steps=3,
                valid_mask=setup["valid_mask"],
            )

        # Check sample is reasonable
        assert not torch.isnan(I_sampled).any()
        assert not torch.isinf(I_sampled).any()

    def test_multiple_training_steps(self, setup):
        """Test multiple training iterations."""
        model = setup["model"]
        model.train()

        losses = []
        for _ in range(3):
            out_dict = model(
                setup["I_true"],
                setup["s_trunk"],
                setup["valid_mask"],
            )

            loss_dict = model.compute_loss(out_dict, setup["valid_mask"])
            loss = loss_dict["loss"]
            losses.append(loss.item())

            # Zero gradients for next iteration
            model.zero_grad()

        # Check all losses are valid
        assert all(not (torch.isnan(torch.tensor(l)) or torch.isinf(torch.tensor(l))) for l in losses)

    def test_different_mask_ratios(self, setup):
        """Test with different mask ratios."""
        model = setup["model"]
        
        for mask_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
            model.mask_ratio_min = mask_ratio - 0.05
            model.mask_ratio_max = mask_ratio + 0.05

            out_dict = model(
                setup["I_true"],
                setup["s_trunk"],
                setup["valid_mask"],
            )

            loss_dict = model.compute_loss(out_dict, setup["valid_mask"])
            assert not torch.isnan(loss_dict["loss"])


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])

