# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang

from __future__ import annotations

from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import nn
from torch.nn import Module

import boltz.model.layers.initialize as init
from boltz.data import const
from boltz.model.loss.diffusionv2 import (
    smooth_lddt_loss,
    weighted_rigid_align,
)
from boltz.model.modules.encodersv2 import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    SingleConditioning,
)
from boltz.model.modules.transformersv2 import (
    DiffusionTransformer,
)
from boltz.model.modules.utils import (
    LinearNoBias,
    center_random_augmentation,
    compute_random_augmentation,
    default,
    log,
)
from boltz.model.potentials.potentials import get_potentials

# Optional support for interaction tracking during diffusion
# Note: We use simple distance-based geometric approximation
# Independent of plif_validity or external constants
try:
    import prolif as plf
    PROLIF_AVAILABLE = True
except ImportError:
    PROLIF_AVAILABLE = False


class DiffusionModule(Module):
    """Diffusion module"""

    def __init__(
        self,
        token_s: int,
        atom_s: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        sigma_data: int = 16,
        dim_fourier: int = 256,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        token_transformer_depth: int = 24,
        token_transformer_heads: int = 8,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        conditioning_transition_layers: int = 2,
        activation_checkpointing: bool = False,
        transformer_post_ln: bool = False,
    ) -> None:
        super().__init__()

        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.sigma_data = sigma_data
        self.activation_checkpointing = activation_checkpointing

        # conditioning
        self.single_conditioner = SingleConditioning(
            sigma_data=sigma_data,
            token_s=token_s,
            dim_fourier=dim_fourier,
            num_transitions=conditioning_transition_layers,
        )

        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=atom_s,
            token_s=token_s,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            structure_prediction=True,
            activation_checkpointing=activation_checkpointing,
            transformer_post_layer_norm=transformer_post_ln,
        )

        self.s_to_a_linear = nn.Sequential(
            nn.LayerNorm(2 * token_s), LinearNoBias(2 * token_s, 2 * token_s)
        )
        init.final_init_(self.s_to_a_linear[1].weight)

        self.token_transformer = DiffusionTransformer(
            dim=2 * token_s,
            dim_single_cond=2 * token_s,
            depth=token_transformer_depth,
            heads=token_transformer_heads,
            activation_checkpointing=activation_checkpointing,
            # post_layer_norm=transformer_post_ln,
        )

        self.a_norm = nn.LayerNorm(
            2 * token_s
        )  # if not transformer_post_ln else nn.Identity()

        self.atom_attention_decoder = AtomAttentionDecoder(
            atom_s=atom_s,
            token_s=token_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            atom_decoder_depth=atom_decoder_depth,
            atom_decoder_heads=atom_decoder_heads,
            activation_checkpointing=activation_checkpointing,
            # transformer_post_layer_norm=transformer_post_ln,
        )

    def forward(
        self,
        s_inputs,  # Float['b n ts']
        s_trunk,  # Float['b n ts']
        r_noisy,  # Float['bm m 3']
        times,  # Float['bm 1 1']
        feats,
        diffusion_conditioning,
        multiplicity=1,
    ):
        if self.activation_checkpointing and self.training:
            s, normed_fourier = torch.utils.checkpoint.checkpoint(
                self.single_conditioner,
                times,
                s_trunk.repeat_interleave(multiplicity, 0),
                s_inputs.repeat_interleave(multiplicity, 0),
            )
        else:
            s, normed_fourier = self.single_conditioner(
                times,
                s_trunk.repeat_interleave(multiplicity, 0),
                s_inputs.repeat_interleave(multiplicity, 0),
            )

        # Sequence-local Atom Attention and aggregation to coarse-grained tokens
        a, q_skip, c_skip, to_keys = self.atom_attention_encoder(
            feats=feats,
            q=diffusion_conditioning["q"].float(),
            c=diffusion_conditioning["c"].float(),
            atom_enc_bias=diffusion_conditioning["atom_enc_bias"].float(),
            to_keys=diffusion_conditioning["to_keys"],
            r=r_noisy,  # Float['b m 3'],
            multiplicity=multiplicity,
        )

        # Full self-attention on token level
        a = a + self.s_to_a_linear(s)

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        a = self.token_transformer(
            a,
            mask=mask.float(),
            s=s,
            bias=diffusion_conditioning[
                "token_trans_bias"
            ].float(),  # note z is not expanded with multiplicity until after bias is computed
            multiplicity=multiplicity,
        )
        a = self.a_norm(a)

        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        r_update = self.atom_attention_decoder(
            a=a,
            q=q_skip,
            c=c_skip,
            atom_dec_bias=diffusion_conditioning["atom_dec_bias"].float(),
            feats=feats,
            multiplicity=multiplicity,
            to_keys=to_keys,
        )

        return r_update


class AtomDiffusion(Module):
    def __init__(
        self,
        score_model_args,
        num_sampling_steps: int = 5,  # number of sampling steps
        sigma_min: float = 0.0004,  # min noise level
        sigma_max: float = 160.0,  # max noise level
        sigma_data: float = 16.0,  # standard deviation of data distribution
        rho: float = 7,  # controls the sampling schedule
        P_mean: float = -1.2,  # mean of log-normal distribution from which noise is drawn for training
        P_std: float = 1.5,  # standard deviation of log-normal distribution from which noise is drawn for training
        gamma_0: float = 0.8,
        gamma_min: float = 1.0,
        noise_scale: float = 1.003,
        step_scale: float = 1.5,
        step_scale_random: list = None,
        coordinate_augmentation: bool = True,
        coordinate_augmentation_inference=None,
        compile_score: bool = False,
        alignment_reverse_diff: bool = False,
        synchronize_sigmas: bool = False,
    ):
        super().__init__()
        self.score_model = DiffusionModule(
            **score_model_args,
        )
        if compile_score:
            self.score_model = torch.compile(
                self.score_model, dynamic=False, fullgraph=False
            )

        # parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sampling_steps = num_sampling_steps
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.step_scale_random = step_scale_random
        self.coordinate_augmentation = coordinate_augmentation
        self.coordinate_augmentation_inference = (
            coordinate_augmentation_inference
            if coordinate_augmentation_inference is not None
            else coordinate_augmentation
        )
        self.alignment_reverse_diff = alignment_reverse_diff
        self.synchronize_sigmas = synchronize_sigmas

        self.token_s = score_model_args["token_s"]
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)
        
        # Interaction tracking (optional)
        # Uses distance-based approximation during diffusion
        self.enable_prolif_tracking = False
        self.prolif_interactions = []
        
        # Distance cutoffs for interaction classification (in Angstroms)
        # Based on standard structural biology literature
        self.interaction_cutoffs = {
            'close_contact': 4.0,      # VdW contact distance
            'hydrophobic': 5.0,        # Hydrophobic interaction limit
            'hbond_max': 3.7,          # H-bond maximum distance
            'hbond_min': 2.5,          # H-bond minimum distance
            'pistacking_max': 5.5,     # Pi-stacking maximum
            'pistacking_min': 3.5,     # Pi-stacking minimum
            'ionic': 5.0,              # Ionic interaction limit
        }

    @property
    def device(self):
        return next(self.score_model.parameters()).device

    def c_skip(self, sigma):
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma):
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        return log(sigma / self.sigma_data) * 0.25

    def preconditioned_network_forward(
        self,
        noised_atom_coords,  #: Float['b m 3'],
        sigma,  #: Float['b'] | Float[' '] | float,
        network_condition_kwargs: dict,
    ):
        batch, device = noised_atom_coords.shape[0], noised_atom_coords.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = rearrange(sigma, "b -> b 1 1")

        r_update = self.score_model(
            r_noisy=self.c_in(padded_sigma) * noised_atom_coords,
            times=self.c_noise(sigma),
            **network_condition_kwargs,
        )

        denoised_coords = (
            self.c_skip(padded_sigma) * noised_atom_coords
            + self.c_out(padded_sigma) * r_update
        )
        return denoised_coords

    def sample_schedule(self, num_sampling_steps=None):
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        inv_rho = 1 / self.rho

        steps = torch.arange(
            num_sampling_steps, device=self.device, dtype=torch.float32
        )
        sigmas = (
            self.sigma_max**inv_rho
            + steps
            / (num_sampling_steps - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = sigmas * self.sigma_data

        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
        return sigmas

    def sample(
        self,
        atom_mask,
        num_sampling_steps=None,
        multiplicity=1,
        max_parallel_samples=None,
        steering_args=None,
        **network_condition_kwargs,
    ):
        if steering_args is not None and (
            steering_args["fk_steering"]
            or steering_args["physical_guidance_update"]
            or steering_args["contact_guidance_update"]
        ):
            potentials = get_potentials(steering_args, boltz2=True)

        if steering_args["fk_steering"]:
            multiplicity = multiplicity * steering_args["num_particles"]
            energy_traj = torch.empty((multiplicity, 0), device=self.device)
            resample_weights = torch.ones(multiplicity, device=self.device).reshape(
                -1, steering_args["num_particles"]
            )
        if (
            steering_args["physical_guidance_update"]
            or steering_args["contact_guidance_update"]
        ):
            scaled_guidance_update = torch.zeros(
                (multiplicity, *atom_mask.shape[1:], 3),
                dtype=torch.float32,
                device=self.device,
            )
        if max_parallel_samples is None:
            max_parallel_samples = multiplicity

        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        shape = (*atom_mask.shape, 3)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sampling_steps)
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))
        if self.training and self.step_scale_random is not None:
            step_scale = np.random.choice(self.step_scale_random)
        else:
            step_scale = self.step_scale

        # atom position is noise at the beginning
        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(shape, device=self.device)
        token_repr = None
        atom_coords_denoised = None

        # gradually denoise
        for step_idx, (sigma_tm, sigma_t, gamma) in enumerate(sigmas_and_gammas):
            random_R, random_tr = compute_random_augmentation(
                multiplicity, device=atom_coords.device, dtype=atom_coords.dtype
            )
            atom_coords = atom_coords - atom_coords.mean(dim=-2, keepdims=True)
            atom_coords = (
                torch.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr
            )
            if atom_coords_denoised is not None:
                atom_coords_denoised -= atom_coords_denoised.mean(dim=-2, keepdims=True)
                atom_coords_denoised = (
                    torch.einsum("bmd,bds->bms", atom_coords_denoised, random_R)
                    + random_tr
                )
            if (
                steering_args["physical_guidance_update"]
                or steering_args["contact_guidance_update"]
            ) and scaled_guidance_update is not None:
                scaled_guidance_update = torch.einsum(
                    "bmd,bds->bms", scaled_guidance_update, random_R
                )

            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

            t_hat = sigma_tm * (1 + gamma)
            steering_t = 1.0 - (step_idx / num_sampling_steps)
            noise_var = self.noise_scale**2 * (t_hat**2 - sigma_tm**2)
            eps = sqrt(noise_var) * torch.randn(shape, device=self.device)
            atom_coords_noisy = atom_coords + eps

            with torch.no_grad():
                atom_coords_denoised = torch.zeros_like(atom_coords_noisy)
                sample_ids = torch.arange(multiplicity).to(atom_coords_noisy.device)
                sample_ids_chunks = sample_ids.chunk(
                    multiplicity % max_parallel_samples + 1
                )

                for sample_ids_chunk in sample_ids_chunks:
                    atom_coords_denoised_chunk = self.preconditioned_network_forward(
                        atom_coords_noisy[sample_ids_chunk],
                        t_hat,
                        network_condition_kwargs=dict(
                            multiplicity=sample_ids_chunk.numel(),
                            **network_condition_kwargs,
                        ),
                    )
                    atom_coords_denoised[sample_ids_chunk] = atom_coords_denoised_chunk

                if steering_args["fk_steering"] and (
                    (
                        step_idx % steering_args["fk_resampling_interval"] == 0
                        and noise_var > 0
                    )
                    or step_idx == num_sampling_steps - 1
                ):
                    # Compute energy of x_0 prediction
                    energy = torch.zeros(multiplicity, device=self.device)
                    for potential in potentials:
                        parameters = potential.compute_parameters(steering_t)
                        if parameters["resampling_weight"] > 0:
                            component_energy = potential.compute(
                                atom_coords_denoised,
                                network_condition_kwargs["feats"],
                                parameters,
                            )
                            energy += parameters["resampling_weight"] * component_energy
                    energy_traj = torch.cat((energy_traj, energy.unsqueeze(1)), dim=1)

                    # Compute log G values
                    if step_idx == 0:
                        log_G = -1 * energy
                    else:
                        log_G = energy_traj[:, -2] - energy_traj[:, -1]

                    # Compute ll difference between guided and unguided transition distribution
                    if (
                        steering_args["physical_guidance_update"]
                        or steering_args["contact_guidance_update"]
                    ) and noise_var > 0:
                        ll_difference = (
                            eps**2 - (eps + scaled_guidance_update) ** 2
                        ).sum(dim=(-1, -2)) / (2 * noise_var)
                    else:
                        ll_difference = torch.zeros_like(energy)

                    # Compute resampling weights
                    resample_weights = F.softmax(
                        (ll_difference + steering_args["fk_lambda"] * log_G).reshape(
                            -1, steering_args["num_particles"]
                        ),
                        dim=1,
                    )

                # Compute guidance update to x_0 prediction
                if (
                    steering_args["physical_guidance_update"]
                    or steering_args["contact_guidance_update"]
                ) and step_idx < num_sampling_steps - 1:
                    guidance_update = torch.zeros_like(atom_coords_denoised)
                    for guidance_step in range(steering_args["num_gd_steps"]):
                        energy_gradient = torch.zeros_like(atom_coords_denoised)
                        for potential in potentials:
                            parameters = potential.compute_parameters(steering_t)
                            if (
                                parameters["guidance_weight"] > 0
                                and (guidance_step) % parameters["guidance_interval"]
                                == 0
                            ):
                                energy_gradient += parameters[
                                    "guidance_weight"
                                ] * potential.compute_gradient(
                                    atom_coords_denoised + guidance_update,
                                    network_condition_kwargs["feats"],
                                    parameters,
                                )
                        guidance_update -= energy_gradient
                    atom_coords_denoised += guidance_update
                    scaled_guidance_update = (
                        guidance_update
                        * -1
                        * self.step_scale
                        * (sigma_t - t_hat)
                        / t_hat
                    )

                if steering_args["fk_steering"] and (
                    (
                        step_idx % steering_args["fk_resampling_interval"] == 0
                        and noise_var > 0
                    )
                    or step_idx == num_sampling_steps - 1
                ):
                    resample_indices = (
                        torch.multinomial(
                            resample_weights,
                            resample_weights.shape[1]
                            if step_idx < num_sampling_steps - 1
                            else 1,
                            replacement=True,
                        )
                        + resample_weights.shape[1]
                        * torch.arange(
                            resample_weights.shape[0], device=resample_weights.device
                        ).unsqueeze(-1)
                    ).flatten()

                    atom_coords = atom_coords[resample_indices]
                    atom_coords_noisy = atom_coords_noisy[resample_indices]
                    atom_mask = atom_mask[resample_indices]
                    if atom_coords_denoised is not None:
                        atom_coords_denoised = atom_coords_denoised[resample_indices]
                    energy_traj = energy_traj[resample_indices]
                    if (
                        steering_args["physical_guidance_update"]
                        or steering_args["contact_guidance_update"]
                    ):
                        scaled_guidance_update = scaled_guidance_update[
                            resample_indices
                        ]
                    if token_repr is not None:
                        token_repr = token_repr[resample_indices]

            if self.alignment_reverse_diff:
                with torch.autocast("cuda", enabled=False):
                    atom_coords_noisy = weighted_rigid_align(
                        atom_coords_noisy.float(),
                        atom_coords_denoised.float(),
                        atom_mask.float(),
                        atom_mask.float(),
                    )

                atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

            denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
            atom_coords_next = (
                atom_coords_noisy + step_scale * (sigma_t - t_hat) * denoised_over_sigma
            )

            atom_coords = atom_coords_next
            
            # Compute ProLIF interactions if enabled (every 25 steps)
            if (
                self.enable_prolif_tracking 
                and not self.training
                and PROLIF_AVAILABLE
            ):
                # Track every 25 steps, plus first and last step
                should_track = (
                    step_idx == 0 or  # First step
                    step_idx == num_sampling_steps - 1 or  # Last step
                    step_idx % 25 == 0  # Every 25 steps
                )
                
                if should_track:
                    progress_pct = int(100 * step_idx / max(num_sampling_steps - 1, 1))
                    print(f"\n[Diffusion {progress_pct}%] Computing interactions at step {step_idx}/{num_sampling_steps-1}")
                    self._annotate_interactions(
                        atom_coords_denoised,
                        network_condition_kwargs['feats'],
                        step_idx,
                        f"diffusion_step_{step_idx}"
                    )

        return dict(sample_atom_coords=atom_coords, diff_token_repr=token_repr)

    def loss_weight(self, sigma):
        return (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)

    def noise_distribution(self, batch_size):
        return (
            self.sigma_data
            * (
                self.P_mean
                + self.P_std * torch.randn((batch_size,), device=self.device)
            ).exp()
        )

    def forward(
        self,
        s_inputs,
        s_trunk,
        feats,
        diffusion_conditioning,
        multiplicity=1,
    ):
        # training diffusion step
        batch_size = feats["coords"].shape[0] // multiplicity

        if self.synchronize_sigmas:
            sigmas = self.noise_distribution(batch_size).repeat_interleave(
                multiplicity, 0
            )
        else:
            sigmas = self.noise_distribution(batch_size * multiplicity)
        padded_sigmas = rearrange(sigmas, "b -> b 1 1")

        atom_coords = feats["coords"]

        atom_mask = feats["atom_pad_mask"]
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        atom_coords = center_random_augmentation(
            atom_coords, atom_mask, augmentation=self.coordinate_augmentation
        )

        noise = torch.randn_like(atom_coords)
        noised_atom_coords = atom_coords + padded_sigmas * noise

        denoised_atom_coords = self.preconditioned_network_forward(
            noised_atom_coords,
            sigmas,
            network_condition_kwargs={
                "s_inputs": s_inputs,
                "s_trunk": s_trunk,
                "feats": feats,
                "multiplicity": multiplicity,
                "diffusion_conditioning": diffusion_conditioning,
            },
        )

        return {
            "denoised_atom_coords": denoised_atom_coords,
            "sigmas": sigmas,
            "aligned_true_atom_coords": atom_coords,
        }

    def compute_loss(
        self,
        feats,
        out_dict,
        add_smooth_lddt_loss=True,
        nucleotide_loss_weight=5.0,
        ligand_loss_weight=10.0,
        multiplicity=1,
        filter_by_plddt=0.0,
    ):
        with torch.autocast("cuda", enabled=False):
            denoised_atom_coords = out_dict["denoised_atom_coords"].float()
            sigmas = out_dict["sigmas"].float()

            resolved_atom_mask_uni = feats["atom_resolved_mask"].float()

            if filter_by_plddt > 0:
                plddt_mask = feats["plddt"] > filter_by_plddt
                resolved_atom_mask_uni = resolved_atom_mask_uni * plddt_mask.float()

            resolved_atom_mask = resolved_atom_mask_uni.repeat_interleave(
                multiplicity, 0
            )

            align_weights = denoised_atom_coords.new_ones(denoised_atom_coords.shape[:2])
            atom_type = (
                torch.bmm(
                    feats["atom_to_token"].float(),
                    feats["mol_type"].unsqueeze(-1).float(),
                )
                .squeeze(-1)
                .long()
            )
            atom_type_mult = atom_type.repeat_interleave(multiplicity, 0)

            align_weights = (
                align_weights
                * (
                    1
                    + nucleotide_loss_weight
                    * (
                        torch.eq(atom_type_mult, const.chain_type_ids["DNA"]).float()
                        + torch.eq(atom_type_mult, const.chain_type_ids["RNA"]).float()
                    )
                    + ligand_loss_weight
                    * torch.eq(
                        atom_type_mult, const.chain_type_ids["NONPOLYMER"]
                    ).float()
                ).float()
            )

            atom_coords = out_dict["aligned_true_atom_coords"].float()
            atom_coords_aligned_ground_truth = weighted_rigid_align(
                atom_coords.detach(),
                denoised_atom_coords.detach(),
                align_weights.detach(),
                mask=feats["atom_resolved_mask"]
                .float()
                .repeat_interleave(multiplicity, 0)
                .detach(),
            )

            # Cast back
            atom_coords_aligned_ground_truth = atom_coords_aligned_ground_truth.to(
                denoised_atom_coords
            )

            # weighted MSE loss of denoised atom positions
            mse_loss = (
                (denoised_atom_coords - atom_coords_aligned_ground_truth) ** 2
            ).sum(dim=-1)
            mse_loss = torch.sum(
                mse_loss * align_weights * resolved_atom_mask, dim=-1
            ) / (torch.sum(3 * align_weights * resolved_atom_mask, dim=-1) + 1e-5)

            # weight by sigma factor
            loss_weights = self.loss_weight(sigmas)
            mse_loss = (mse_loss * loss_weights).mean()

            total_loss = mse_loss

            # proposed auxiliary smooth lddt loss
            lddt_loss = self.zero
            if add_smooth_lddt_loss:
                lddt_loss = smooth_lddt_loss(
                    denoised_atom_coords,
                    feats["coords"],
                    torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
                    + torch.eq(atom_type, const.chain_type_ids["RNA"]).float(),
                    coords_mask=resolved_atom_mask_uni,
                    multiplicity=multiplicity,
                )

                total_loss = total_loss + lddt_loss

            loss_breakdown = {
                "mse_loss": mse_loss,
                "smooth_lddt_loss": lddt_loss,
            }

        return {"loss": total_loss, "loss_breakdown": loss_breakdown}
    
    def _annotate_interactions(
        self,
        coords: torch.Tensor,
        feats: dict,
        step_idx: int,
        label: str = "step"
    ):
        """
        Compute protein-ligand interactions from coordinates during diffusion.
        Uses fast distance-based approximation with configurable cutoffs.
        
        Note: This is NOT full ProLIF analysis - it's a fast geometric approximation
        suitable for tracking interaction convergence during diffusion.
        
        Args:
            coords: Atom coordinates tensor [batch, n_atoms, 3]
            feats: Feature dictionary from model
            step_idx: Current step index
            label: Label for this computation
        """
        if not PROLIF_AVAILABLE:
            return
        
        try:
            import time
            start_time = time.time()
            # Convert to numpy and take first sample if batched
            coords_np = coords.detach().cpu().numpy()
            if len(coords_np.shape) == 3:
                coords_np = coords_np[0]
            
            # Extract protein and ligand masks from features
            protein_mask, ligand_mask = self._extract_protein_ligand_masks(feats)
            
            if protein_mask is None or ligand_mask is None:
                return
            
            protein_coords = coords_np[protein_mask]
            ligand_coords = coords_np[ligand_mask]
            
            if len(protein_coords) == 0 or len(ligand_coords) == 0:
                return
                        
            # 1. Filter to tight pocket (6Å instead of 10Å)
            ligand_center = ligand_coords.mean(axis=0)
            distances = np.linalg.norm(protein_coords - ligand_center, axis=1)
            pocket_mask = distances < 6.0  # Tight pocket for speed
            protein_coords_pocket = protein_coords[pocket_mask]
            
            if len(protein_coords_pocket) == 0:
                print(f"  [{label}]: No protein atoms near ligand, skipping")
                return
            
            # 2. Limit to 50 closest atoms maximum (very aggressive)
            if len(protein_coords_pocket) > 50:
                pocket_distances = distances[pocket_mask]
                closest_indices = np.argsort(pocket_distances)[:50]
                protein_coords_pocket = protein_coords_pocket[closest_indices]
            
            # 3. Limit ligand atoms if too many
            if len(ligand_coords) > 50:
                ligand_coords = ligand_coords[:50]
            
            print(f"  [{label}]: Processing {len(protein_coords_pocket)} protein atoms, "
                  f"{len(ligand_coords)} ligand atoms")
            
            # Use fast distance-based interaction detection
            # ProLIF requires full atom typing and hydrogens which we don't have during diffusion
            # Distance-based approach is fast and suitable for tracking convergence
            from collections import defaultdict
            interaction_counts = defaultdict(int)
            interacting_residues = set()
            
            # Compute pairwise distances
            prot_expanded = protein_coords_pocket[:, np.newaxis, :]  # (n_prot, 1, 3)
            lig_expanded = ligand_coords[np.newaxis, :, :]  # (1, n_lig, 3)
            distances = np.linalg.norm(prot_expanded - lig_expanded, axis=2)
            
            # For each protein atom, find closest ligand atom
            min_distances = distances.min(axis=1)  # (n_prot,)
            
            # Classify interactions by distance using configurable cutoffs
            cutoffs = self.interaction_cutoffs
            
            for i, min_dist in enumerate(min_distances):
                res_id = f"RES{i+1}"
                
                # Close contact (VdW)
                if min_dist < cutoffs['close_contact']:
                    interaction_counts['CloseContact'] += 1
                    interacting_residues.add(res_id)
                
                # Hydrophobic interactions
                if cutoffs['close_contact'] <= min_dist < cutoffs['hydrophobic']:
                    interaction_counts['Hydrophobic'] += 1
                    interacting_residues.add(res_id)
                
                # H-bond potential (without atom typing, approximate based on distance)
                if cutoffs['hbond_min'] <= min_dist <= cutoffs['hbond_max']:
                    interaction_counts['HBond'] += 1
                    interacting_residues.add(res_id)
                
                # Pi-stacking potential (aromatic rings)
                if cutoffs['pistacking_min'] <= min_dist <= cutoffs['pistacking_max']:
                    interaction_counts['PiStack'] += 1
                    interacting_residues.add(res_id)
                
                # Ionic interactions (charged residues)
                if min_dist < cutoffs['ionic']:
                    interaction_counts['Ionic'] += 1
                    interacting_residues.add(res_id)
            
            # Store results including coordinates
            result = {
                'label': label,
                'step_idx': step_idx,
                'total_interactions': sum(interaction_counts.values()),
                'n_interacting_residues': len(interacting_residues),
                'interaction_counts': dict(interaction_counts),
                'interacting_residues': sorted(list(interacting_residues)),
                'coordinates': {
                    'protein_pocket': protein_coords_pocket.tolist(),
                    'ligand': ligand_coords.tolist(),
                },
                'n_protein_atoms': len(protein_coords_pocket),
                'n_ligand_atoms': len(ligand_coords),
            }
            
            self.prolif_interactions.append(result)
            
            elapsed = time.time() - start_time
            # Print summary with coordinate info
            print(f"  [{label}]: {result['total_interactions']} interactions, "
                  f"{result['n_interacting_residues']} residues, "
                  f"coords saved: {len(protein_coords_pocket)}+{len(ligand_coords)} atoms ({elapsed:.2f}s)")
            
        except Exception as e:
            print(f"  [{label}]: ProLIF failed: {str(e)[:100]}")
    
    def _extract_protein_ligand_masks(self, feats: dict):
        """
        Extract protein and ligand atom masks from feature dictionary.
        
        Args:
            feats: Feature dictionary
            
        Returns:
            Tuple of (protein_mask, ligand_mask) as numpy arrays
        """
        try:
            # Get molecule types
            if 'mol_type' not in feats or 'atom_to_token' not in feats:
                return None, None
            
            mol_type = feats['mol_type']
            atom_to_token = feats['atom_to_token']
            
            # Convert to numpy
            if torch.is_tensor(mol_type):
                mol_type = mol_type.cpu().numpy()
            if torch.is_tensor(atom_to_token):
                atom_to_token = atom_to_token.cpu().numpy()
            
            # Handle batch dimension
            if len(mol_type.shape) > 1:
                mol_type = mol_type[0]
            if len(atom_to_token.shape) == 3:
                atom_to_token = atom_to_token[0]
            
            # Map each atom to its molecule type
            # atom_to_token: [n_atoms, n_tokens]
            # mol_type: [n_tokens]
            atom_mol_types = atom_to_token @ mol_type.reshape(-1, 1)
            atom_mol_types = atom_mol_types.flatten()
            
            # Chain type IDs: 0=protein, 1=RNA, 2=DNA, 3=NONPOLYMER
            protein_mask = atom_mol_types <= 2  # protein/RNA/DNA
            ligand_mask = atom_mol_types == 3   # NONPOLYMER (ligand)
            
            return protein_mask, ligand_mask
            
        except Exception as e:
            print(f"  Error extracting masks: {e}")
            return None, None
    
    def reset_prolif_tracking(self):
        """Reset ProLIF tracking data."""
        self.prolif_interactions = []
    
    @property
    def prolif_interaction_types(self):
        """Get list of tracked interaction types (for compatibility)."""
        return ['CloseContact', 'Hydrophobic', 'HBond', 'PiStack', 'Ionic']
    
    def get_prolif_summary(self):
        """
        Get summary of tracked interactions across all diffusion steps.
        
        Returns:
            Dictionary with summary statistics including:
            - n_steps: Number of tracked steps
            - interactions: List of interaction data per step
            - convergence: Metrics showing how interactions change
            - stability: First step where interactions stabilize
        """
        if not self.prolif_interactions:
            return {'n_steps': 0, 'interactions': []}
        
        summary = {
            'n_steps': len(self.prolif_interactions),
            'interactions': self.prolif_interactions,
        }
        
        # Compute convergence if we have multiple steps
        if len(self.prolif_interactions) >= 2:
            first = self.prolif_interactions[0]
            last = self.prolif_interactions[-1]
            
            first_residues = set(first.get('interacting_residues', []))
            last_residues = set(last.get('interacting_residues', []))
            
            summary['convergence'] = {
                'initial_interactions': first.get('total_interactions', 0),
                'final_interactions': last.get('total_interactions', 0),
                'change': last.get('total_interactions', 0) - first.get('total_interactions', 0),
                'persistent_residues': len(first_residues & last_residues),
                'gained_residues': len(last_residues - first_residues),
                'lost_residues': len(first_residues - last_residues),
            }
        
        # Detect stability: find first step where interactions don't change for 10 consecutive steps
        stability = self._detect_interaction_stability(window_size=3)
        if stability:
            summary['stability'] = stability
        
        return summary
    
    def _detect_interaction_stability(self, window_size=10):
        """
        Detect the first step where key interactions remain stable for a given window.
        Only checks HBond and PiStack as these are the most specific interactions.
        
        Args:
            window_size: Number of consecutive steps that must be unchanged
            
        Returns:
            Dictionary with stability information or None if not stable
        """
        if len(self.prolif_interactions) < window_size + 1:
            return None
        
        # Key interaction types to check for stability
        key_interactions = ['HBond', 'PiStack']
        
        # Check each potential starting point
        for i in range(len(self.prolif_interactions) - window_size):
            # Get key interaction counts at step i
            reference = self.prolif_interactions[i]
            ref_counts = reference.get('interaction_counts', {})
            ref_key_counts = {k: ref_counts.get(k, 0) for k in key_interactions}
            
            # Check if next window_size steps have same key interactions
            is_stable = True
            for j in range(1, window_size + 1):
                current = self.prolif_interactions[i + j]
                curr_counts = current.get('interaction_counts', {})
                curr_key_counts = {k: curr_counts.get(k, 0) for k in key_interactions}
                
                # Check if key interaction counts match
                if ref_key_counts != curr_key_counts:
                    is_stable = False
                    break
            
            if is_stable:
                # Found first stable point
                first_stable_step = self.prolif_interactions[i]
                last_checked_step = self.prolif_interactions[i + window_size]
                
                # Get full interaction counts for reporting
                all_counts = first_stable_step.get('interaction_counts', {})
                
                return {
                    'first_stable_step': first_stable_step['step_idx'],
                    'stable_from_label': first_stable_step['label'],
                    'last_checked_step': last_checked_step['step_idx'],
                    'window_size': window_size,
                    'stable_interaction_counts': dict(all_counts),
                    'key_stable_counts': ref_key_counts,  # Only HBond and PiStack
                    'stable_residues': sorted(list(first_stable_step.get('interacting_residues', []))),
                    'n_stable_interactions': sum(all_counts.values()),
                    'n_key_interactions': sum(ref_key_counts.values()),
                    'n_stable_residues': first_stable_step.get('n_interacting_residues', 0),
                    'checked_types': key_interactions,
                    'message': f'Key interactions (HBond, PiStack) stabilized at step {first_stable_step["step_idx"]} '
                              f'and remained unchanged for {window_size} consecutive steps'
                }
        
        return None
