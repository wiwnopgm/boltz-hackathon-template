from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Set, List, Union

import torch
import numpy as np
from boltz.data import const
from boltz.model.potentials.schedules import (
    ParameterSchedule,
    ExponentialInterpolation,
    PiecewiseStepFunction,
)
from boltz.model.loss.diffusionv2 import weighted_rigid_align


class Potential(ABC):
    def __init__(
        self,
        parameters: Optional[
            Dict[str, Union[ParameterSchedule, float, int, bool]]
        ] = None,
    ):
        self.parameters = parameters

    def compute(self, coords, feats, parameters):
        index, args, com_args, ref_args, operator_args = self.compute_args(
            feats, parameters
        )

        if index.shape[1] == 0:
            return torch.zeros(coords.shape[:-2], device=coords.device)

        if com_args is not None:
            com_index, atom_pad_mask = com_args
            unpad_com_index = com_index[atom_pad_mask]
            unpad_coords = coords[..., atom_pad_mask, :]
            coords = torch.zeros(
                (*unpad_coords.shape[:-2], unpad_com_index.max() + 1, 3),
                device=coords.device,
            ).scatter_reduce(
                -2,
                unpad_com_index.unsqueeze(-1).expand_as(unpad_coords),
                unpad_coords,
                "mean",
            )
        else:
            com_index, atom_pad_mask = None, None

        if ref_args is not None:
            ref_coords, ref_mask, ref_atom_index, ref_token_index = ref_args
            coords = coords[..., ref_atom_index, :]
        else:
            ref_coords, ref_mask, ref_atom_index, ref_token_index = (
                None,
                None,
                None,
                None,
            )

        if operator_args is not None:
            negation_mask, union_index = operator_args
        else:
            negation_mask, union_index = None, None

        value = self.compute_variable(
            coords,
            index,
            ref_coords=ref_coords,
            ref_mask=ref_mask,
            compute_gradient=False,
        )
        energy = self.compute_function(
            value, *args, negation_mask=negation_mask, compute_derivative=False
        )

        if union_index is not None:
            neg_exp_energy = torch.exp(-1 * parameters["union_lambda"] * energy)
            Z = torch.zeros(
                (*energy.shape[:-1], union_index.max() + 1), device=union_index.device
            ).scatter_reduce(
                -1,
                union_index.expand_as(neg_exp_energy),
                neg_exp_energy,
                "sum",
            )
            softmax_energy = neg_exp_energy / Z[..., union_index]
            softmax_energy[Z[..., union_index] == 0] = 0
            return (energy * softmax_energy).sum(dim=-1)

        return energy.sum(dim=tuple(range(1, energy.dim())))

    def compute_gradient(self, coords, feats, parameters):
        index, args, com_args, ref_args, operator_args = self.compute_args(
            feats, parameters
        )
        if index.shape[1] == 0:
            return torch.zeros_like(coords)

        if com_args is not None:
            com_index, atom_pad_mask = com_args
            unpad_coords = coords[..., atom_pad_mask, :]
            unpad_com_index = com_index[atom_pad_mask]
            coords = torch.zeros(
                (*unpad_coords.shape[:-2], unpad_com_index.max() + 1, 3),
                device=coords.device,
            ).scatter_reduce(
                -2,
                unpad_com_index.unsqueeze(-1).expand_as(unpad_coords),
                unpad_coords,
                "mean",
            )
            com_counts = torch.bincount(com_index[atom_pad_mask])
        else:
            com_index, atom_pad_mask = None, None

        if ref_args is not None:
            ref_coords, ref_mask, ref_atom_index, ref_token_index = ref_args
            coords = coords[..., ref_atom_index, :]
        else:
            ref_coords, ref_mask, ref_atom_index, ref_token_index = (
                None,
                None,
                None,
                None,
            )

        if operator_args is not None:
            negation_mask, union_index = operator_args
        else:
            negation_mask, union_index = None, None

        value, grad_value = self.compute_variable(
            coords,
            index,
            ref_coords=ref_coords,
            ref_mask=ref_mask,
            compute_gradient=True,
        )
        energy, dEnergy = self.compute_function(
            value, 
            *args, negation_mask=negation_mask, compute_derivative=True
        )
        if union_index is not None:
            neg_exp_energy = torch.exp(-1 * parameters["union_lambda"] * energy)
            Z = torch.zeros(
                (*energy.shape[:-1], union_index.max() + 1), device=union_index.device
            ).scatter_reduce(
                -1,
                union_index.expand_as(energy),
                neg_exp_energy,
                "sum",
            )
            softmax_energy = neg_exp_energy / Z[..., union_index]
            softmax_energy[Z[..., union_index] == 0] = 0
            f = torch.zeros(
                (*energy.shape[:-1], union_index.max() + 1), device=union_index.device
            ).scatter_reduce(
                -1,
                union_index.expand_as(energy),
                energy * softmax_energy,
                "sum",
            )
            dSoftmax = (
                dEnergy
                * softmax_energy
                * (1 + parameters["union_lambda"] * (energy - f[..., union_index]))
            )
            prod = dSoftmax.tile(grad_value.shape[-3]).unsqueeze(
                -1
            ) * grad_value.flatten(start_dim=-3, end_dim=-2)
            if prod.dim() > 3:
                prod = prod.sum(dim=list(range(1, prod.dim() - 2)))
            grad_atom = torch.zeros_like(coords).scatter_reduce(
                -2,
                index.flatten(start_dim=0, end_dim=1)
                .unsqueeze(-1)
                .expand((*coords.shape[:-2], -1, 3)),
                dSoftmax.tile(grad_value.shape[-3]).unsqueeze(-1)
                * grad_value.flatten(start_dim=-3, end_dim=-2),
                "sum",
            )
        else:
            prod = dEnergy.tile(grad_value.shape[-3]).unsqueeze(
                -1
            ) * grad_value.flatten(start_dim=-3, end_dim=-2)
            if prod.dim() > 3:
                prod = prod.sum(dim=list(range(1, prod.dim() - 2)))
            grad_atom = torch.zeros_like(coords).scatter_reduce(
                -2,
                index.flatten(start_dim=0, end_dim=1)
                .unsqueeze(-1)
                .expand((*coords.shape[:-2], -1, 3)),  # 9 x 516 x 3
                prod,
                "sum",
            )

        if com_index is not None:
            grad_atom = grad_atom[..., com_index, :]
        elif ref_token_index is not None:
            grad_atom = grad_atom[..., ref_token_index, :]

        return grad_atom

    def compute_parameters(self, t):
        if self.parameters is None:
            return None
        parameters = {
            name: parameter
            if not isinstance(parameter, ParameterSchedule)
            else parameter.compute(t)
            for name, parameter in self.parameters.items()
        }
        return parameters

    @abstractmethod
    def compute_function(
        self, value, *args, negation_mask=None, compute_derivative=False
    ):
        raise NotImplementedError

    @abstractmethod
    def compute_variable(self, coords, index, compute_gradient=False):
        raise NotImplementedError

    @abstractmethod
    def compute_args(self, t, feats, **parameters):
        raise NotImplementedError

    def get_reference_coords(self, feats, parameters):
        return None, None


class FlatBottomPotential(Potential):
    def compute_function(
        self,
        value,
        k,
        lower_bounds,
        upper_bounds,
        negation_mask=None,
        compute_derivative=False,
    ):
        if lower_bounds is None:
            lower_bounds = torch.full_like(value, float("-inf"))
        if upper_bounds is None:
            upper_bounds = torch.full_like(value, float("inf"))
        lower_bounds = lower_bounds.expand_as(value).clone()
        upper_bounds = upper_bounds.expand_as(value).clone()

        if negation_mask is not None:
            unbounded_below_mask = torch.isneginf(lower_bounds)
            unbounded_above_mask = torch.isposinf(upper_bounds)
            unbounded_mask = unbounded_below_mask + unbounded_above_mask
            assert torch.all(unbounded_mask + negation_mask)
            lower_bounds[~unbounded_above_mask * ~negation_mask] = upper_bounds[
                ~unbounded_above_mask * ~negation_mask
            ]
            upper_bounds[~unbounded_above_mask * ~negation_mask] = float("inf")
            upper_bounds[~unbounded_below_mask * ~negation_mask] = lower_bounds[
                ~unbounded_below_mask * ~negation_mask
            ]
            lower_bounds[~unbounded_below_mask * ~negation_mask] = float("-inf")

        neg_overflow_mask = value < lower_bounds
        pos_overflow_mask = value > upper_bounds

        energy = torch.zeros_like(value)
        energy[neg_overflow_mask] = (k * (lower_bounds - value))[neg_overflow_mask]
        energy[pos_overflow_mask] = (k * (value - upper_bounds))[pos_overflow_mask]
        if not compute_derivative:
            return energy

        dEnergy = torch.zeros_like(value)
        dEnergy[neg_overflow_mask] = (
            -1 * k.expand_as(neg_overflow_mask)[neg_overflow_mask]
        )
        dEnergy[pos_overflow_mask] = (
            1 * k.expand_as(pos_overflow_mask)[pos_overflow_mask]
        )

        return energy, dEnergy


class ReferencePotential(Potential):
    def compute_variable(
        self, coords, index, ref_coords, ref_mask, compute_gradient=False
    ):
        aligned_ref_coords = weighted_rigid_align(
            ref_coords.float(),
            coords[:, index].float(),
            ref_mask,
            ref_mask,
        )

        r = coords[:, index] - aligned_ref_coords
        r_norm = torch.linalg.norm(r, dim=-1)

        if not compute_gradient:
            return r_norm

        r_hat = r / r_norm.unsqueeze(-1)
        grad = (r_hat * ref_mask.unsqueeze(-1)).unsqueeze(1)
        return r_norm, grad


class DistancePotential(Potential):
    def compute_variable(
        self, coords, index, ref_coords=None, ref_mask=None, compute_gradient=False
    ):
        # Ensure index is [2, N] format
        if index.dim() == 2 and index.shape[0] == 2:
            index_0 = index[0]
            index_1 = index[1]
        elif index.dim() == 1:
            # If index is 1D, assume it's a flat representation - this shouldn't happen but handle it
            raise ValueError(f"DistancePotential expects index to be [2, N] but got shape {index.shape}")
        else:
            raise ValueError(f"DistancePotential expects index to be [2, N] but got shape {index.shape}")
        
        # Ensure indices are 1D vectors for index_select
        if index_0.dim() != 1 or index_1.dim() != 1:
            raise ValueError(f"DistancePotential index[0] and index[1] must be 1D vectors, got shapes {index_0.shape} and {index_1.shape}")
        
        r_ij = coords.index_select(-2, index_0) - coords.index_select(-2, index_1)
        r_ij_norm = torch.linalg.norm(r_ij, dim=-1)
        r_hat_ij = r_ij / r_ij_norm.unsqueeze(-1)

        if not compute_gradient:
            return r_ij_norm

        grad_i = r_hat_ij
        grad_j = -1 * r_hat_ij
        grad = torch.stack((grad_i, grad_j), dim=1)
        return r_ij_norm, grad


class DihedralPotential(Potential):
    def compute_variable(
        self, coords, index, ref_coords=None, ref_mask=None, compute_gradient=False
    ):
        r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        r_kj = coords.index_select(-2, index[2]) - coords.index_select(-2, index[1])
        r_kl = coords.index_select(-2, index[2]) - coords.index_select(-2, index[3])

        n_ijk = torch.cross(r_ij, r_kj, dim=-1)
        n_jkl = torch.cross(r_kj, r_kl, dim=-1)

        r_kj_norm = torch.linalg.norm(r_kj, dim=-1)
        n_ijk_norm = torch.linalg.norm(n_ijk, dim=-1)
        n_jkl_norm = torch.linalg.norm(n_jkl, dim=-1)

        sign_phi = torch.sign(
            r_kj.unsqueeze(-2) @ torch.cross(n_ijk, n_jkl, dim=-1).unsqueeze(-1)
        ).squeeze(-1, -2)
        phi = sign_phi * torch.arccos(
            torch.clamp(
                (n_ijk.unsqueeze(-2) @ n_jkl.unsqueeze(-1)).squeeze(-1, -2)
                / (n_ijk_norm * n_jkl_norm),
                -1 + 1e-8,
                1 - 1e-8,
            )
        )

        if not compute_gradient:
            return phi

        a = (
            (r_ij.unsqueeze(-2) @ r_kj.unsqueeze(-1)).squeeze(-1, -2) / (r_kj_norm**2)
        ).unsqueeze(-1)
        b = (
            (r_kl.unsqueeze(-2) @ r_kj.unsqueeze(-1)).squeeze(-1, -2) / (r_kj_norm**2)
        ).unsqueeze(-1)

        grad_i = n_ijk * (r_kj_norm / n_ijk_norm**2).unsqueeze(-1)
        grad_l = -1 * n_jkl * (r_kj_norm / n_jkl_norm**2).unsqueeze(-1)
        grad_j = (a - 1) * grad_i - b * grad_l
        grad_k = (b - 1) * grad_l - a * grad_i
        grad = torch.stack((grad_i, grad_j, grad_k, grad_l), dim=1)
        return phi, grad


class AbsDihedralPotential(DihedralPotential):
    def compute_variable(
        self, coords, index, ref_coords=None, ref_mask=None, compute_gradient=False
    ):
        if not compute_gradient:
            phi = super().compute_variable(
                coords, index, compute_gradient=compute_gradient
            )
            phi = torch.abs(phi)
            return phi

        phi, grad = super().compute_variable(
            coords, index, compute_gradient=compute_gradient
        )
        grad[(phi < 0)[..., None, :, None].expand_as(grad)] *= -1
        phi = torch.abs(phi)

        return phi, grad


class PoseBustersPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        pair_index = feats["rdkit_bounds_index"][0]
        lower_bounds = feats["rdkit_lower_bounds"][0].clone()
        upper_bounds = feats["rdkit_upper_bounds"][0].clone()
        bond_mask = feats["rdkit_bounds_bond_mask"][0]
        angle_mask = feats["rdkit_bounds_angle_mask"][0]

        lower_bounds[bond_mask * ~angle_mask] *= 1.0 - parameters["bond_buffer"]
        upper_bounds[bond_mask * ~angle_mask] *= 1.0 + parameters["bond_buffer"]
        lower_bounds[~bond_mask * angle_mask] *= 1.0 - parameters["angle_buffer"]
        upper_bounds[~bond_mask * angle_mask] *= 1.0 + parameters["angle_buffer"]
        lower_bounds[bond_mask * angle_mask] *= 1.0 - min(
            parameters["bond_buffer"], parameters["angle_buffer"]
        )
        upper_bounds[bond_mask * angle_mask] *= 1.0 + min(
            parameters["bond_buffer"], parameters["angle_buffer"]
        )
        lower_bounds[~bond_mask * ~angle_mask] *= 1.0 - parameters["clash_buffer"]
        upper_bounds[~bond_mask * ~angle_mask] = float("inf")

        vdw_radii = torch.zeros(
            const.num_elements, dtype=torch.float32, device=pair_index.device
        )
        vdw_radii[1:119] = torch.tensor(
            const.vdw_radii, dtype=torch.float32, device=pair_index.device
        )
        atom_vdw_radii = (
            feats["ref_element"].float() @ vdw_radii.unsqueeze(-1)
        ).squeeze(-1)[0]
        bond_cutoffs = 0.35 + atom_vdw_radii[pair_index].mean(dim=0)
        lower_bounds[~bond_mask] = torch.max(lower_bounds[~bond_mask], bond_cutoffs[~bond_mask])
        upper_bounds[bond_mask] = torch.min(upper_bounds[bond_mask], bond_cutoffs[bond_mask])

        k = torch.ones_like(lower_bounds)

        return pair_index, (k, lower_bounds, upper_bounds), None, None, None


class ConnectionsPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        pair_index = feats["connected_atom_index"][0]
        lower_bounds = None
        upper_bounds = torch.full(
            (pair_index.shape[1],), parameters["buffer"], device=pair_index.device
        )
        k = torch.ones_like(upper_bounds)

        return pair_index, (k, lower_bounds, upper_bounds), None, None, None


class VDWOverlapPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        atom_chain_id = (
            torch.bmm(
                feats["atom_to_token"].float(), feats["asym_id"].unsqueeze(-1).float()
            )
            .squeeze(-1)
            .long()
        )[0]
        atom_pad_mask = feats["atom_pad_mask"][0].bool()
        chain_sizes = torch.bincount(atom_chain_id[atom_pad_mask])
        single_ion_mask = (chain_sizes > 1)[atom_chain_id]

        vdw_radii = torch.zeros(
            const.num_elements, dtype=torch.float32, device=atom_chain_id.device
        )
        vdw_radii[1:119] = torch.tensor(
            const.vdw_radii, dtype=torch.float32, device=atom_chain_id.device
        )
        atom_vdw_radii = (
            feats["ref_element"].float() @ vdw_radii.unsqueeze(-1)
        ).squeeze(-1)[0]

        pair_index = torch.triu_indices(
            atom_chain_id.shape[0],
            atom_chain_id.shape[0],
            1,
            device=atom_chain_id.device,
        )

        pair_pad_mask = atom_pad_mask[pair_index].all(dim=0)
        pair_ion_mask = single_ion_mask[pair_index[0]] * single_ion_mask[pair_index[1]]

        num_chains = atom_chain_id.max() + 1
        connected_chain_index = feats["connected_chain_index"][0]
        connected_chain_matrix = torch.eye(
            num_chains, device=atom_chain_id.device, dtype=torch.bool
        )
        connected_chain_matrix[connected_chain_index[0], connected_chain_index[1]] = (
            True
        )
        connected_chain_matrix[connected_chain_index[1], connected_chain_index[0]] = (
            True
        )
        connected_chain_mask = connected_chain_matrix[
            atom_chain_id[pair_index[0]], atom_chain_id[pair_index[1]]
        ]

        pair_index = pair_index[
            :, pair_pad_mask * pair_ion_mask * ~connected_chain_mask
        ]

        lower_bounds = atom_vdw_radii[pair_index].sum(dim=0) * (
            1.0 - parameters["buffer"]
        )
        upper_bounds = None
        k = torch.ones_like(lower_bounds)

        return pair_index, (k, lower_bounds, upper_bounds), None, None, None


class SymmetricChainCOMPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        atom_chain_id = (
            torch.bmm(
                feats["atom_to_token"].float(), feats["asym_id"].unsqueeze(-1).float()
            )
            .squeeze(-1)
            .long()
        )[0]
        atom_pad_mask = feats["atom_pad_mask"][0].bool()
        chain_sizes = torch.bincount(atom_chain_id[atom_pad_mask])
        single_ion_mask = chain_sizes > 1

        pair_index = feats["symmetric_chain_index"][0]
        pair_ion_mask = single_ion_mask[pair_index[0]] * single_ion_mask[pair_index[1]]
        pair_index = pair_index[:, pair_ion_mask]
        lower_bounds = torch.full(
            (pair_index.shape[1],),
            parameters["buffer"],
            dtype=torch.float32,
            device=pair_index.device,
        )
        upper_bounds = None
        k = torch.ones_like(lower_bounds)

        return (
            pair_index,
            (k, lower_bounds, upper_bounds),
            (atom_chain_id, atom_pad_mask),
            None,
            None,
        )


class StereoBondPotential(FlatBottomPotential, AbsDihedralPotential):
    def compute_args(self, feats, parameters):
        stereo_bond_index = feats["stereo_bond_index"][0]
        stereo_bond_orientations = feats["stereo_bond_orientations"][0].bool()

        lower_bounds = torch.zeros(
            stereo_bond_orientations.shape, device=stereo_bond_orientations.device
        )
        upper_bounds = torch.zeros(
            stereo_bond_orientations.shape, device=stereo_bond_orientations.device
        )
        lower_bounds[stereo_bond_orientations] = torch.pi - parameters["buffer"]
        upper_bounds[stereo_bond_orientations] = float("inf")
        lower_bounds[~stereo_bond_orientations] = float("-inf")
        upper_bounds[~stereo_bond_orientations] = parameters["buffer"]

        k = torch.ones_like(lower_bounds)

        return stereo_bond_index, (k, lower_bounds, upper_bounds), None, None, None


class ChiralAtomPotential(FlatBottomPotential, DihedralPotential):
    def compute_args(self, feats, parameters):
        chiral_atom_index = feats["chiral_atom_index"][0]
        chiral_atom_orientations = feats["chiral_atom_orientations"][0].bool()

        lower_bounds = torch.zeros(
            chiral_atom_orientations.shape, device=chiral_atom_orientations.device
        )
        upper_bounds = torch.zeros(
            chiral_atom_orientations.shape, device=chiral_atom_orientations.device
        )
        lower_bounds[chiral_atom_orientations] = parameters["buffer"]
        upper_bounds[chiral_atom_orientations] = float("inf")
        upper_bounds[~chiral_atom_orientations] = -1 * parameters["buffer"]
        lower_bounds[~chiral_atom_orientations] = float("-inf")

        k = torch.ones_like(lower_bounds)
        return chiral_atom_index, (k, lower_bounds, upper_bounds), None, None, None


class PlanarBondPotential(FlatBottomPotential, AbsDihedralPotential):
    def compute_args(self, feats, parameters):
        double_bond_index = feats["planar_bond_index"][0].T
        double_bond_improper_index = torch.tensor(
            [
                [1, 2, 3, 0],
                [4, 5, 0, 3],
            ],
            device=double_bond_index.device,
        ).T
        improper_index = (
            double_bond_index[:, double_bond_improper_index]
            .swapaxes(0, 1)
            .flatten(start_dim=1)
        )
        lower_bounds = None
        upper_bounds = torch.full(
            (improper_index.shape[1],),
            parameters["buffer"],
            device=improper_index.device,
        )
        k = torch.ones_like(upper_bounds)

        return improper_index, (k, lower_bounds, upper_bounds), None, None, None


class TemplateReferencePotential(FlatBottomPotential, ReferencePotential):
    def compute_args(self, feats, parameters):
        if "template_mask_cb" not in feats or "template_force" not in feats:
            return torch.empty([1, 0]), None, None, None, None

        template_mask = feats["template_mask_cb"][feats["template_force"]]
        if template_mask.shape[0] == 0:
            return torch.empty([1, 0]), None, None, None, None

        ref_coords = feats["template_cb"][feats["template_force"]].clone()
        ref_mask = feats["template_mask_cb"][feats["template_force"]].clone()
        ref_atom_index = (
            torch.bmm(
                feats["token_to_rep_atom"].float(),
                torch.arange(
                    feats["atom_pad_mask"].shape[1],
                    device=feats["atom_pad_mask"].device,
                    dtype=torch.float32,
                )[None, :, None],
            )
            .squeeze(-1)
            .long()
        )[0]
        ref_token_index = (
            torch.bmm(
                feats["atom_to_token"].float(),
                feats["token_index"].unsqueeze(-1).float(),
            )
            .squeeze(-1)
            .long()
        )[0]

        index = torch.arange(
            template_mask.shape[-1], dtype=torch.long, device=template_mask.device
        )[None]
        upper_bounds = torch.full(
            template_mask.shape, float("inf"), device=index.device, dtype=torch.float32
        )
        ref_idxs = torch.argwhere(template_mask).T
        upper_bounds[ref_idxs.unbind()] = feats["template_force_threshold"][
            feats["template_force"]
        ][ref_idxs[0]]

        lower_bounds = None
        k = torch.ones_like(upper_bounds)
        return (
            index,
            (k, lower_bounds, upper_bounds),
            None,
            (ref_coords, ref_mask, ref_atom_index, ref_token_index),
            None,
        )


class ContactPotentital(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        index = feats["contact_pair_index"][0]
        union_index = feats["contact_union_index"][0]
        negation_mask = feats["contact_negation_mask"][0]
        lower_bounds = None
        upper_bounds = feats["contact_thresholds"][0].clone()
        k = torch.ones_like(upper_bounds)
        return (
            index,
            (k, lower_bounds, upper_bounds),
            None,
            None,
            (negation_mask, union_index),
        )
        
class RepulsionContactPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        index = feats["contact_pair_index"][0]
        union_index = feats["contact_union_index"][0]
        negation_mask = feats["contact_negation_mask"][0]
        lower_bounds = feats["contact_thresholds"][0].clone()
        upper_bounds = None
        k = torch.ones_like(lower_bounds)
        return (
            index,
            (k, lower_bounds, upper_bounds),
            None,
            None,
            (negation_mask, union_index),
        )


class PLIPInteractionPotential(FlatBottomPotential, DistancePotential):
    """
    Potential for PLIP-derived protein-ligand interactions.
    
    Supports multiple interaction types:
    - Hydrogen bonds (distance constraints on donor-acceptor pairs)
    - Hydrophobic contacts (favorable distance ranges)
    - Salt bridges (ionic interactions)
    - Pi-stacking and Pi-cation interactions
    - Halogen bonds
    
    The potential reads interaction data from feats and applies distance
    constraints to guide the model toward preserving known interactions.
    """
    
    def compute_args(self, feats, parameters):
        """
        Extract PLIP interaction indices and distance constraints from feats.
        
        Expected features:
        - plip_interaction_index: [2, N] tensor of atom pair indices
        - plip_interaction_type: [N] tensor of interaction type codes
          (0: H-bond, 1: hydrophobic, 2: salt bridge, 3: pi-stacking, etc.)
        - plip_target_distance: [N] tensor of target/reference distances
        - plip_interaction_mask: [N] boolean mask for valid interactions
        
        Parameters:
        - hbond_tolerance: tolerance for hydrogen bond distances (Å)
        - hydrophobic_tolerance: tolerance for hydrophobic contacts (Å)
        - ionic_tolerance: tolerance for salt bridges (Å)
        - strength_weight: weight factor for interaction strength
        """
        # Check if PLIP features are present
        if "plip_interaction_index" not in feats:
            print("DEBUG: plip_interaction_index not in feats")
            return torch.empty([2, 0], device=feats["atom_pad_mask"].device), (
                torch.empty([0]), None, None
            ), None, None, None
        
        raw_pair_index = feats["plip_interaction_index"][0]
        print(f"DEBUG: plip_interaction_index raw shape: {raw_pair_index.shape}, dim: {raw_pair_index.dim()}")
        
        # Handle different possible shapes: [1, 2, N] or [2, N]
        if raw_pair_index.dim() == 3:
            # Remove batch dimension: [1, 2, N] -> [2, N]
            pair_index = raw_pair_index.squeeze(0)
            print(f"DEBUG: After squeeze(0), shape: {pair_index.shape}")
        elif raw_pair_index.dim() == 2:
            pair_index = raw_pair_index
        else:
            raise ValueError(f"plip_interaction_index must be [2, N] or [1, 2, N] but got shape {raw_pair_index.shape}")
        
        # Check if empty after shape handling
        if pair_index.shape[1] == 0:
            print("DEBUG: pair_index is empty after shape handling")
            return torch.empty([2, 0], device=pair_index.device), (
                torch.empty([0]), None, None
            ), None, None, None
        
        # Ensure pair_index is [2, N] format
        if pair_index.dim() != 2 or pair_index.shape[0] != 2:
            raise ValueError(f"plip_interaction_index must be [2, N] but got shape {pair_index.shape} after processing")
        
        print(f"DEBUG: Final pair_index shape: {pair_index.shape}")
        
        raw_interaction_types = feats["plip_interaction_type"][0]
        raw_target_distances = feats["plip_target_distance"][0]
        raw_interaction_mask = feats.get("plip_interaction_mask", [torch.ones(pair_index.shape[1], dtype=torch.bool, device=pair_index.device)])[0]
        
        # Handle batch dimension if present
        if raw_interaction_types.dim() == 2:
            interaction_types = raw_interaction_types.squeeze(0)
        else:
            interaction_types = raw_interaction_types
        if raw_target_distances.dim() == 2:
            target_distances = raw_target_distances.squeeze(0)
        else:
            target_distances = raw_target_distances
        if raw_interaction_mask.dim() == 2:
            interaction_mask = raw_interaction_mask.squeeze(0)
        else:
            interaction_mask = raw_interaction_mask
        
        print(f"DEBUG: interaction_types shape: {interaction_types.shape}, target_distances shape: {target_distances.shape}, mask shape: {interaction_mask.shape}")
        
        # Filter by mask
        pair_index = pair_index[:, interaction_mask]
        interaction_types = interaction_types[interaction_mask]
        target_distances = target_distances[interaction_mask]
        
        if pair_index.shape[1] == 0:
            return torch.empty([2, 0], device=pair_index.device), (
                torch.empty([0], device=pair_index.device), None, None
            ), None, None, None
        
        # Get interaction strengths if available
        raw_interaction_strengths = feats.get("plip_interaction_strength", [torch.ones_like(target_distances)])[0]
        if raw_interaction_strengths.dim() == 2:
            interaction_strengths = raw_interaction_strengths.squeeze(0)
        else:
            interaction_strengths = raw_interaction_strengths
        interaction_strengths = interaction_strengths[interaction_mask]
        
        # Define tolerances for different interaction types
        # Type codes: 0=H-bond, 1=hydrophobic, 2=salt_bridge, 3=pi-stack, 4=pi-cation, 5=halogen
        tolerances = torch.zeros_like(target_distances)
        
        # Hydrogen bonds: tight constraints (0=H-bond)
        hbond_mask = interaction_types == 0
        tolerances[hbond_mask] = parameters.get("hbond_tolerance", 0.3)
        
        # Hydrophobic contacts: wider tolerance (1=hydrophobic)
        hydrophobic_mask = interaction_types == 1
        tolerances[hydrophobic_mask] = parameters.get("hydrophobic_tolerance", 0.5)
        
        # Salt bridges: moderate tolerance (2=salt_bridge)
        saltbridge_mask = interaction_types == 2
        tolerances[saltbridge_mask] = parameters.get("ionic_tolerance", 0.4)
        
        # Pi-stacking: moderate tolerance (3=pi-stack)
        pistack_mask = interaction_types == 3
        tolerances[pistack_mask] = parameters.get("pi_tolerance", 0.4)
        
        # Pi-cation: moderate tolerance (4=pi-cation)
        pication_mask = interaction_types == 4
        tolerances[pication_mask] = parameters.get("pi_tolerance", 0.4)
        
        # Halogen bonds: tight constraints (5=halogen)
        halogen_mask = interaction_types == 5
        tolerances[halogen_mask] = parameters.get("halogen_tolerance", 0.3)
        
        # Set lower and upper bounds based on target distances and tolerances
        lower_bounds = target_distances - tolerances
        upper_bounds = target_distances + tolerances
        
        # Ensure non-negative distances
        lower_bounds = torch.clamp(lower_bounds, min=0.0)
        
        # Apply strength-based weighting to force constants
        k = interaction_strengths * parameters.get("strength_weight", 1.0)
        
        return pair_index, (k, lower_bounds, upper_bounds), None, None, None


class PLIPHydrogenBondPotential(FlatBottomPotential, DistancePotential):
    """
    Specialized potential for PLIP hydrogen bonds with angle considerations.
    
    This potential specifically targets hydrogen bonds and can incorporate
    angle information if available (donor-H-acceptor angle).
    """
    
    def compute_args(self, feats, parameters):
        """
        Extract hydrogen bond constraints from PLIP annotations.
        
        Expected features:
        - plip_hbond_donor_idx: [N] donor atom indices
        - plip_hbond_acceptor_idx: [N] acceptor atom indices  
        - plip_hbond_distance: [N] optimal H...A distances
        - plip_hbond_strength: [N] bond strength scores
        """
        if "plip_hbond_donor_idx" not in feats:
            print("DEBUG: plip_hbond_donor_idx not in feats")
            return torch.empty([2, 0], device=feats["atom_pad_mask"].device), (
                torch.empty([0]), None, None
            ), None, None, None
        
        raw_donor_idx = feats["plip_hbond_donor_idx"][0]
        raw_acceptor_idx = feats["plip_hbond_acceptor_idx"][0]
        raw_target_distances = feats["plip_hbond_distance"][0]
        
        print(f"DEBUG: plip_hbond_donor_idx shape: {raw_donor_idx.shape}, acceptor_idx shape: {raw_acceptor_idx.shape}")
        
        # Handle batch dimension if present: [1, N] -> [N]
        if raw_donor_idx.dim() == 2:
            donor_idx = raw_donor_idx.squeeze(0)
        else:
            donor_idx = raw_donor_idx
        if raw_acceptor_idx.dim() == 2:
            acceptor_idx = raw_acceptor_idx.squeeze(0)
        else:
            acceptor_idx = raw_acceptor_idx
        if raw_target_distances.dim() == 2:
            target_distances = raw_target_distances.squeeze(0)
        else:
            target_distances = raw_target_distances
        
        # Check if empty
        if donor_idx.shape[0] == 0:
            print("DEBUG: donor_idx is empty")
            return torch.empty([2, 0], device=donor_idx.device), (
                torch.empty([0]), None, None
            ), None, None, None
        
        # Ensure indices are 1D
        if donor_idx.dim() != 1:
            donor_idx = donor_idx.flatten()
        if acceptor_idx.dim() != 1:
            acceptor_idx = acceptor_idx.flatten()
        
        # Ensure they have the same length
        if donor_idx.shape[0] != acceptor_idx.shape[0]:
            raise ValueError(f"Donor and acceptor indices must have same length, got {donor_idx.shape[0]} and {acceptor_idx.shape[0]}")
        
        # Stack into pair index
        pair_index = torch.stack([donor_idx, acceptor_idx], dim=0)
        print(f"DEBUG: Stacked pair_index shape: {pair_index.shape}")
        
        # Get strength information
        raw_hbond_strengths = feats.get("plip_hbond_strength", [torch.ones_like(target_distances)])[0]
        if raw_hbond_strengths.dim() == 2:
            hbond_strengths = raw_hbond_strengths.squeeze(0)
        else:
            hbond_strengths = raw_hbond_strengths
        
        # Hydrogen bonds should be tight - use narrow tolerance window
        tolerance = parameters.get("hbond_tolerance", 0.25)  # ±0.25 Å
        
        lower_bounds = torch.clamp(target_distances - tolerance, min=1.5)  # H-bonds typically > 1.5 Å
        upper_bounds = target_distances + tolerance
        
        # Stronger H-bonds get higher force constants
        k = hbond_strengths * parameters.get("hbond_force_constant", 1.0)
        
        return pair_index, (k, lower_bounds, upper_bounds), None, None, None


class PLIPHydrophobicPotential(FlatBottomPotential, DistancePotential):
    """
    Potential for hydrophobic interactions from PLIP analysis.
    
    Hydrophobic contacts are generally weaker and have more flexible
    distance requirements compared to H-bonds.
    """
    
    def compute_args(self, feats, parameters):
        """
        Extract hydrophobic contact constraints from PLIP annotations.
        
        Expected features:
        - plip_hydrophobic_ligand_idx: [N] ligand atom indices
        - plip_hydrophobic_protein_idx: [N] protein atom indices
        - plip_hydrophobic_distance: [N] observed distances
        """
        if "plip_hydrophobic_ligand_idx" not in feats:
            print("DEBUG: plip_hydrophobic_ligand_idx not in feats")
            return torch.empty([2, 0], device=feats["atom_pad_mask"].device), (
                torch.empty([0]), None, None
            ), None, None, None
        
        raw_ligand_idx = feats["plip_hydrophobic_ligand_idx"][0]
        raw_protein_idx = feats["plip_hydrophobic_protein_idx"][0]
        raw_target_distances = feats["plip_hydrophobic_distance"][0]
        
        print(f"DEBUG: plip_hydrophobic_ligand_idx shape: {raw_ligand_idx.shape}, protein_idx shape: {raw_protein_idx.shape}")
        
        # Handle batch dimension if present: [1, N] -> [N]
        if raw_ligand_idx.dim() == 2:
            ligand_idx = raw_ligand_idx.squeeze(0)
        else:
            ligand_idx = raw_ligand_idx
        if raw_protein_idx.dim() == 2:
            protein_idx = raw_protein_idx.squeeze(0)
        else:
            protein_idx = raw_protein_idx
        if raw_target_distances.dim() == 2:
            target_distances = raw_target_distances.squeeze(0)
        else:
            target_distances = raw_target_distances
        
        # Check if empty
        if ligand_idx.shape[0] == 0:
            print("DEBUG: ligand_idx is empty")
            return torch.empty([2, 0], device=ligand_idx.device), (
                torch.empty([0]), None, None
            ), None, None, None
        
        # Ensure indices are 1D
        if ligand_idx.dim() != 1:
            ligand_idx = ligand_idx.flatten()
        if protein_idx.dim() != 1:
            protein_idx = protein_idx.flatten()
        
        # Ensure they have the same length
        if ligand_idx.shape[0] != protein_idx.shape[0]:
            raise ValueError(f"Ligand and protein indices must have same length, got {ligand_idx.shape[0]} and {protein_idx.shape[0]}")
        
        pair_index = torch.stack([ligand_idx, protein_idx], dim=0)
        
        # Hydrophobic contacts are more flexible
        tolerance = parameters.get("hydrophobic_tolerance", 0.6)  # ±0.6 Å
        
        lower_bounds = torch.clamp(target_distances - tolerance, min=2.5)  # Typically > 2.5 Å
        upper_bounds = target_distances + tolerance
        
        # Uniform force constant for hydrophobic contacts
        k = torch.ones_like(target_distances) * parameters.get("hydrophobic_force_constant", 0.5)
        
        return pair_index, (k, lower_bounds, upper_bounds), None, None, None

def get_potentials(steering_args, boltz2=False):
    potentials = []
    if steering_args["fk_steering"] or steering_args["physical_guidance_update"]:
        potentials.extend(
            [
                SymmetricChainCOMPotential(
                    parameters={
                        "guidance_interval": 4,
                        "guidance_weight": 0.5
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 0.5,
                        "buffer": ExponentialInterpolation(
                            start=1.0, end=5.0, alpha=-2.0
                        ),
                    }
                ),
                VDWOverlapPotential(
                    parameters={
                        "guidance_interval": 5,
                        "guidance_weight": (
                            PiecewiseStepFunction(thresholds=[0.4], values=[0.125, 0.0])
                            if steering_args["physical_guidance_update"]
                            else 0.0
                        ),
                        "resampling_weight": PiecewiseStepFunction(
                            thresholds=[0.6], values=[0.01, 0.0]
                        ),
                        "buffer": 0.225,
                    }
                ),
                ConnectionsPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.15
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 2.0,
                    }
                ),
                # PoseBustersPotential(
                #     parameters={
                #         "guidance_interval": 1,
                #         "guidance_weight": 0.01
                #         if steering_args["physical_guidance_update"]
                #         else 0.0,
                #         "resampling_weight": 0.1,
                #         "bond_buffer": 0.125,
                #         "angle_buffer": 0.125,
                #         "clash_buffer": 0.10,
                #     }
                # ),
                ChiralAtomPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.1
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 0.52360,
                    }
                ),
                StereoBondPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.05
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 0.52360,
                    }
                ),
                PlanarBondPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.05
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 0.26180,
                    }
                ),
            ]
        )
    if boltz2 and (
        steering_args["fk_steering"] or steering_args["contact_guidance_update"]
    ):
        potentials.extend(
            [
                ContactPotentital(
                    parameters={
                        "guidance_interval": 4,
                        "guidance_weight": (
                            PiecewiseStepFunction(
                                thresholds=[0.25, 0.75], values=[0.0, 0.5, 1.0]
                            )
                            if steering_args["contact_guidance_update"]
                            else 0.0
                        ),
                        "resampling_weight": 1.0,
                        "union_lambda": ExponentialInterpolation(
                            start=8.0, end=0.0, alpha=-2.0
                        ),
                    }
                ),
                TemplateReferencePotential(
                    parameters={
                        "guidance_interval": 2,
                        "guidance_weight": 0.1
                        if steering_args["contact_guidance_update"]
                        else 0.0,
                        "resampling_weight": 1.0,
                    }
                ),
            ]
        )
    
    # Add PLIP-based interaction potentials if enabled
    # These are applied only in late time steps (when structure is more formed)
    # to avoid enforcing interactions on invalid/early structures
    if steering_args.get("interaction_guidance_update", False):
        print("Adding PLIP-based interaction potentials (late time steps only)")
        # Use PiecewiseStepFunction to activate only in late time steps
        # In diffusion, t goes from 1.0 (noise) to 0.0 (data)
        # thresholds=[0.3] with values=[0.0, weight] means:
        #   - t > 0.3 (early steps): inactive (0.0)
        #   - t <= 0.3 (late steps): active (weight)
        late_step_threshold = 0.5  # Activate in last 30% of diffusion process
        
        potentials.append(
            PLIPHydrogenBondPotential(
                parameters={
                    "guidance_interval": 2,
                    "guidance_weight": PiecewiseStepFunction(
                        thresholds=[late_step_threshold], 
                        values=[0.0, 10.0]  # Inactive early, active late with increased weight 1.0
                    ),
                    "resampling_weight": 4.0,
                    "hbond_tolerance": 0.1,  # ±0.25 Å
                    "hbond_force_constant": 1.5,  # Increased from 1.5 to 3.0 for stronger enforcement
                }
            )
        )
    
    return potentials
