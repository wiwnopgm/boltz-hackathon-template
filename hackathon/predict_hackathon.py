# predict_hackathon.py
import argparse
import json
import os
import shutil
import subprocess
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple

import numpy as np
import yaml
from hackathon_api import Datapoint, Protein, SmallMolecule
from Bio.PDB import PDBParser, Superimposer, NeighborSearch

# ---------------------------------------------------------------------------
# ---- Orthosteric Pocket Identification Functions --------------------------
# ---------------------------------------------------------------------------

MODEL_COUNT = 5
LIG_NAME = "LIG"
RMSD_CUTOFF = 2.0   # Å
CONTACT_CUTOFF = 4.5  # Å

def _load_structure(path: Path):
    """Load a PDB structure."""
    return PDBParser(QUIET=True).get_structure(path.name, str(path))

def _ca_pairs(ref, mob):
    """Find matching CA atom pairs between reference and mobile structures."""
    ref_atoms, mob_atoms = [], []
    ref_chains = {c.id: c for c in ref.get_chains()}
    for chain in mob.get_chains():
        if chain.id not in ref_chains:
            continue
        for res in chain:
            if "CA" in res:
                try:
                    ref_res = ref_chains[chain.id][(" ", res.id[1], " ")]
                    if "CA" in ref_res:
                        ref_atoms.append(ref_res["CA"])
                        mob_atoms.append(res["CA"])
                except KeyError:
                    continue
    return ref_atoms, mob_atoms

def _superpose_structure(ref, mob):
    """Superpose mobile structure onto reference."""
    ref_atoms, mob_atoms = _ca_pairs(ref, mob)
    if len(ref_atoms) < 3:
        return
    sup = Superimposer()
    sup.set_atoms(ref_atoms, mob_atoms)
    sup.apply(mob.get_atoms())

def _ligand_coords(structure):
    """Extract ligand coordinates from structure."""
    for res in structure.get_residues():
        if res.get_resname().strip().upper() == LIG_NAME:
            return {a.get_name(): a.coord for a in res}
    return {}

def _ligand_rmsd(a, b):
    """Calculate RMSD between two ligand coordinate sets."""
    common = [k for k in a if k in b]
    if not common:
        return float("inf")
    s = 0.0
    for k in common:
        diff = a[k] - b[k]
        s += (diff * diff).sum()
    return math.sqrt(s / len(common))

def _pairwise_rmsd(structs):
    """Calculate pairwise RMSD matrix for ligands."""
    ligs = [_ligand_coords(s) for s in structs]
    n = len(ligs)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            val = _ligand_rmsd(ligs[i], ligs[j])
            mat[i][j] = mat[j][i] = val
    return mat

def _medoid_index(dist, cutoff):
    """Find medoid (representative) structure based on RMSD matrix."""
    n = len(dist)
    counts = [sum(d <= cutoff for d in row) for row in dist]
    maxc = max(counts)
    candidates = [i for i, c in enumerate(counts) if c == maxc]
    if len(candidates) == 1:
        return candidates[0]
    means = [(i, sum(dist[i]) / (n - 1)) for i in candidates]
    means.sort(key=lambda x: x[1])
    return means[0][0]

def _contact_residues(structure, cutoff) -> Set[Tuple[str, str, int]]:
    """Find residues within cutoff distance of ligand."""
    lig_atoms = [a for a in structure.get_atoms()
                 if a.get_parent().get_resname().strip().upper() == LIG_NAME]
    if not lig_atoms:
        return set()
    ns = NeighborSearch(list(structure.get_atoms()))
    contacts = set()
    for lig in lig_atoms:
        for atom in ns.search(lig.coord, cutoff):
            res = atom.get_parent()
            if res.get_resname().upper() == LIG_NAME or res.id[0] != ' ':
                continue
            chain = res.get_parent().id
            contacts.add((chain, res.get_resname(), res.id[1]))
    return contacts

def _calculate_pocket_rmsd_between_structures(ref_struct, mob_struct, pocket_residues: Set[Tuple[str, str, int]]) -> Optional[float]:
    """Calculate CA-RMSD of pocket residues between two structures."""
    ref_cas = []
    mob_cas = []
    
    for chain, resn, resi in pocket_residues:
        try:
            # Get CA from reference structure
            for model in ref_struct:
                for chain_obj in model:
                    if chain_obj.id == chain:
                        ref_res = chain_obj[(" ", resi, " ")]
                        if "CA" in ref_res:
                            ref_ca = ref_res["CA"]
                            # Get corresponding CA from mobile structure
                            for mob_model in mob_struct:
                                for mob_chain in mob_model:
                                    if mob_chain.id == chain:
                                        mob_res = mob_chain[(" ", resi, " ")]
                                        if "CA" in mob_res:
                                            ref_cas.append(ref_ca.coord)
                                            mob_cas.append(mob_res["CA"].coord)
                                            break
                            break
        except (KeyError, StopIteration):
            continue
    
    if len(ref_cas) < 3:
        return None
    
    # Calculate RMSD
    ref_coords = np.array(ref_cas)
    mob_coords = np.array(mob_cas)
    diff = ref_coords - mob_coords
    rmsd = math.sqrt(np.sum(diff * diff) / len(ref_cas))
    return rmsd

def identify_orthosteric_pocket(prediction_dir: Path) -> Set[Tuple[str, str, int]]:
    """
    Identify orthosteric pocket residues from predicted structures.
    
    Args:
        prediction_dir: Directory containing model_0.pdb through model_4.pdb
        
    Returns:
        Set of tuples (chain_id, resname, residue_number) representing pocket residues
    """
    # Load all model structures
    paths = [prediction_dir / f"model_{i}.pdb" for i in range(MODEL_COUNT)]
    for p in paths:
        if not p.exists():
            print(f"WARNING: Missing model file {p}")
            return set()
    
    structs = [_load_structure(p) for p in paths]
    
    # Superpose all structures to the first one
    ref = structs[0]
    for s in structs[1:]:
        _superpose_structure(ref, s)
    
    # Calculate pairwise RMSD and find medoid
    dist = _pairwise_rmsd(structs)
    rep_idx = _medoid_index(dist, RMSD_CUTOFF)
    rep_path = paths[rep_idx]
    print(f"  Representative (medoid) model: {rep_path.name}")
    
    # Identify contact residues in the representative structure
    rep_struct = structs[rep_idx]
    contacts = _contact_residues(rep_struct, CONTACT_CUTOFF)
    
    print(f"  Found {len(contacts)} pocket residues within {CONTACT_CUTOFF} Å of ligand")
    
    return contacts

# ---------------------------------------------------------------------------
# ---- Participants should modify these four functions ----------------------
# ---------------------------------------------------------------------------

def prepare_protein_complex(datapoint_id: str, proteins: List[Protein], input_dict: dict, msa_dir: Optional[Path] = None) -> List[tuple[dict, List[str]]]:
    """
    Prepare input dict and CLI args for a protein complex prediction.
    You can return multiple configurations to run by returning a list of (input_dict, cli_args) tuples.
    Args:
        datapoint_id: The unique identifier for this datapoint
        proteins: List of protein sequences to predict as a complex
        input_dict: Prefilled input dict
        msa_dir: Directory containing MSA files (for computing relative paths)
    Returns:
        List of tuples of (final input dict that will get exported as YAML, list of CLI args). Each tuple represents a separate configuration to run.
    """
    # Please note:
    # `proteins`` will contain 3 chains
    # H,L: heavy and light chain of the Fv or Fab region
    # A: the antigen
    #
    # you can modify input_dict to change the input yaml file going into the prediction, e.g.
    # ```
    # input_dict["constraints"] = [{
    #   "contact": {
    #       "token1" : [CHAIN_ID, RES_IDX/ATOM_NAME], 
    #       "token1" : [CHAIN_ID, RES_IDX/ATOM_NAME]
    #   }
    # }]
    # ```
    #
    # will add contact constraints to the input_dict

    # Example: predict 5 structures
    cli_args = ["--diffusion_samples", "5", "--use_potentials"]
    return [(input_dict, cli_args)]

def prepare_protein_ligand(datapoint_id: str, protein: Protein, ligands: list[SmallMolecule], input_dict: dict, msa_dir: Optional[Path] = None) -> List[tuple[dict, List[str]]]:
    """
    Prepare input dict and CLI args for a protein-ligand prediction.
    You can return multiple configurations to run by returning a list of (input_dict, cli_args) tuples.
    Args:
        datapoint_id: The unique identifier for this datapoint
        protein: The protein sequence
        ligands: A list of a single small molecule ligand object 
        input_dict: Prefilled input dict (may already contain negative_pocket constraint from Stage 2)
        msa_dir: Directory containing MSA files (for computing relative paths)
    Returns:
        List of tuples of (final input dict that will get exported as YAML, list of CLI args). Each tuple represents a separate configuration to run.
    """
    # Please note:
    # `protein` is a single-chain target protein sequence with id A
    # `ligands` contains a single small molecule ligand object with unknown binding sites
    # 
    # If orthosteric pocket was identified in Stage 2, input_dict will already contain
    # a negative_pocket constraint in the format:
    # ```
    # "constraints": [{
    #     "negative_pocket": {
    #         "binder": "B",
    #         "contacts": [["A", 69], ["A", 70], ...],
    #         "min_distance": 10.0,
    #         "force": True
    #     }
    # }]
    # ```
    #
    # You can modify or add additional constraints as needed, e.g.:
    # ```
    # if "constraints" not in input_dict:
    #     input_dict["constraints"] = []
    # 
    # # Add a contact constraint
    # input_dict["constraints"].append({
    #     "contact": {
    #         "token1": ["A", 100],  # Protein residue
    #         "token2": ["B", 1]      # Ligand
    #     }
    # })
    # ```

    # Example: predict 5 structures
    cli_args = ["--diffusion_samples", "5", "--use_potentials"]
    
    return [(input_dict, cli_args)]

def post_process_protein_complex(datapoint: Datapoint, input_dicts: List[dict[str, Any]], cli_args_list: List[list[str]], prediction_dirs: List[Path]) -> List[Path]:
    """
    Return ranked model files for protein complex submission.
    Args:
        datapoint: The original datapoint object
        input_dicts: List of input dictionaries used for predictions (one per config)
        cli_args_list: List of command line arguments used for predictions (one per config)
        prediction_dirs: List of directories containing prediction results (one per config)
    Returns: 
        Sorted pdb file paths that should be used as your submission.
    """
    # Collect all PDBs from all configurations
    all_pdbs = []
    for prediction_dir in prediction_dirs:
        config_pdbs = sorted(prediction_dir.glob(f"{datapoint.datapoint_id}_config_*_model_*.pdb"))
        all_pdbs.extend(config_pdbs)

    # Sort all PDBs and return their paths
    all_pdbs = sorted(all_pdbs)
    return all_pdbs

def post_process_protein_ligand(datapoint: Datapoint, input_dicts: List[dict[str, Any]], cli_args_list: List[list[str]], prediction_dirs: List[Path], pocket_residues: Optional[Set[Tuple[str, str, int]]] = None, unconstrained_dir: Optional[Path] = None) -> List[Path]:
    """
    Return ranked model files for protein-ligand submission.
    Args:
        datapoint: The original datapoint object
        input_dicts: List of input dictionaries used for predictions (one per config)
        cli_args_list: List of command line arguments used for predictions (one per config)
        prediction_dirs: List of directories containing prediction results (one per config)
        pocket_residues: Optional set of orthosteric pocket residues
        unconstrained_dir: Optional directory containing unconstrained prediction models
    Returns: 
        Sorted pdb file paths that should be used as your submission.
    """
    # Collect all PDBs from all configurations
    all_pdbs = []
    for prediction_dir in prediction_dirs:
        config_pdbs = sorted(prediction_dir.glob(f"{datapoint.datapoint_id}_config_*_model_*.pdb"))
        all_pdbs.extend(config_pdbs)
    
    # Sort all PDBs
    all_pdbs = sorted(all_pdbs)
    
    # Calculate pocket RMSD for unconstrained structures if available
    if pocket_residues and unconstrained_dir and unconstrained_dir.exists():
        print(f"\n{'='*80}")
        print(f"Computing pocket RMSD for unconstrained structures")
        print(f"{'='*80}\n")
        
        unconstrained_paths = [unconstrained_dir / f"model_{i}.pdb" for i in range(5)]
        unconstrained_structs = []
        
        for path in unconstrained_paths:
            if path.exists():
                unconstrained_structs.append(_load_structure(path))
        
        if len(unconstrained_structs) >= 2:
            # Use first structure as reference
            ref_struct = unconstrained_structs[0]
            
            # Superpose all structures to reference
            for s in unconstrained_structs[1:]:
                _superpose_structure(ref_struct, s)
            
            # Calculate pocket RMSD for each structure
            print(f"Pocket RMSD analysis (relative to model_0, {len(pocket_residues)} pocket residues):")
            print(f"{'Model':<12} {'Pocket RMSD (Å)':<18} {'Status':<20}")
            print("-" * 50)
            
            rmsds = []
            for i, struct in enumerate(unconstrained_structs):
                if i == 0:
                    rmsd = 0.0  # Reference to itself
                else:
                    rmsd = _calculate_pocket_rmsd_between_structures(ref_struct, struct, pocket_residues)
                
                if rmsd is not None:
                    status = "RMSD > 4 Å" if rmsd > 4.0 else "RMSD ≤ 4 Å"
                    rmsds.append(rmsd)
                    print(f"model_{i:<6} {rmsd:>8.2f}             {status}")
                else:
                    print(f"model_{i:<6} {'N/A':>8}             Could not calculate")
            
            if rmsds:
                avg_rmsd = np.mean(rmsds[1:]) if len(rmsds) > 1 else 0.0  # Exclude reference
                print("-" * 50)
                print(f"Average pocket RMSD (excluding reference): {avg_rmsd:.2f} Å")
                
                high_rmsd_count = sum(1 for r in rmsds[1:] if r > 4.0)
                if high_rmsd_count > 0:
                    print(f"Number of structures with pocket RMSD > 4 Å: {high_rmsd_count}/{len(rmsds)-1}")
            print()
    
    return all_pdbs

# -----------------------------------------------------------------------------
# ---- End of participant section ---------------------------------------------
# -----------------------------------------------------------------------------


DEFAULT_OUT_DIR = Path("predictions")
DEFAULT_SUBMISSION_DIR = Path("submission")
DEFAULT_INPUTS_DIR = Path("inputs")

ap = argparse.ArgumentParser(
    description="Hackathon scaffold for Boltz predictions",
    epilog="Examples:\n"
            "  Single datapoint: python predict_hackathon.py --input-json examples/specs/example_protein_ligand.json --msa-dir ./msa --submission-dir submission --intermediate-dir intermediate\n"
            "  Multiple datapoints: python predict_hackathon.py --input-jsonl examples/test_dataset.jsonl --msa-dir ./msa --submission-dir submission --intermediate-dir intermediate",
    formatter_class=argparse.RawDescriptionHelpFormatter
)

input_group = ap.add_mutually_exclusive_group(required=True)
input_group.add_argument("--input-json", type=str,
                        help="Path to JSON datapoint for a single datapoint")
input_group.add_argument("--input-jsonl", type=str,
                        help="Path to JSONL file with multiple datapoint definitions")

ap.add_argument("--msa-dir", type=Path,
                help="Directory containing MSA files (for computing relative paths in YAML)")
ap.add_argument("--submission-dir", type=Path, required=False, default=DEFAULT_SUBMISSION_DIR,
                help="Directory to place final submissions")
ap.add_argument("--intermediate-dir", type=Path, required=False, default=Path("hackathon_intermediate"),
                help="Directory to place generated input YAML files and predictions")
ap.add_argument("--group-id", type=str, required=False, default=None,
                help="Group ID to set for submission directory (sets group rw access if specified)")
ap.add_argument("--result-folder", type=Path, required=False, default=None,
                help="Directory to save evaluation results. If set, will automatically run evaluation after predictions.")

args = ap.parse_args()

def _prefill_input_dict(datapoint_id: str, proteins: Iterable[Protein], ligands: Optional[list[SmallMolecule]] = None, msa_dir: Optional[Path] = None, constraints: Optional[list] = None, pocket_residues: Optional[Set[Tuple[str, str, int]]] = None) -> dict:
    """
    Prepare input dict for Boltz YAML.
    
    Args:
        datapoint_id: Unique identifier for the datapoint
        proteins: Iterable of protein objects
        ligands: Optional list of small molecule ligands
        msa_dir: Directory containing MSA files
        constraints: Optional list of constraint dictionaries
        pocket_residues: Optional set of orthosteric pocket residues (chain_id, resname, residue_number)
                        Will be added as negative_pocket constraint if provided
    
    Returns:
        Dictionary ready for YAML export
    """
    seqs = []
    for p in proteins:
        if msa_dir and p.msa:
            if Path(p.msa).is_absolute():
                msa_full_path = Path(p.msa)
            else:
                msa_full_path = msa_dir / p.msa
            try:
                msa_relative_path = os.path.relpath(msa_full_path, Path.cwd())
            except ValueError:
                msa_relative_path = str(msa_full_path)
        else:
            msa_relative_path = p.msa
        entry = {
            "protein": {
                "id": p.id,
                "sequence": p.sequence,
                "msa": msa_relative_path
            }
        }
        seqs.append(entry)
    if ligands:
        def _format_ligand(ligand: SmallMolecule) -> dict:
            output =  {
                "ligand": {
                    "id": ligand.id,
                    "smiles": ligand.smiles
                }
            }
            return output
        
        for ligand in ligands:
            seqs.append(_format_ligand(ligand))
    doc = {
        "version": 1,
        "sequences": seqs,
    }
    
    # Start with existing constraints or empty list
    all_constraints = list(constraints) if constraints else []
    
    # Add negative_pocket constraint from identified orthosteric pocket residues
    if pocket_residues and ligands:
        ligand_id = ligands[0].id  # Typically "B" for protein-ligand
        contacts = [[chain, resi] for chain, resn, resi in sorted(pocket_residues)]
        
        negative_pocket_constraint = {
            "negative_pocket": {
                "binder": ligand_id,
                "contacts": contacts,
                "min_distance": 10.0,
                "force": True
            }
        }
        all_constraints.append(negative_pocket_constraint)
        
        print(f"  Added negative_pocket constraint with {len(contacts)} residues (min_distance=10.0 Å)")
    
    # Add all constraints to the document
    if all_constraints:
        doc["constraints"] = all_constraints
    
    return doc

def _run_boltz_and_collect(datapoint) -> None:
    """
    New flow: prepare input dict, write yaml, run boltz, post-process, copy submissions.
    For protein_ligand: First run unconstrained predictions to identify orthosteric pocket.
    """
    out_dir = args.intermediate_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    subdir = args.submission_dir / datapoint.datapoint_id
    subdir.mkdir(parents=True, exist_ok=True)

    # For protein_ligand: Run unconstrained predictions first to identify pocket
    pocket_residues = None
    if datapoint.task_type == "protein_ligand":
        print(f"\n{'='*80}")
        print(f"Stage 1: Running unconstrained predictions to identify orthosteric pocket")
        print(f"{'='*80}\n")
        
        # Prepare unconstrained input dict (no constraints)
        unconstrained_dict = _prefill_input_dict(
            datapoint.datapoint_id, 
            datapoint.proteins, 
            datapoint.ligands, 
            args.msa_dir, 
            constraints=None  # No constraints for initial prediction
        )
        
        # Run unconstrained prediction
        input_dir = args.intermediate_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        
        yaml_path = input_dir / f"{datapoint.datapoint_id}_unconstrained.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(unconstrained_dict, f, sort_keys=False, default_flow_style=False)
        
        # Run boltz for unconstrained prediction
        cache = os.environ.get("BOLTZ_CACHE", str(Path.home() / ".boltz"))
        cmd = [
            "boltz", "predict", str(yaml_path),
            "--devices", "1",
            "--out_dir", str(out_dir),
            "--cache", cache,
            "--no_kernels",
            "--output_format", "pdb",
            "--diffusion_samples", "5",
            "--use_potentials"
        ]
        print(f"Running unconstrained prediction: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
        
        # Find the unconstrained prediction directory
        unconstrained_pred_dir = out_dir / f"boltz_results_{datapoint.datapoint_id}_unconstrained" / "predictions" / f"{datapoint.datapoint_id}_unconstrained"
        
        # Copy unconstrained predictions to submission directory for pocket identification
        temp_pocket_dir = subdir / "unconstrained_for_pocket"
        temp_pocket_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the predicted models
        unconstrained_pdbs = sorted(unconstrained_pred_dir.glob(f"{datapoint.datapoint_id}_unconstrained_model_*.pdb"))
        for i, pdb_path in enumerate(unconstrained_pdbs[:5]):
            shutil.copy2(pdb_path, temp_pocket_dir / f"model_{i}.pdb")
        
        # Identify orthosteric pocket
        print(f"\n{'='*80}")
        print(f"Stage 2: Identifying orthosteric pocket residues")
        print(f"{'='*80}\n")
        
        pocket_residues = identify_orthosteric_pocket(temp_pocket_dir)
        
        # Save pocket residues to file
        pocket_file = subdir / "orthosteric_pocket.txt"
        with open(pocket_file, "w") as f:
            f.write(f"# Orthosteric pocket residues for {datapoint.datapoint_id}\n")
            f.write(f"# Format: Chain ResName ResNum\n")
            f.write(f"# Contact cutoff: {CONTACT_CUTOFF} Å\n")
            f.write(f"# Total residues: {len(pocket_residues)}\n\n")
            for chain, resn, resi in sorted(pocket_residues):
                f.write(f"{chain}\t{resn}\t{resi}\n")
        
        print(f"\nOrthosteric pocket residues saved to: {pocket_file}")
        print(f"\nPocket residues:")
        for chain, resn, resi in sorted(pocket_residues):
            print(f"  Chain {chain}, {resn} {resi}")
        
        print(f"\n{'='*80}")
        print(f"Stage 3: Running final predictions (can use pocket info if needed)")
        print(f"{'='*80}\n")

    # Prepare input dict and CLI args for final predictions
    # Pass pocket_residues to _prefill_input_dict so they can be accessed in prepare_* functions
    base_input_dict = _prefill_input_dict(
        datapoint.datapoint_id, 
        datapoint.proteins, 
        datapoint.ligands, 
        args.msa_dir, 
        datapoint.constraints,
        pocket_residues=pocket_residues  # Pass identified pocket residues
    )

    if datapoint.task_type == "protein_complex":
        configs = prepare_protein_complex(datapoint.datapoint_id, datapoint.proteins, base_input_dict, args.msa_dir)
    elif datapoint.task_type == "protein_ligand":
        configs = prepare_protein_ligand(datapoint.datapoint_id, datapoint.proteins[0], datapoint.ligands, base_input_dict, args.msa_dir)
    else:
        raise ValueError(f"Unknown task_type: {datapoint.task_type}")

    # Run boltz for each configuration
    all_input_dicts = []
    all_cli_args = []
    all_pred_subfolders = []
    
    input_dir = args.intermediate_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    for config_idx, (input_dict, cli_args) in enumerate(configs):
        # Write input YAML with config index suffix
        yaml_path = input_dir / f"{datapoint.datapoint_id}_config_{config_idx}.yaml"
        with open(yaml_path, "w") as f:
            # Custom YAML formatting to keep contacts as inline lists
            class FlowListDumper(yaml.SafeDumper):
                pass
            
            def represent_list(dumper, data):
                # If this is a constraint contacts list (list of 2-element lists), use flow style
                if data and isinstance(data, list) and len(data) > 0:
                    if all(isinstance(item, list) and len(item) == 2 for item in data):
                        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
                return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)
            
            FlowListDumper.add_representer(list, represent_list)
            
            yaml.dump(input_dict, f, Dumper=FlowListDumper, sort_keys=False, default_flow_style=False)

        # Run boltz
        cache = os.environ.get("BOLTZ_CACHE", str(Path.home() / ".boltz"))
        fixed = [
            "boltz", "predict", str(yaml_path),
            "--devices", "1",
            "--out_dir", str(out_dir),
            "--cache", cache,
            "--no_kernels",
            "--output_format", "pdb",
        ]
        cmd = fixed + cli_args
        print(f"Running config {config_idx}:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        # Compute prediction subfolder for this config
        pred_subfolder = out_dir / f"boltz_results_{datapoint.datapoint_id}_config_{config_idx}" / "predictions" / f"{datapoint.datapoint_id}_config_{config_idx}"
        
        all_input_dicts.append(input_dict)
        all_cli_args.append(cli_args)
        all_pred_subfolders.append(pred_subfolder)

    # Post-process and copy submissions
    if datapoint.task_type == "protein_complex":
        ranked_files = post_process_protein_complex(datapoint, all_input_dicts, all_cli_args, all_pred_subfolders)
    elif datapoint.task_type == "protein_ligand":
        # Pass pocket residues and unconstrained prediction directory for RMSD analysis
        unconstrained_dir = subdir / "unconstrained_for_pocket" if pocket_residues else None
        ranked_files = post_process_protein_ligand(
            datapoint, all_input_dicts, all_cli_args, all_pred_subfolders, 
            pocket_residues=pocket_residues, unconstrained_dir=unconstrained_dir
        )
    else:
        raise ValueError(f"Unknown task_type: {datapoint.task_type}")

    if not ranked_files:
        raise FileNotFoundError(f"No model files found for {datapoint.datapoint_id}")

    for i, file_path in enumerate(ranked_files[:5]):
        target = subdir / (f"model_{i}.pdb" if file_path.suffix == ".pdb" else f"model_{i}{file_path.suffix}")
        shutil.copy2(file_path, target)
        print(f"Saved: {target}")

    if args.group_id:
        try:
            subprocess.run(["chgrp", "-R", args.group_id, str(subdir)], check=True)
            subprocess.run(["chmod", "-R", "g+rw", str(subdir)], check=True)
        except Exception as e:
            print(f"WARNING: Failed to set group ownership or permissions: {e}")

def _load_datapoint(path: Path):
    """Load JSON datapoint file."""
    with open(path) as f:
        return Datapoint.from_json(f.read())

def _run_evaluation(input_file: str, task_type: str, submission_dir: Path, result_folder: Path):
    """
    Run the appropriate evaluation script based on task type.
    
    Args:
        input_file: Path to the input JSON or JSONL file
        task_type: Either "protein_complex" or "protein_ligand"
        submission_dir: Directory containing prediction submissions
        result_folder: Directory to save evaluation results
    """
    script_dir = Path(__file__).parent
    
    if task_type == "protein_complex":
        eval_script = script_dir / "evaluate_abag.py"
        cmd = [
            "python", str(eval_script),
            "--dataset-file", input_file,
            "--submission-folder", str(submission_dir),
            "--result-folder", str(result_folder)
        ]
    elif task_type == "protein_ligand":
        eval_script = script_dir / "evaluate_asos.py"
        cmd = [
            "python", str(eval_script),
            "--dataset-file", input_file,
            "--submission-folder", str(submission_dir),
            "--result-folder", str(result_folder)
        ]
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
    print(f"\n{'=' * 80}")
    print(f"Running evaluation for {task_type}...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")
    
    subprocess.run(cmd, check=True)
    print(f"\nEvaluation complete. Results saved to {result_folder}")

def _process_jsonl(jsonl_path: str, msa_dir: Optional[Path] = None):
    """Process multiple datapoints from a JSONL file."""
    print(f"Processing JSONL file: {jsonl_path}")

    for line_num, line in enumerate(Path(jsonl_path).read_text().splitlines(), 1):
        if not line.strip():
            continue

        print(f"\n--- Processing line {line_num} ---")

        try:
            datapoint = Datapoint.from_json(line)
            _run_boltz_and_collect(datapoint)

        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON on line {line_num}: {e}")
            continue
        except Exception as e:
            print(f"ERROR: Failed to process datapoint on line {line_num}: {e}")
            raise e
            continue

def _process_json(json_path: str, msa_dir: Optional[Path] = None):
    """Process a single datapoint from a JSON file."""
    print(f"Processing JSON file: {json_path}")

    try:
        datapoint = _load_datapoint(Path(json_path))
        _run_boltz_and_collect(datapoint)
    except Exception as e:
        print(f"ERROR: Failed to process datapoint: {e}")
        raise

def main():
    """Main entry point for the hackathon scaffold."""
    # Determine task type from first datapoint for evaluation
    task_type = None
    input_file = None
    
    if args.input_json:
        input_file = args.input_json
        _process_json(args.input_json, args.msa_dir)
        # Get task type from the single datapoint
        try:
            datapoint = _load_datapoint(Path(args.input_json))
            task_type = datapoint.task_type
        except Exception as e:
            print(f"WARNING: Could not determine task type: {e}")
    elif args.input_jsonl:
        input_file = args.input_jsonl
        _process_jsonl(args.input_jsonl, args.msa_dir)
        # Get task type from first datapoint in JSONL
        try:
            with open(args.input_jsonl) as f:
                first_line = f.readline().strip()
                if first_line:
                    first_datapoint = Datapoint.from_json(first_line)
                    task_type = first_datapoint.task_type
        except Exception as e:
            print(f"WARNING: Could not determine task type: {e}")
    
    # Run evaluation if result folder is specified and task type was determined
    if args.result_folder and task_type and input_file:
        try:
            _run_evaluation(input_file, task_type, args.submission_dir, args.result_folder)
        except Exception as e:
            print(f"WARNING: Evaluation failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
