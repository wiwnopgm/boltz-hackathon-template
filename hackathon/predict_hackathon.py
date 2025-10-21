# predict_hackathon.py
import yaml
import sys
import argparse
import json
import glob
import os
import shutil
import subprocess
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
try:
    from hackathon.contrib import predict_binding_sites
except ImportError:
    # Fallback for when running as script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / 'contrib'))
    from simple_binding_predictor import predict_binding_sites
import boltz
from boltz.main import predict
import yaml
from hackathon_api import Datapoint, Protein, SmallMolecule
from Bio.PDB import PDBParser, Superimposer, NeighborSearch

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("WARNING: RDKit not available. Ligand similarity checking will be disabled.")

# ---------------------------------------------------------------------------
# ---- Orthosteric Pocket Identification Functions --------------------------
# ---------------------------------------------------------------------------

MODEL_COUNT = 5
LIG_NAME = "LIG"
RMSD_CUTOFF = 2.0   # Å
CONTACT_CUTOFF = 4.5  # Å

# Variance thresholds for allosteric detection (based on variance_comparison_ortho_allo.csv)
# Orthosteric: ~0.0023, Allosteric: ~0.0115
CONFIDENCE_STD_THRESHOLD = 0.005  
# Orthosteric: ~0.0025, Allosteric: ~0.031
IPTM_STD_THRESHOLD = 0.01
LIGAND_IPTM_STD_THRESHOLD = 0.01

# Ligand similarity threshold for allosteric detection
LIGAND_SIMILARITY_THRESHOLD = 0.80  # 80% Tanimoto similarity

# Cache for reference allosteric ligands
_REFERENCE_ALLOSTERIC_LIGANDS = None

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

def load_reference_allosteric_ligands(reference_dataset_path: Optional[Path] = None) -> List[str]:
    """
    Load SMILES strings of all allosteric ligands from the reference dataset.
    
    Args:
        reference_dataset_path: Path to the reference JSONL dataset file
        
    Returns:
        List of SMILES strings for allosteric ligands
    """
    global _REFERENCE_ALLOSTERIC_LIGANDS
    
    # Return cached results if available
    if _REFERENCE_ALLOSTERIC_LIGANDS is not None:
        return _REFERENCE_ALLOSTERIC_LIGANDS
    
    allosteric_smiles = []
    
    # Try to find the reference dataset
    if reference_dataset_path is None:
        # Look in common locations
        possible_paths = [
            Path("/home/ubuntu/will/boltz-hackathon-template/hackathon_data/datasets/asos_public/asos_public.jsonl"),
            Path.cwd() / "hackathon_data" / "datasets" / "asos_public" / "asos_public.jsonl",
        ]
        for path in possible_paths:
            if path.exists():
                reference_dataset_path = path
                break
    
    if reference_dataset_path is None or not reference_dataset_path.exists():
        print("WARNING: Reference dataset not found for ligand similarity checking")
        _REFERENCE_ALLOSTERIC_LIGANDS = []
        return []
    
    try:
        with open(reference_dataset_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Check if this datapoint has allosteric ligands
                    if "ground_truth" in data and "ligand_types" in data["ground_truth"]:
                        for ligand_type_info in data["ground_truth"]["ligand_types"]:
                            if ligand_type_info.get("type") == "allosteric":
                                # Get the corresponding ligand SMILES
                                if "ligands" in data:
                                    for ligand in data["ligands"]:
                                        # Match by ligand ID
                                        if ligand["id"] == ligand_type_info.get("ligand_id"):
                                            allosteric_smiles.append(ligand["smiles"])
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"WARNING: Error loading reference dataset: {e}")
        _REFERENCE_ALLOSTERIC_LIGANDS = []
        return []
    
    _REFERENCE_ALLOSTERIC_LIGANDS = allosteric_smiles
    print(f"Loaded {len(allosteric_smiles)} reference allosteric ligands from dataset")
    
    return allosteric_smiles

def compute_tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    """
    Compute Tanimoto similarity between two molecules given as SMILES.
    
    Args:
        smiles1: SMILES string of first molecule
        smiles2: SMILES string of second molecule
        
    Returns:
        Tanimoto similarity score (0.0 to 1.0), or 0.0 if comparison fails
    """
    if not RDKIT_AVAILABLE:
        return 0.0
    
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        # Generate Morgan fingerprints (ECFP4 equivalent)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
        
        # Calculate Tanimoto similarity
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        
        return float(similarity)
    except Exception as e:
        print(f"WARNING: Error computing similarity: {e}")
        return 0.0

def check_ligand_similarity_to_allosteric(query_smiles: str, reference_dataset_path: Optional[Path] = None, threshold: float = LIGAND_SIMILARITY_THRESHOLD) -> Tuple[bool, float, Optional[str]]:
    """
    Check if a query ligand is similar to any known allosteric ligands.
    
    Args:
        query_smiles: SMILES string of the query ligand
        reference_dataset_path: Path to reference dataset (optional)
        threshold: Similarity threshold (default 0.80)
        
    Returns:
        Tuple of (is_similar, max_similarity, most_similar_smiles)
    """
    if not RDKIT_AVAILABLE:
        return False, 0.0, None
    
    # Load reference allosteric ligands
    allosteric_ligands = load_reference_allosteric_ligands(reference_dataset_path)
    
    if not allosteric_ligands:
        return False, 0.0, None
    
    max_similarity = 0.0
    most_similar = None
    
    for ref_smiles in allosteric_ligands:
        similarity = compute_tanimoto_similarity(query_smiles, ref_smiles)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar = ref_smiles
    
    is_similar = max_similarity >= threshold
    
    return is_similar, max_similarity, most_similar

def extract_confidence_metrics(prediction_output_dir: Path, datapoint_id: str) -> Dict[str, List[float]]:
    """
    Extract confidence metrics from Boltz prediction outputs.
    
    Args:
        prediction_output_dir: Directory containing Boltz prediction outputs
        datapoint_id: ID of the datapoint
        
    Returns:
        Dictionary with lists of confidence scores, ligand iPTM, and protein iPTM for each model
    """
    metrics = {
        "confidence_scores": [],
        "ligand_iptm": [],
        "protein_iptm": []
    }
    
    # Try to find individual model confidence JSON files (format: confidence_{datapoint_id}_model_{i}.json)
    for i in range(MODEL_COUNT):
        # Try the standard naming pattern used by Boltz
        confidence_file = prediction_output_dir / f"confidence_{datapoint_id}_model_{i}.json"
        
        if confidence_file.exists():
            try:
                with open(confidence_file) as f:
                    data = json.load(f)
                    if "confidence_score" in data:
                        metrics["confidence_scores"].append(data["confidence_score"])
                    if "ligand_iptm" in data:
                        metrics["ligand_iptm"].append(data["ligand_iptm"])
                    if "iptm" in data:
                        metrics["protein_iptm"].append(data["iptm"])
            except Exception as e:
                print(f"WARNING: Could not parse {confidence_file}: {e}")
    
    # If no files found, report the issue
    if not metrics["confidence_scores"] and not metrics["ligand_iptm"] and not metrics["protein_iptm"]:
        print(f"WARNING: No confidence files found in {prediction_output_dir}")
        print(f"         Looking for pattern: confidence_{datapoint_id}_model_*.json")
    
    return metrics

def is_allosteric_binder(metrics: Dict[str, List[float]], ligand_smiles: Optional[str] = None, reference_dataset_path: Optional[Path] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Determine if binding is allosteric based on:
    1. Variance in confidence metrics
    2. Ligand similarity to known allosteric binders (if ligand_smiles provided)
    
    Args:
        metrics: Dictionary with lists of confidence scores, ligand iPTM, and protein iPTM
        ligand_smiles: Optional SMILES string of the query ligand for similarity checking
        reference_dataset_path: Optional path to reference dataset for ligand similarity
        
    Returns:
        Tuple of (is_allosteric, detection_stats)
    """
    detection_stats = {}
    
    # 1. Variance-based detection
    # Compute standard deviations
    if len(metrics["confidence_scores"]) > 1:
        detection_stats["confidence_std"] = float(np.std(metrics["confidence_scores"]))
    else:
        detection_stats["confidence_std"] = 0.0
    
    if len(metrics["ligand_iptm"]) > 1:
        detection_stats["ligand_iptm_std"] = float(np.std(metrics["ligand_iptm"]))
    else:
        detection_stats["ligand_iptm_std"] = 0.0
    
    if len(metrics["protein_iptm"]) > 1:
        detection_stats["protein_iptm_std"] = float(np.std(metrics["protein_iptm"]))
    else:
        detection_stats["protein_iptm_std"] = 0.0
    
    # Check if allosteric based on variance thresholds
    is_allosteric_by_variance = (
        detection_stats["confidence_std"] > CONFIDENCE_STD_THRESHOLD or
        detection_stats["ligand_iptm_std"] > LIGAND_IPTM_STD_THRESHOLD or
        detection_stats["protein_iptm_std"] > IPTM_STD_THRESHOLD
    )
    detection_stats["allosteric_by_variance"] = is_allosteric_by_variance
    
    # 2. Ligand similarity-based detection
    is_allosteric_by_similarity = False
    if ligand_smiles and RDKIT_AVAILABLE:
        is_similar, max_similarity, most_similar_smiles = check_ligand_similarity_to_allosteric(
            ligand_smiles, reference_dataset_path, LIGAND_SIMILARITY_THRESHOLD
        )
        detection_stats["allosteric_by_similarity"] = is_similar
        detection_stats["max_ligand_similarity"] = max_similarity
        detection_stats["most_similar_allosteric_ligand"] = most_similar_smiles
        is_allosteric_by_similarity = is_similar
    else:
        detection_stats["allosteric_by_similarity"] = False
        detection_stats["max_ligand_similarity"] = 0.0
        detection_stats["most_similar_allosteric_ligand"] = None
    
    # Combined decision: allosteric if EITHER variance OR similarity indicates it
    is_allosteric = is_allosteric_by_variance or is_allosteric_by_similarity
    detection_stats["detection_method"] = []
    if is_allosteric_by_variance:
        detection_stats["detection_method"].append("variance")
    if is_allosteric_by_similarity:
        detection_stats["detection_method"].append("similarity")
    
    return is_allosteric, detection_stats

# ---------------------------------------------------------------------------
# ---- Helper functions for binding site analysis --------------------------
# ---------------------------------------------------------------------------

def get_top_binding_sites(binding_probs, top_n, threshold=0.5):
    """
    Extract the top N binding site positions based on binding probabilities.
    
    Args:
        binding_probs: List of binding probabilities for each residue
        top_n: Number of top predictions to return
        threshold: Minimum binding probability threshold
        
    Returns:
        List of residue indices (1-based) with highest binding probabilities
    """
    import numpy as np
    
    # Convert to numpy array for easier manipulation
    probs = np.array(binding_probs)
    
    # Filter by threshold
    valid_indices = np.where(probs >= threshold)[0]
    
    if len(valid_indices) == 0:
        # If no residues meet threshold, get top N regardless
        top_indices = np.argsort(probs)[-top_n:][::-1]
    else:
        # Get top N from valid indices
        valid_probs = probs[valid_indices]
        top_valid_indices = np.argsort(valid_probs)[-min(top_n, len(valid_indices)):][::-1]
        top_indices = valid_indices[top_valid_indices]
    
    # Convert to 1-based indexing (residue numbers start from 1)
    return (top_indices + 1).tolist()

def create_pocket_constraints(residue_indices, protein_chain_id, ligand_id="B"):
    """
    Create pocket constraints for the input dictionary.
    
    Args:
        residue_indices: List of residue indices (1-based)
        protein_chain_id: Protein chain identifier (e.g., 'A')
        ligand_id: Ligand identifier (default: 'B')
        
    Returns:
        Dictionary with pocket constraint configuration
    """
    # Create contacts list with protein chain and residue index
    contacts = [[protein_chain_id, idx] for idx in residue_indices]
    
    return {
        "binder": ligand_id,
        "contacts": contacts,
        "max_distance": 2.0,
        "force": True
    }

# ---------------------------------------------------------------------------
# ---- Participants should modify these four functions ----------------------
# ---------------------------------------------------------------------------

def center_with_character(text: str, width: int = 40, char: str = "=") -> str:
    return text.center(width, char)


def predict_3d_structure(datapoint_id: str, input_dict: dict, diffusion_samples: int = 5) -> Path:
    input_dir = Path("intermediate_pdb_files") / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Prepare YAML input file for boltz
    yaml_path = input_dir / f"{datapoint_id}_config.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(input_dict, f, sort_keys=False)

    # Run boltz
    out_dir = Path("intermediate_pdb_files") / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = os.environ.get("BOLTZ_CACHE", str(Path.home() / ".boltz"))
    fixed = [
        "boltz", "predict", str(yaml_path),
        "--devices", "1",
        "--out_dir", str(out_dir),
        "--cache", cache,
        "--no_kernels",
        "--output_format", "pdb",
    ]
    cli_args = ["--diffusion_samples", f"{diffusion_samples}"]
    cmd = fixed + cli_args
    print(f"Running for {datapoint_id}:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    # Get pdb file with the best score
    pred_subfolder = out_dir / f"boltz_results_{datapoint_id}_config" / "predictions" / f"{datapoint_id}_config"
    sample_id_max = 0
    confidence_max = -1
    for sample_id in range(diffusion_samples):
        confidence_filename = pred_subfolder / f"confidence_{datapoint_id}_config_model_{sample_id}.json"
        with open(confidence_filename, "r") as fin:
            data = json.load(fin)
            confidence = data["confidence_score"]
            if confidence > confidence_max:
                sample_id_max = sample_id

    pdb_file_result = pred_subfolder / f"{datapoint_id}_config_model_{sample_id_max}.pdb"
    return pdb_file_result


def find_pockets(pdb_file: Path, pockets_limit: int = 20):
    # Run pocket finding
    cmd = [
        "fpocket", "-f", str(pdb_file),
    ]
    print(f"Running pocket prediction for {pdb_file}:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    # Get pockets residues
    pred_subfolder = pdb_file.parent / f"{pdb_file.stem}_out" / "pockets"

    residue_pockets = defaultdict(int)
    for pocket_file in glob.glob(f"{pred_subfolder}/pocket*_atm.pdb"):
        with open(pocket_file) as f:
            for line in f:
                if line.startswith("ATOM"):
                    chain, resnum = line.split()[4:6]
                    residue_pockets[(chain, resnum)] += 1

    return list(residue_pockets.keys())[:pockets_limit]


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

    print(center_with_character(text="prepare_protein_complex:predict_3d_structure", width=60))
    pdb_file = predict_3d_structure(datapoint_id, input_dict)
    print(pdb_file)

    print(center_with_character(text="prepare_protein_complex:find_pockets", width=60))
    pockets = find_pockets(pdb_file)
    print(pockets)

    # Example: predict 5 structures
    cli_args = ["--diffusion_samples", "5", "--use_potentials"]
    return [(input_dict, cli_args)]

def prepare_protein_ligand(datapoint_id: str, protein: Protein, ligands: list[SmallMolecule], input_dict: dict, msa_dir: Optional[Path] = None, is_allosteric: bool = False) -> List[tuple[dict, List[str]]]:
    """
    Prepare input dict and CLI args for a protein-ligand prediction.
    You can return multiple configurations to run by returning a list of (input_dict, cli_args) tuples.
    Args:
        datapoint_id: The unique identifier for this datapoint
        protein: The protein sequence
        ligands: A list of a single small molecule ligand object 
        input_dict: Prefilled input dict (may already contain negative_pocket constraint if allosteric)
        msa_dir: Directory containing MSA files (for computing relative paths)
        is_allosteric: Whether the binding is detected as allosteric based on variance
    Returns:
        List of tuples of (final input dict that will get exported as YAML, list of CLI args). Each tuple represents a separate configuration to run.
    """
    

    # Check if automatic pocket scanning is enabled
    if args.use_auto_pocket_scanner:
        # Extract protein sequence and ligand SMILES
        protein_sequence = protein.sequence
        ligand_smiles = ligands[0].smiles if ligands else ""
        
        # Run binding site prediction
        binding_results = predict_binding_sites(
            protein_sequence=protein_sequence,
            smiles=ligand_smiles,
            output_dir=f"binding_output/{datapoint_id}/",
            use_boltz=False,
        )
        
        print(f"Binding site prediction completed for {datapoint_id}")
        print(f"Found {len(binding_results['binding_probabilities'])} residues with binding probabilities")
        print(f"Results saved to: {binding_results['result_csv_path']}")
        
        # Extract top N binding site predictions and create pocket constraints
        # You can customize these parameters:
        top_n_predictions = 5 # Number of top predictions to use
        binding_threshold = 0.8  # Minimum binding probability threshold
        
        # Get top binding site positions
        binding_probs = binding_results['binding_probabilities']
        top_indices = get_top_binding_sites(binding_probs, top_n_predictions, binding_threshold)
        
        print(f"Selected {len(top_indices)} binding site positions: {top_indices}")
        
        # Create pocket constraints for the input dictionary
        pocket_constraints = create_pocket_constraints(top_indices, protein.id, ligands[0].id if ligands else "B")
        
        # Add pocket constraints to input_dict
        if "constraints" not in input_dict:
            input_dict["constraints"] = []
        
        input_dict["constraints"].append({
            "pocket": pocket_constraints
        })
    else:
        print(f"Automatic pocket scanning disabled for {datapoint_id} - running without constraints")

    # Please note:
    # `protein` is a single-chain target protein sequence with id A
    # `ligands` contains a single small molecule ligand object with unknown binding sites
    # The binding site prediction results have been used to add pocket constraints to input_dict

    cli_args = ["--diffusion_samples", "10", "--sampling_steps", "300", "--use_potentials"]
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

def post_process_protein_ligand(datapoint: Datapoint, input_dicts: List[dict[str, Any]], cli_args_list: List[list[str]], prediction_dirs: List[Path], pocket_residues: Optional[Set[Tuple[str, str, int]]] = None, unconstrained_dir: Optional[Path] = None, is_allosteric: bool = False) -> List[Path]:
    """
    Return ranked model files for protein-ligand submission.
    Args:
        datapoint: The original datapoint object
        input_dicts: List of input dictionaries used for predictions (one per config)
        cli_args_list: List of command line arguments used for predictions (one per config)
        prediction_dirs: List of directories containing prediction results (one per config)
        pocket_residues: Optional set of orthosteric pocket residues
        unconstrained_dir: Optional directory containing unconstrained prediction models
        is_allosteric: If True, filter structures by ligand RMSD > 2A and pocket RMSD > 4A
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
    
    # For allosteric binders, filter structures based on ligand RMSD relative to unconstrained predictions
    if is_allosteric and unconstrained_dir and unconstrained_dir.exists() and len(all_pdbs) > 0:
        print(f"\n{'='*80}")
        print(f"Filtering ALLOSTERIC structures by ligand RMSD (relative to unconstrained)")
        print(f"{'='*80}\n")
        
        # Load unconstrained structures
        unconstrained_paths = [unconstrained_dir / f"model_{i}.pdb" for i in range(5)]
        unconstrained_structs = []
        
        for path in unconstrained_paths:
            if path.exists():
                try:
                    unconstrained_structs.append(_load_structure(path))
                except Exception as e:
                    print(f"WARNING: Could not load {path}: {e}")
        
        if len(unconstrained_structs) > 0:
            # Load all predicted structures
            pred_structs = []
            for pdb_path in all_pdbs:
                try:
                    pred_structs.append((pdb_path, _load_structure(pdb_path)))
                except Exception as e:
                    print(f"WARNING: Could not load {pdb_path}: {e}")
            
            if len(pred_structs) > 0:
                # Use first unconstrained structure as reference for alignment
                ref_struct = unconstrained_structs[0]
                
                # Superpose all unconstrained structures to the reference
                print(f"Aligning {len(unconstrained_structs)} unconstrained structures to reference...")
                for struct in unconstrained_structs[1:]:
                    _superpose_structure(ref_struct, struct)
                
                # Superpose all predicted structures to the same reference
                print(f"Aligning {len(pred_structs)} predicted structures to reference...")
                for _, struct in pred_structs:
                    _superpose_structure(ref_struct, struct)
                
                # Extract ligand coordinates from all unconstrained structures (after alignment)
                unconstrained_ligands = []
                for struct in unconstrained_structs:
                    lig_coords = _ligand_coords(struct)
                    if lig_coords:
                        unconstrained_ligands.append(lig_coords)
                
                if len(unconstrained_ligands) > 0:
                    # Calculate ligand RMSD for each predicted structure (using already-aligned structures)
                    print(f"Ligand RMSD analysis for allosteric predictions (vs {len(unconstrained_ligands)} unconstrained models):")
                    print(f"{'Model':<40} {'Min Lig RMSD (Å)':<18} {'Avg Lig RMSD (Å)':<18} {'Status':<20}")
                    print("-" * 100)
                    
                    filtered_pdbs = []
                    
                    for pdb_path, pred_struct in pred_structs:
                        try:
                            pred_lig = _ligand_coords(pred_struct)
                            
                            if pred_lig:
                                # Calculate RMSD to all unconstrained ligands (already aligned)
                                rmsds = []
                                for unc_lig in unconstrained_ligands:
                                    rmsd = _ligand_rmsd(pred_lig, unc_lig)
                                    if rmsd != float("inf"):
                                        rmsds.append(rmsd)
                                
                                if rmsds:
                                    min_rmsd = min(rmsds)
                                    avg_rmsd = np.mean(rmsds)
                                    
                                    # For allosteric binders, we want ligand RMSD > 2A (different binding site)
                                    if min_rmsd > 2.0:
                                        status = "PASS (> 2 Å)"
                                        filtered_pdbs.append(pdb_path)
                                    else:
                                        status = "FILTERED (≤ 2 Å)"
                                    
                                    print(f"{pdb_path.name:<40} {min_rmsd:>8.2f}             {avg_rmsd:>8.2f}             {status}")
                                else:
                                    # If can't calculate, keep it
                                    print(f"{pdb_path.name:<40} {'N/A':>8}             {'N/A':>8}             PASS (cannot calc)")
                                    filtered_pdbs.append(pdb_path)
                            else:
                                # No ligand found, keep it anyway
                                print(f"{pdb_path.name:<40} {'N/A':>8}             {'N/A':>8}             PASS (no ligand)")
                                filtered_pdbs.append(pdb_path)
                                
                        except Exception as e:
                            print(f"WARNING: Could not process {pdb_path}: {e}")
                            filtered_pdbs.append(pdb_path)  # Keep on error
                    
                    print("-" * 100)
                    print(f"Filtered structures (ligand RMSD > 2 Å from unconstrained): {len(filtered_pdbs)}/{len(all_pdbs)}")
                    
                    # If all structures filtered out, use unfiltered set
                    if len(filtered_pdbs) == 0:
                        print("WARNING: All structures filtered out by ligand RMSD. Using unfiltered set.")
                    else:
                        print(f"Using {len(filtered_pdbs)} structures that pass ligand RMSD > 2 Å filter")
                        all_pdbs = sorted(filtered_pdbs)
                    print()
                else:
                    print("WARNING: No ligands found in unconstrained structures. Skipping ligand RMSD filter.")
            else:
                print("WARNING: Could not load predicted structures. Skipping ligand RMSD filter.")
        else:
            print("WARNING: Could not load unconstrained structures. Skipping ligand RMSD filter.")
    
    # For allosteric binders, filter structures based on pocket RMSD
    if is_allosteric and pocket_residues and len(all_pdbs) > 0:
        print(f"\n{'='*80}")
        print(f"Filtering ALLOSTERIC structures by pocket RMSD")
        print(f"{'='*80}\n")
        
        # Load all predicted structures
        pred_structs = []
        for pdb_path in all_pdbs:
            try:
                pred_structs.append((pdb_path, _load_structure(pdb_path)))
            except Exception as e:
                print(f"WARNING: Could not load {pdb_path}: {e}")
        
        if len(pred_structs) >= 2:
            # Use first structure as reference
            ref_path, ref_struct = pred_structs[0]
            
            # Superpose all structures to reference
            for _, struct in pred_structs[1:]:
                _superpose_structure(ref_struct, struct)
            
            # Calculate pocket RMSD and filter
            print(f"Pocket RMSD analysis for allosteric predictions ({len(pocket_residues)} pocket residues):")
            print(f"{'Model':<40} {'Pocket RMSD (Å)':<18} {'Status':<20}")
            print("-" * 80)
            
            filtered_pdbs = []
            unfiltered_pdbs = []
            
            for pdb_path, struct in pred_structs:
                if pdb_path == ref_path:
                    rmsd = 0.0  # Reference to itself
                else:
                    rmsd = _calculate_pocket_rmsd_between_structures(ref_struct, struct, pocket_residues)
                
                if rmsd is not None:
                    # For allosteric binders, we want RMSD > 4A (far from orthosteric pocket)
                    if rmsd > 4.0:
                        status = "PASS (> 4 Å)"
                        filtered_pdbs.append(pdb_path)
                    else:
                        status = "FILTERED (≤ 4 Å)"
                    unfiltered_pdbs.append(pdb_path)
                    print(f"{pdb_path.name:<40} {rmsd:>8.2f}             {status}")
                else:
                    # If can't calculate, keep it
                    print(f"{pdb_path.name:<40} {'N/A':>8}             PASS (cannot calc)")
                    filtered_pdbs.append(pdb_path)
                    unfiltered_pdbs.append(pdb_path)
            
            print("-" * 80)
            print(f"Filtered structures (RMSD > 4 Å from orthosteric pocket): {len(filtered_pdbs)}/{len(pred_structs)}")
            
            # If all structures filtered out, use unfiltered set
            if len(filtered_pdbs) == 0:
                print("WARNING: All structures filtered out. Using unfiltered set.")
                all_pdbs = sorted(unfiltered_pdbs)
            else:
                print(f"Using {len(filtered_pdbs)} structures that pass RMSD > 4 Å filter")
                all_pdbs = sorted(filtered_pdbs)
            print()
    
    # Calculate pocket RMSD for unconstrained structures if available (for reference)
    if pocket_residues and unconstrained_dir and unconstrained_dir.exists():
        print(f"\n{'='*80}")
        print(f"Computing pocket RMSD for unconstrained structures (reference)")
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
    
    # Rank structures by ligand iPTM scores
    print(f"\n{'='*80}")
    print(f"Ranking structures by ligand iPTM scores")
    print(f"{'='*80}\n")
    
    # Extract ligand iPTM scores for each structure
    pdb_scores = []
    for pdb_path in all_pdbs:
        # Parse the PDB filename to extract config and model indices
        # Format: {datapoint_id}_config_{config_idx}_model_{model_idx}.pdb
        filename = pdb_path.stem  # Remove .pdb extension
        parts = filename.split("_")
        
        try:
            # Find config and model indices
            config_idx = None
            model_idx = None
            for i, part in enumerate(parts):
                if part == "config" and i + 1 < len(parts):
                    config_idx = parts[i + 1]
                elif part == "model" and i + 1 < len(parts):
                    model_idx = parts[i + 1]
            
            if config_idx is not None and model_idx is not None:
                # Find the prediction directory for this config
                pred_dir = None
                for prediction_dir in prediction_dirs:
                    if f"config_{config_idx}" in str(prediction_dir):
                        pred_dir = prediction_dir
                        break
                
                if pred_dir:
                    # Look for confidence file
                    confidence_file = pred_dir / f"confidence_{datapoint.datapoint_id}_config_{config_idx}_model_{model_idx}.json"
                    
                    if confidence_file.exists():
                        try:
                            with open(confidence_file) as f:
                                data = json.load(f)
                                ligand_iptm = data.get("ligand_iptm", None)
                                if ligand_iptm is not None:
                                    pdb_scores.append((pdb_path, ligand_iptm))
                                else:
                                    print(f"WARNING: No ligand_iptm found in {confidence_file.name}")
                                    pdb_scores.append((pdb_path, -1.0))  # Use -1 for missing scores
                        except Exception as e:
                            print(f"WARNING: Could not parse {confidence_file}: {e}")
                            pdb_scores.append((pdb_path, -1.0))
                    else:
                        print(f"WARNING: Confidence file not found: {confidence_file}")
                        pdb_scores.append((pdb_path, -1.0))
                else:
                    print(f"WARNING: Could not find prediction directory for config {config_idx}")
                    pdb_scores.append((pdb_path, -1.0))
            else:
                print(f"WARNING: Could not parse config/model indices from {pdb_path.name}")
                pdb_scores.append((pdb_path, -1.0))
        except Exception as e:
            print(f"WARNING: Error processing {pdb_path.name}: {e}")
            pdb_scores.append((pdb_path, -1.0))
    
    # Sort by ligand iPTM score (descending - higher is better)
    pdb_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Display ranking
    print(f"{'Rank':<8} {'Model':<40} {'Ligand iPTM':<15}")
    print("-" * 80)
    for rank, (pdb_path, score) in enumerate(pdb_scores, 1):
        if score >= 0:
            print(f"{rank:<8} {pdb_path.name:<40} {score:<15.4f}")
        else:
            print(f"{rank:<8} {pdb_path.name:<40} {'N/A':<15}")
    print("-" * 80)
    print(f"Ranked {len(pdb_scores)} structures by ligand iPTM score\n")
    
    # Return ranked PDB paths
    all_pdbs = [pdb_path for pdb_path, _ in pdb_scores]
    
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
ap.add_argument("--use-auto-pocket-scanner", action="store_true", default=True,
                help="Use automatic pocket scanning (binding site prediction) to add constraints (default: True)")

args = ap.parse_args()

def _prefill_input_dict(datapoint_id: str, proteins: Iterable[Protein], ligands: Optional[list[SmallMolecule]] = None, msa_dir: Optional[Path] = None, constraints: Optional[list] = None, pocket_residues: Optional[Set[Tuple[str, str, int]]] = None, is_allosteric: bool = False) -> dict:
    """
    Prepare input dict for Boltz YAML.
    
    Args:
        datapoint_id: Unique identifier for the datapoint
        proteins: Iterable of protein objects
        ligands: Optional list of small molecule ligands
        msa_dir: Directory containing MSA files
        constraints: Optional list of constraint dictionaries
        pocket_residues: Optional set of orthosteric pocket residues (chain_id, resname, residue_number)
                        Will be added as negative_pocket constraint if is_allosteric is True
        is_allosteric: If True, add negative_pocket constraint to avoid orthosteric pocket
    
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
    # Only add for allosteric binders to force binding away from orthosteric pocket
    if pocket_residues and ligands and is_allosteric:
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
        
        print(f"  Added negative_pocket constraint for ALLOSTERIC binding with {len(contacts)} residues (min_distance=10.0 Å)")
    
    # Add all constraints to the document
    if all_constraints:
        doc["constraints"] = all_constraints
    
    return doc

def _run_boltz_and_collect(datapoint) -> None:
    """
    New flow: prepare input dict, write yaml, run boltz, post-process, copy submissions.
    For protein_ligand: First run unconstrained predictions to identify orthosteric pocket
    and detect allosteric binding based on confidence variance.
    """
    out_dir = args.intermediate_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    subdir = args.submission_dir / datapoint.datapoint_id
    subdir.mkdir(parents=True, exist_ok=True)

    # For protein_ligand: Run unconstrained predictions first to identify pocket and detect allosteric binding
    pocket_residues = None
    is_allosteric = False
    variance_stats = None
    
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
        print(f"Stage 2: Identifying orthosteric pocket residues and detecting allosteric binding")
        print(f"{'='*80}\n")
        
        pocket_residues = identify_orthosteric_pocket(temp_pocket_dir)
        
        # Extract confidence metrics and determine if allosteric
        print(f"\nAnalyzing confidence variance and ligand similarity to detect allosteric binding...")
        metrics = extract_confidence_metrics(unconstrained_pred_dir, f"{datapoint.datapoint_id}_unconstrained")
        
        # Get ligand SMILES for similarity checking
        ligand_smiles = datapoint.ligands[0].smiles if datapoint.ligands else None
        
        if metrics["confidence_scores"] or metrics["ligand_iptm"] or metrics["protein_iptm"]:
            is_allosteric, detection_stats = is_allosteric_binder(metrics, ligand_smiles=ligand_smiles)
            
            print(f"\n{'='*60}")
            print(f"ALLOSTERIC DETECTION RESULTS")
            print(f"{'='*60}")
            print(f"\n1. Variance-Based Detection:")
            print(f"  Confidence Score Std Dev: {detection_stats.get('confidence_std', 0.0):.6f} (threshold: {CONFIDENCE_STD_THRESHOLD})")
            print(f"  Ligand iPTM Std Dev: {detection_stats.get('ligand_iptm_std', 0.0):.6f} (threshold: {LIGAND_IPTM_STD_THRESHOLD})")
            print(f"  Protein iPTM Std Dev: {detection_stats.get('protein_iptm_std', 0.0):.6f} (threshold: {IPTM_STD_THRESHOLD})")
            print(f"  Allosteric by variance: {'YES' if detection_stats.get('allosteric_by_variance') else 'NO'}")
            
            print(f"\n2. Ligand Similarity-Based Detection:")
            if ligand_smiles and RDKIT_AVAILABLE:
                print(f"  Max similarity to allosteric ligands: {detection_stats.get('max_ligand_similarity', 0.0):.3f} (threshold: {LIGAND_SIMILARITY_THRESHOLD})")
                print(f"  Allosteric by similarity: {'YES' if detection_stats.get('allosteric_by_similarity') else 'NO'}")
                if detection_stats.get('allosteric_by_similarity'):
                    print(f"  Most similar allosteric ligand: {detection_stats.get('most_similar_allosteric_ligand', 'N/A')[:50]}...")
            else:
                print(f"  Ligand similarity checking: DISABLED (RDKit not available or no ligand)")
            
            print(f"\n{'='*60}")
            print(f"FINAL CLASSIFICATION: {'ALLOSTERIC' if is_allosteric else 'ORTHOSTERIC'}")
            if is_allosteric:
                methods = detection_stats.get('detection_method', [])
                print(f"Detection method(s): {', '.join(methods).upper()}")
            print(f"{'='*60}")
            
            # Save detection analysis to file
            detection_file = subdir / "allosteric_detection.json"
            with open(detection_file, "w") as f:
                json.dump({
                    "is_allosteric": is_allosteric,
                    "detection_stats": detection_stats,
                    "metrics": metrics,
                    "ligand_smiles": ligand_smiles,
                    "thresholds": {
                        "confidence_std": CONFIDENCE_STD_THRESHOLD,
                        "ligand_iptm_std": LIGAND_IPTM_STD_THRESHOLD,
                        "protein_iptm_std": IPTM_STD_THRESHOLD,
                        "ligand_similarity": LIGAND_SIMILARITY_THRESHOLD
                    }
                }, f, indent=2)
            print(f"\nAllosteric detection analysis saved to: {detection_file}")
        else:
            print("WARNING: Could not extract confidence metrics from predictions")
            is_allosteric = False
        
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
        print(f"Stage 3: Running final predictions with {'negative pocket constraints (allosteric)' if is_allosteric else 'identified pocket info'}")
        print(f"{'='*80}\n")

    # Prepare input dict and CLI args for final predictions
    # Pass pocket_residues and is_allosteric flag to _prefill_input_dict
    base_input_dict = _prefill_input_dict(
        datapoint.datapoint_id, 
        datapoint.proteins, 
        datapoint.ligands, 
        args.msa_dir, 
        datapoint.constraints,
        pocket_residues=pocket_residues,  # Pass identified pocket residues
        is_allosteric=is_allosteric  # Pass allosteric detection flag
    )

    if datapoint.task_type == "protein_complex":
        configs = prepare_protein_complex(datapoint.datapoint_id, datapoint.proteins, base_input_dict, args.msa_dir)
    elif datapoint.task_type == "protein_ligand":
        configs = prepare_protein_ligand(datapoint.datapoint_id, datapoint.proteins[0], datapoint.ligands, base_input_dict, args.msa_dir, is_allosteric=is_allosteric)
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
        
        # Write YAML with custom formatting for contacts
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
            f.write("version: 1\n")
            f.write("sequences:\n")
            
            # Write protein sequence
            f.write("- protein:\n")
            f.write(f"    id: {input_dict['sequences'][0]['protein']['id']}\n")
            f.write(f"    sequence: {input_dict['sequences'][0]['protein']['sequence']}\n")
            f.write(f"    msa: {input_dict['sequences'][0]['protein']['msa']}\n")
            
            # Write ligand sequence
            f.write("- ligand:\n")
            f.write(f"    id: {input_dict['sequences'][1]['ligand']['id']}\n")
            f.write(f"    smiles: {input_dict['sequences'][1]['ligand']['smiles']}\n")
            
            # Write constraints (if any)
            if 'constraints' in input_dict and input_dict['constraints']:
                f.write("constraints:\n")
                for constraint in input_dict['constraints']:
                    if 'pocket' in constraint:
                        f.write("- pocket:\n")
                        f.write(f"    binder: {constraint['pocket']['binder']}\n")
                        f.write("    contacts: [")
                        contacts = constraint['pocket']['contacts']
                        for i, contact in enumerate(contacts):
                            if i > 0:
                                f.write(", ")
                            f.write(f"[ {contact[0]}, {contact[1]} ]")
                        f.write(" ]\n")
                        f.write(f"    max_distance: {constraint['pocket']['max_distance']}\n")
                        f.write(f"    force: {str(constraint['pocket']['force']).lower()}\n")

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
        # Pass pocket residues, unconstrained prediction directory, and allosteric flag for RMSD analysis
        unconstrained_dir = subdir / "unconstrained_for_pocket" if pocket_residues else None
        ranked_files = post_process_protein_ligand(
            datapoint, all_input_dicts, all_cli_args, all_pred_subfolders, 
            pocket_residues=pocket_residues, unconstrained_dir=unconstrained_dir,
            is_allosteric=is_allosteric
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
