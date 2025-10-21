# predict_hackathon.py
import yaml
import sys
import argparse
import json
import glob
import os
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, List, Optional

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
    cli_args = ["--diffusion_samples", "5"]
    return [(input_dict, cli_args)]

def prepare_protein_ligand(datapoint_id: str, protein: Protein, ligands: list[SmallMolecule], input_dict: dict, msa_dir: Optional[Path] = None) -> List[tuple[dict, List[str]]]:
    """
    Prepare input dict and CLI args for a protein-ligand prediction.
    You can return multiple configurations to run by returning a list of (input_dict, cli_args) tuples.
    Args:
        datapoint_id: The unique identifier for this datapoint
        protein: The protein sequence
        ligands: A list of a single small molecule ligand object 
        input_dict: Prefilled input dict
        msa_dir: Directory containing MSA files (for computing relative paths)
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

def post_process_protein_ligand(datapoint: Datapoint, input_dicts: List[dict[str, Any]], cli_args_list: List[list[str]], prediction_dirs: List[Path]) -> List[Path]:
    """
    Return ranked model files for protein-ligand submission.
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
ap.add_argument("--use-auto-pocket-scanner", action="store_true", 
                help="Use automatic pocket scanning (binding site prediction) to add constraints")

args = ap.parse_args()

def _prefill_input_dict(datapoint_id: str, proteins: Iterable[Protein], ligands: Optional[list[SmallMolecule]] = None, msa_dir: Optional[Path] = None) -> dict:
    """
    Prepare input dict for Boltz YAML.
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
    return doc

def _run_boltz_and_collect(datapoint) -> None:
    """
    New flow: prepare input dict, write yaml, run boltz, post-process, copy submissions.
    """
    out_dir = args.intermediate_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    subdir = args.submission_dir / datapoint.datapoint_id
    subdir.mkdir(parents=True, exist_ok=True)

    # Prepare input dict and CLI args
    base_input_dict = _prefill_input_dict(datapoint.datapoint_id, datapoint.proteins, datapoint.ligands, args.msa_dir)

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
        
        # Write YAML with custom formatting for contacts
        with open(yaml_path, "w") as f:
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
        ranked_files = post_process_protein_ligand(datapoint, all_input_dicts, all_cli_args, all_pred_subfolders)
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
