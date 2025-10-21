#!/usr/bin/env python3
"""
Validate negative pocket constraints for Boltz predictions using JSONL config.
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple


def parse_pdb(pdb_file: str) -> Tuple[Dict[Tuple[str, int], List[np.ndarray]], Dict[str, List[np.ndarray]]]:
    """Parse PDB file to extract protein residue atoms and ligand atoms."""
    residue_atoms = {}
    ligand_atoms = {}
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                chain = line[21:22].strip()
                resnum = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                
                key = (chain, resnum)
                if key not in residue_atoms:
                    residue_atoms[key] = []
                residue_atoms[key].append(np.array([x, y, z]))
                
            elif line.startswith('HETATM'):
                chain = line[21:22].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                
                if chain not in ligand_atoms:
                    ligand_atoms[chain] = []
                ligand_atoms[chain].append(np.array([x, y, z]))
    
    return residue_atoms, ligand_atoms


def calculate_min_distance(ligand_coords: List[np.ndarray], pocket_coords: List[np.ndarray]) -> float:
    """Calculate minimum distance between any ligand atom and any pocket atom."""
    min_dist = float('inf')
    
    for ligand_atom in ligand_coords:
        for pocket_atom in pocket_coords:
            dist = np.linalg.norm(ligand_atom - pocket_atom)
            min_dist = min(min_dist, dist)
    
    return min_dist


def validate_prediction(pdb_file: str, datapoint: Dict, verbose: bool = False) -> Dict:
    """Validate negative pocket constraint for a single prediction."""
    datapoint_id = datapoint['datapoint_id']
    
    # Extract negative pocket constraints
    negative_pocket = None
    if 'constraints' in datapoint:
        for constraint in datapoint['constraints']:
            if 'negative_pocket' in constraint:
                negative_pocket = constraint['negative_pocket']
                break
    
    if negative_pocket is None:
        return {
            'datapoint_id': datapoint_id,
            'status': 'no_constraint',
            'message': 'No negative_pocket constraint found'
        }
    
    binder_chain = negative_pocket['binder']
    contacts = negative_pocket['contacts']
    min_distance = negative_pocket['min_distance']
    
    # Parse PDB
    try:
        residue_atoms, ligand_atoms = parse_pdb(pdb_file)
    except Exception as e:
        return {
            'datapoint_id': datapoint_id,
            'status': 'error',
            'message': f'Failed to parse PDB: {str(e)}'
        }
    
    if binder_chain not in ligand_atoms:
        return {
            'datapoint_id': datapoint_id,
            'status': 'error',
            'message': f'Binder chain {binder_chain} not found in PDB'
        }
    
    ligand_coords = ligand_atoms[binder_chain]
    
    # Collect pocket atoms
    pocket_coords = []
    missing_residues = []
    
    for contact in contacts:
        chain, resnum = contact
        key = (chain, resnum)
        if key in residue_atoms:
            pocket_coords.extend(residue_atoms[key])
        else:
            missing_residues.append(f"{chain}:{resnum}")
    
    if not pocket_coords:
        return {
            'datapoint_id': datapoint_id,
            'status': 'error',
            'message': 'No pocket residue atoms found in PDB'
        }
    
    # Calculate minimum distance
    min_dist = calculate_min_distance(ligand_coords, pocket_coords)
    
    # Check constraint satisfaction
    constraint_satisfied = min_dist >= min_distance
    
    # Calculate per-residue distances
    residue_distances = {}
    for contact in contacts:
        chain, resnum = contact
        key = (chain, resnum)
        if key in residue_atoms:
            res_coords = residue_atoms[key]
            res_min_dist = calculate_min_distance(ligand_coords, res_coords)
            residue_distances[f"{chain}:{resnum}"] = res_min_dist
    
    num_violations = len([d for d in residue_distances.values() if d < min_distance])
    
    if verbose:
        status_symbol = '✓' if constraint_satisfied else '✗'
        print(f"{status_symbol} {datapoint_id}: ", end='')
        if constraint_satisfied:
            print(f"PASSED (min: {min_dist:.2f} Å, margin: {min_dist - min_distance:.2f} Å)")
        else:
            print(f"FAILED (min: {min_dist:.2f} Å, deficit: {min_distance - min_dist:.2f} Å, {num_violations} violations)")
            if verbose > 1:  # Extra verbose
                violations = [(res, dist) for res, dist in residue_distances.items() if dist < min_distance]
                for res, dist in sorted(violations, key=lambda x: x[1])[:5]:
                    print(f"    {res}: {dist:.2f} Å (deficit: {min_distance - dist:.2f} Å)")
    
    return {
        'datapoint_id': datapoint_id,
        'pdb_file': pdb_file,
        'status': 'success',
        'constraint_satisfied': constraint_satisfied,
        'min_distance_found': min_dist,
        'min_distance_required': min_distance,
        'deficit': max(0, min_distance - min_dist),
        'safety_margin': max(0, min_dist - min_distance),
        'num_violations': num_violations,
        'num_pocket_residues': len(contacts),
        'num_missing_residues': len(missing_residues),
        'num_ligand_atoms': len(ligand_coords),
        'residue_distances': residue_distances
    }


def batch_validate(
    predictions_dir: str,
    jsonl_file: str,
    output_csv: str = None,
    verbose: int = 0
):
    """Validate negative pocket constraints for all predictions."""
    predictions_path = Path(predictions_dir)
    
    # Load JSONL
    datapoints = {}
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                datapoints[data['datapoint_id']] = data
    
    print(f"Loaded {len(datapoints)} datapoints from {jsonl_file}")
    print(f"Searching for predictions in {predictions_dir}\n")
    print(f"{'='*100}")
    
    results = []
    
    # Find and validate all predictions
    for structure_dir in sorted(predictions_path.iterdir()):
        if not structure_dir.is_dir():
            continue
        
        structure_name = structure_dir.name
        
        if structure_name not in datapoints:
            if verbose:
                print(f"⊘ {structure_name}: No matching datapoint in JSONL")
            continue
        
        datapoint = datapoints[structure_name]
        
        # Process each model
        for model_file in sorted(structure_dir.glob('model_*.pdb')):
            model_name = f"{structure_name}/{model_file.name}"
            
            result = validate_prediction(str(model_file), datapoint, verbose=verbose)
            
            if result['status'] == 'success':
                result['model_name'] = model_name
                result['model_file'] = str(model_file)
                results.append(result)
    
    if not results:
        print("\nNo valid results to report.")
        return None
    
    # Summary statistics
    print(f"\n{'='*100}")
    print(f"SUMMARY")
    print(f"{'='*100}")
    
    total = len(results)
    passed = sum(1 for r in results if r['constraint_satisfied'])
    failed = total - passed
    
    avg_distance = sum(r['min_distance_found'] for r in results) / total
    avg_required = sum(r['min_distance_required'] for r in results) / total
    
    print(f"Total predictions validated: {total}")
    print(f"Passed: {passed} ({100*passed/total:.1f}%)")
    print(f"Failed: {failed} ({100*failed/total:.1f}%)")
    print(f"Average minimum distance found: {avg_distance:.2f} Å")
    print(f"Average required distance: {avg_required:.2f} Å")
    
    if failed > 0:
        avg_deficit_failed = sum(r['deficit'] for r in results if not r['constraint_satisfied']) / failed
        print(f"Average deficit (failed): {avg_deficit_failed:.2f} Å")
        
        # Show worst violations
        worst = sorted([r for r in results if not r['constraint_satisfied']], 
                      key=lambda x: x['deficit'], reverse=True)[:5]
        print(f"\nWorst violations:")
        for r in worst:
            print(f"  {r['model_name']}: {r['min_distance_found']:.2f} Å (deficit: {r['deficit']:.2f} Å, {r['num_violations']} residues)")
    
    if passed > 0:
        avg_margin_passed = sum(r['safety_margin'] for r in results if r['constraint_satisfied']) / passed
        print(f"Average safety margin (passed): {avg_margin_passed:.2f} Å")
    
    # Save results
    if output_csv:
        df = pd.DataFrame([{
            'model_name': r['model_name'],
            'datapoint_id': r['datapoint_id'],
            'constraint_satisfied': r['constraint_satisfied'],
            'min_distance_found': r['min_distance_found'],
            'min_distance_required': r['min_distance_required'],
            'deficit': r['deficit'],
            'safety_margin': r['safety_margin'],
            'num_violations': r['num_violations'],
            'num_pocket_residues': r['num_pocket_residues'],
            'num_ligand_atoms': r['num_ligand_atoms']
        } for r in results])
        
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to: {output_csv}")
    
    print(f"{'='*100}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Validate negative pocket constraints from JSONL config',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate all predictions
  python validate_negative_pocket_from_jsonl.py \\
    --predictions test_results/test_negative_pocket_v2 \\
    --jsonl hackathon_data/datasets/asos_public/asos_public_test1.jsonl \\
    --output validation_results.csv
  
  # With verbose output
  python validate_negative_pocket_from_jsonl.py \\
    --predictions test_results/test_negative_pocket_v2 \\
    --jsonl hackathon_data/datasets/asos_public/asos_public_test1.jsonl \\
    --output validation_results.csv \\
    -vv
        """
    )
    
    parser.add_argument('--predictions', required=True,
                       help='Directory containing prediction subdirectories with model_*.pdb files')
    parser.add_argument('--jsonl', required=True,
                       help='JSONL file with datapoint configurations')
    parser.add_argument('--output', '--output-csv', dest='output_csv',
                       help='Output CSV file for results')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                       help='Increase verbosity (can be repeated: -v, -vv)')
    
    args = parser.parse_args()
    
    batch_validate(
        args.predictions,
        args.jsonl,
        output_csv=args.output_csv,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()



