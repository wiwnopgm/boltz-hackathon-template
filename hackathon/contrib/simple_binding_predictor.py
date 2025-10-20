"""
Simple wrapper for protein-ligand binding site prediction.
This provides a clean interface for the hackathon integration.
"""

import os
import tempfile
from pathlib import Path
from predict_binding_sites_function import predict_binding_sites_from_text


def predict_binding_sites(protein_sequence: str, smiles: str, output_dir: str = "./binding_output/") -> dict:
    """
    Simple function to predict protein-ligand binding sites.
    
    Args:
        protein_sequence (str): Protein sequence as a string
        smiles (str): SMILES string for the ligand
        output_dir (str): Output directory for results (default: "./binding_output/")
        
    Returns:
        dict: Dictionary containing:
            - 'binding_probabilities': List of binding probabilities per residue
            - 'protein_sequence': Input protein sequence
            - 'ligand_smiles': Input SMILES string
            - 'result_csv_path': Path to the generated CSV file
            - 'pocket_pdb_path': Path to pocket PDB file (if available)
    """
    
    # Generate simple IDs
    protein_id = "protein_1"
    ligand_id = "ligand_1"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Run prediction
    results = predict_binding_sites_from_text(
        protein_sequence=protein_sequence,
        protein_id=protein_id,
        smiles=smiles,
        ligand_id=ligand_id,
        output_dir=output_dir,
        batch_size=1,
        device_ids=[0],
        cluster=True
    )
    
    return results


def predict_binding_sites_batch(protein_ligand_pairs: list, output_dir: str = "./binding_output/") -> list:
    """
    Predict binding sites for multiple protein-ligand pairs.
    
    Args:
        protein_ligand_pairs (list): List of tuples (protein_sequence, smiles)
        output_dir (str): Output directory for results
        
    Returns:
        list: List of result dictionaries for each pair
    """
    results = []
    
    for i, (protein_sequence, smiles) in enumerate(protein_ligand_pairs):
        # Create subdirectory for each pair
        pair_output_dir = os.path.join(output_dir, f"pair_{i+1}")
        
        result = predict_binding_sites(protein_sequence, smiles, pair_output_dir)
        results.append(result)
    
    return results


# Example usage
if __name__ == "__main__":
    # Example protein sequence (insulin)
    protein_seq = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
    
    # Example SMILES (glucose)
    smiles_str = "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O"
    
    # Run prediction
    result = predict_binding_sites(protein_seq, smiles_str)
    
    print("Binding Site Prediction Results:")
    print(f"Protein length: {len(result['protein_sequence'])}")
    print(f"Ligand SMILES: {result['ligand_smiles']}")
    print(f"Number of binding probabilities: {len(result['binding_probabilities'])}")
    print(f"First 10 binding probabilities: {result['binding_probabilities'][:10]}")
    print(f"Results saved to: {result['result_csv_path']}")
