"""
Refactored version of predict_binding_sites.py that provides a function-based interface
for predicting protein-ligand binding sites using text inputs instead of file paths.
"""

import gc
import os
import shutil
import pickle as pkl
import pandas as pd
import torch
import tempfile
from pathlib import Path
from tqdm import tqdm
from scipy.spatial import cKDTree
from Bio.PDB.DSSP import DSSP
from Bio import SeqIO
from lxml import etree
from Bio.PDB.ResidueDepth import get_surface
from model import LABind
from readData import LoadData
from config import nn_config, pretrain_path
from torch.utils.data import DataLoader
from torch import nn
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, EsmForProteinFolding
from Bio.PDB import PDBParser
from download_weights import download_all_weights
from ast import literal_eval
from utils import *
import numpy as np


def predict_binding_sites_from_text(
    protein_sequence: str,
    protein_id: str,
    smiles: str,
    ligand_id: str,
    output_dir: str = "./output/",
    batch_size: int = 1,
    device_ids: list = [0],
    cluster: bool = False,
    input_pdb_path: str = None,
    use_boltz: bool = False,
) -> dict:
    """
    Predict protein-ligand binding sites using text inputs instead of file paths.
    
    Args:
        protein_sequence (str): Protein sequence as a string
        protein_id (str): Identifier for the protein
        smiles (str): SMILES string for the ligand
        ligand_id (str): Identifier for the ligand
        output_dir (str): Output directory for results
        batch_size (int): Batch size for prediction
        device_ids (list): GPU device IDs to use
        cluster (bool): Whether to perform clustering on residues
        input_pdb_path (str): Optional path to existing PDB file
        
    Returns:
        dict: Dictionary containing prediction results with keys:
            - 'binding_probabilities': List of binding probabilities per residue
            - 'protein_sequence': Input protein sequence
            - 'ligand_smiles': Input SMILES string
            - 'result_csv_path': Path to the generated CSV file
            - 'pocket_pdb_path': Path to pocket PDB file (if clustering enabled)
    """
    
    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Create necessary subdirectories
        pdb_dir = temp_dir / "pdb"
        ankh_dir = temp_dir / "ankh"
        dssp_dir = temp_dir / "dssp"
        pos_dir = temp_dir / "pos"
        
        for dir_path in [pdb_dir, ankh_dir, dssp_dir, pos_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create temporary FASTA file with protein_id and ligand_id in header
        fasta_file = temp_dir / "protein.fa"
        with open(fasta_file, 'w') as f:
            f.write(f">{protein_id} {ligand_id}\n{protein_sequence}\n")
        
        # Create temporary SMILES file
        smiles_file = temp_dir / "smiles.txt"
        with open(smiles_file, 'w') as f:
            f.write(f"{ligand_id} {smiles}\n")
        
        # Download weights if needed
        download_all_weights(pretrain_path=pretrain_path)
        
        # Set up device
        run_device = f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu'
        
        # Handle existing PDB file if provided
        if input_pdb_path and os.path.exists(input_pdb_path):
            shutil.copy(input_pdb_path, pdb_dir / f"{protein_id}.pdb")
        
        # Generate PDB structure using ESMFold
        _generate_pdb_structure(fasta_file, pdb_dir, run_device, use_boltz)
        
        # Extract features
        _extract_dssp_features(pdb_dir, dssp_dir)
        _extract_msms_features(pdb_dir, pos_dir)
        _extract_ankh_embeddings(fasta_file, ankh_dir, run_device)
        _extract_mol_embeddings(fasta_file, smiles_file, temp_dir, run_device)
        
        # Run prediction
        results = _run_prediction(
            fasta_file, 
            temp_dir, 
            batch_size, 
            device_ids, 
            run_device
        )
        
        # Perform clustering if requested
        if cluster:
            _cluster_residues(temp_dir)
        
        # Copy results to output directory
        os.makedirs(output_dir, exist_ok=True)
        result_csv = shutil.copy(temp_dir / "RESULT.csv", output_dir)
        
        pocket_pdb = None
        if cluster and os.path.exists(temp_dir / "site_centers.csv"):
            shutil.copy(temp_dir / "site_centers.csv", output_dir)
            pocket_pdb = os.path.join(output_dir, f"{protein_id}_{ligand_id}.pdb")
            if os.path.exists(temp_dir / "pocket" / f"{protein_id}_{ligand_id}.pdb"):
                shutil.copy(temp_dir / "pocket" / f"{protein_id}_{ligand_id}.pdb", pocket_pdb)
        
        # Parse results
        df = pd.read_csv(result_csv)
        binding_probs = df[df['Protein Name'] == protein_id]['Binding Site Probability'].iloc[0]
        if isinstance(binding_probs, str):
            binding_probs = literal_eval(binding_probs)
        
        return {
            'binding_probabilities': binding_probs,
            'protein_sequence': protein_sequence,
            'ligand_smiles': smiles,
            'result_csv_path': result_csv,
            'pocket_pdb_path': pocket_pdb
        }


def _generate_pdb_structure(fasta_file, pdb_dir, device, use_boltz: bool = False):
    """Generate PDB structure using ESMFold."""
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    

    if use_boltz:
        pass
    else:
        # Load ESMFold model
        fold_path = pretrain_path['esmfold_path']
        tokenizer = AutoTokenizer.from_pretrained(fold_path)
        model = EsmForProteinFolding.from_pretrained(fold_path, low_cpu_mem_usage=True)
        model = model.eval()
        model.esm = model.esm.half()
        model = model.to(device)
        
        for record in tqdm(sequences, desc='ESMFold running', ncols=80, unit='proteins'):
            if os.path.exists(pdb_dir / f"{record.id}.pdb"):
                continue
            
            tokenized_input = tokenizer([str(record.seq)], return_tensors="pt", add_special_tokens=False)['input_ids']
            tokenized_input = tokenized_input.to(device)
            
            with torch.no_grad():
                output = model(tokenized_input)
            
            pdb = convert_outputs_to_pdb(output)
            with open(pdb_dir / f"{record.id}.pdb", "w") as f:
                f.write(''.join(pdb))
        
        del model
        gc.collect()


def _extract_dssp_features(pdb_dir, dssp_dir):
    """Extract DSSP features from PDB files."""
    mapSS = {' ':[0,0,0,0,0,0,0,0,0],
             '-':[1,0,0,0,0,0,0,0,0],
             'H':[0,1,0,0,0,0,0,0,0],
             'B':[0,0,1,0,0,0,0,0,0],
             'E':[0,0,0,1,0,0,0,0,0],
             'G':[0,0,0,0,1,0,0,0,0],
             'I':[0,0,0,0,0,1,0,0,0],
             'P':[0,0,0,0,0,0,1,0,0],
             'T':[0,0,0,0,0,0,0,1,0],
             'S':[0,0,0,0,0,0,0,0,1]}
    
    p = PDBParser(QUIET=True)
    # Get the absolute path to DSSP
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dssp_path = os.path.join(current_dir, 'checkpoints', 'mkdssp')
    
    for pdb_file_name in tqdm(os.listdir(pdb_dir), desc='DSSP running', ncols=80, unit='proteins'):
        pdb_file = pdb_dir / pdb_file_name
        save_file = dssp_dir / pdb_file_name.replace('.pdb', '.npy')
        
        if os.path.exists(save_file):
            continue
        
        structure = p.get_structure("tmp", str(pdb_file))
        model = structure[0]
        
        try:
            dssp = DSSP(model, str(pdb_file), dssp=dssp_path)
            keys = list(dssp.keys())
        except:
            keys = []
        
        res_np = []
        for chain in model:
            for residue in chain:
                res_key = (chain.id,(' ', residue.id[1], residue.id[2]))
                if res_key in keys:
                    tuple_dssp = dssp[res_key]
                    res_np.append(mapSS[tuple_dssp[2]] + list(tuple_dssp[3:]))
                else:
                    res_np.append(np.zeros(20))
        
        res_data = np.array(res_np)
        if res_data.dtype == '<U32':
            res_data = np.where(res_data == 'NA', 0, res_data).astype(np.float32)
        np.save(save_file, np.array(res_np))


def _extract_msms_features(pdb_dir, pos_dir):
    """Extract MSMS features from PDB files."""
    # Get the absolute path to MSMS
    current_dir = os.path.dirname(os.path.abspath(__file__))
    msms_path = os.path.join(current_dir, 'checkpoints', 'msms')
    
    for pdb_file_name in tqdm(os.listdir(pdb_dir), desc='MSMS running', ncols=80, unit='proteins'):
        pdb_file = pdb_dir / pdb_file_name
        save_file = pos_dir / pdb_file_name.replace('.pdb', '.npy')
        
        if os.path.exists(save_file):
            continue
        
        parser = PDBParser(QUIET=True)
        X = []
        chain_atom = ['N', 'CA', 'C', 'O']
        model = parser.get_structure('model', str(pdb_file))[0]
        chain = next(model.get_chains())
        
        try:
            surf = get_surface(chain, MSMS=msms_path)
            surf_tree = cKDTree(surf)
        except:
            surf = np.empty(0)
            surf_tree = None
        
        for residue in chain:
            line = []
            atoms_coord = np.array([atom.get_coord() for atom in residue])
            if surf.size > 0 and surf_tree is not None:
                dist, _ = surf_tree.query(atoms_coord)
                closest_atom = np.argmin(dist)
                closest_pos = atoms_coord[closest_atom]
            else:
                closest_pos = atoms_coord[-1]
            
            atoms = list(residue.get_atoms())
            ca_pos = residue['CA'].get_coord() if 'CA' in residue else residue.child_list[0].get_coord()
            pos_s = 0
            un_s = 0
            
            for atom in atoms:
                if atom.name in chain_atom:
                    line.append(atom.get_coord())
                else:
                    pos_s += calMass(atom, True)
                    un_s += calMass(atom, False)
            
            if len(line) != 4:
                line = line + [list(ca_pos)] * (4 - len(line))
            
            if un_s == 0:
                R_pos = ca_pos
            else:
                R_pos = pos_s / un_s
            
            line.append(R_pos)
            line.append(closest_pos)
            X.append(line)
        
        np.save(save_file, X)


def _extract_ankh_embeddings(fasta_file, ankh_dir, device):
    """Extract Ankh embeddings from protein sequences."""
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    
    if len(sequences) == len(os.listdir(ankh_dir)):
        print('Ankh embeddings already exist, skipping.')
        return
    
    embed_path = pretrain_path['ankh_path']
    tokenizer = AutoTokenizer.from_pretrained(embed_path)
    model = T5EncoderModel.from_pretrained(embed_path)
    model.to(device)
    model.eval()
    
    for record in tqdm(sequences, desc='Ankh running', ncols=80, unit='proteins'):
        if os.path.exists(ankh_dir / f'{record.id}.npy'):
            continue
        
        ids = tokenizer.batch_encode_plus([list(record.seq)], add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")
        input_ids = ids['input_ids'].to(device)
        attention_mask = ids['attention_mask'].to(device)
        
        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
            emb = embedding_repr.last_hidden_state[0, :len(record.seq)].cpu().numpy()
            np.save(ankh_dir / f'{record.id}.npy', emb)
    
    del model
    gc.collect()


def _extract_mol_embeddings(fasta_file, smiles_file, temp_dir, device):
    """Extract molecular embeddings from SMILES."""
    with open(smiles_file, 'r') as f:
        smiles_dict = {line.split()[0]: line.split()[1] for line in f}
    
    mol_path = pretrain_path['molformer_path']
    model = AutoModel.from_pretrained(mol_path, deterministic_eval=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(mol_path, trust_remote_code=True)
    model.to(device)
    
    res_dict = {}
    for smiles_name in tqdm(smiles_dict, desc='MolFormer running', ncols=80, unit='molecules'):
        smiles = smiles_dict[smiles_name]
        with torch.no_grad():
            inputs = tokenizer(smiles, padding=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            pooler = outputs.pooler_output.cpu().numpy()
            res_dict[smiles_name] = pooler
    
    # Save ligand embeddings
    with open(temp_dir / 'ligand.pkl', 'wb') as f:
        pkl.dump(res_dict, f)
    
    del model
    gc.collect()


def _run_prediction(fasta_file, temp_dir, batch_size, device_ids, run_device):
    """Run the main prediction using the LABind model."""
    # Load models
    models = []
    model_path = pretrain_path['model_path']
    
    for fold in range(5):
        state_dict = torch.load(model_path + f'fold{fold}.ckpt', run_device)
        model = LABind(
            rfeat_dim=nn_config['rfeat_dim'], 
            ligand_dim=nn_config['ligand_dim'], 
            hidden_dim=nn_config['hidden_dim'], 
            heads=nn_config['heads'], 
            augment_eps=nn_config['augment_eps'], 
            rbf_num=nn_config['rbf_num'],
            top_k=nn_config['top_k'], 
            attn_drop=nn_config['attn_drop'], 
            dropout=nn_config['dropout'], 
            num_layers=nn_config['num_layers']
        ).to(run_device)
        model = nn.DataParallel(model, device_ids=device_ids)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    
    # Load representations
    with open(temp_dir / 'ligand.pkl', 'rb') as f:
        ligand_dict = pkl.load(f)
    
    # Get the absolute path to repr.pkl
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repr_path = os.path.join(current_dir, 'checkpoints', 'repr.pkl')
    with open(repr_path, 'rb') as f:
        repr_dict = pkl.load(f)
    
    # Load data
    fa_dict = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
    with open(temp_dir / 'smiles.txt', 'r') as f:
        smiles_dict = {line.split()[0]: line.split()[1] for line in f}
    
    test_list = readDataList(fasta_file)
    test_data = LoadData(
        name_list=test_list,
        proj_dir=str(temp_dir),
        lig_dict=ligand_dict,
        repr_dict=repr_dict
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=test_data.collate_fn, shuffle=False, drop_last=False)
    
    # Run prediction
    df = pd.DataFrame(columns=['Protein Name', 'Protein Sequence', 'Ligand Name', 'Ligand SMILES', 'Binding Site Probability'])
    
    with torch.no_grad():
        for names, ligs, rfeat, ligand, xyz, mask in tqdm(test_loader, desc='LABind running', ncols=80, unit='batches'):
            tensors = [rfeat, ligand, xyz, mask]
            tensors = [tensor.to(run_device) for tensor in tensors]
            rfeat, ligand, xyz, mask = tensors
            
            logits = [model(rfeat, ligand, xyz, mask).sigmoid() for model in models]
            logits = torch.stack(logits, 0).mean(0)
            logits = logits.half()
            logits = logits.cpu().detach().numpy()
            
            for idx, logit in enumerate(logits):
                df = pd.concat([df, pd.DataFrame([[
                    names[idx], 
                    str(fa_dict[names[idx]].seq), 
                    ligs[idx], 
                    smiles_dict[ligs[idx]], 
                    list(logit[:mask[idx].sum()])
                ]], columns=df.columns)])
    
    # Save results
    df.to_csv(temp_dir / 'RESULT.csv', index=False)
    return df


def _cluster_residues(temp_dir):
    """Perform clustering on residues to identify binding site centers."""
    from sklearn.cluster import MeanShift
    from Bio.PDB import PDBIO, Select
    
    class PocketSelect(Select):
        def __init__(self, pocket):
            self.pocket = pocket
        def accept_residue(self, residue):
            return residue.get_id() in self.pocket
    
    ms = MeanShift(bandwidth=12.0)
    parser = PDBParser(QUIET=1)
    io = PDBIO()
    
    pred_site_df = pd.read_csv(temp_dir / "RESULT.csv", na_filter=False, converters={"Binding Site Probability": literal_eval})
    pdb_path = temp_dir / "pdb"
    pkt_path = temp_dir / "pocket"
    pkt_path.mkdir(exist_ok=True)
    
    site_center_df = pd.DataFrame(columns=['Protein Name', 'Ligand Name', 'Binding Site Center'])
    
    for idx, row in tqdm(pred_site_df.iterrows(), desc='Clustering residues', ncols=80, unit='proteins'):
        prot_name = row['Protein Name']
        ligd_name = row['Ligand Name']
        bind_resi = row["Binding Site Probability"]
        bind_resi = [1 if resi > 0.48 else 0 for resi in bind_resi]
        
        pdb_structure = parser.get_structure(prot_name, pdb_path / f"{prot_name}.pdb")
        residues = list(pdb_structure.get_residues())
        pocket = set()
        
        for idx, res in enumerate(residues): 
            if bind_resi[idx] == 1: 
                pocket.add(res.get_id())
        
        io.set_structure(pdb_structure)
        pdb_file_path = pkt_path / f"{prot_name}_{ligd_name}.pdb"
        with open(pdb_file_path, 'w') as f:
            io.save(f, select=PocketSelect(pocket))
        
        pkt_structure = parser.get_structure(f"{prot_name}_{ligd_name}", pkt_path / f"{prot_name}_{ligd_name}.pdb")
        atoms = list(pkt_structure.get_atoms())
        coords = np.array([atom.coord for atom in atoms])
        
        if len(coords) > 0:
            ms.fit(coords)
            cluster_label = ms.labels_
            cluster_centers = np.array([coords[cluster_label == i].mean(axis=0) for i in range(cluster_label.max() + 1)])
            site_center_df = pd.concat([site_center_df, pd.DataFrame({
                'Protein Name': [prot_name], 
                'Ligand Name': [ligd_name], 
                'Binding Site Center': [cluster_centers.tolist()]
            })], ignore_index=True)
    
    site_center_df.to_csv(temp_dir / 'site_centers.csv', index=False)


# Example usage function
def example_usage():
    """Example of how to use the function-based interface."""
    
    # Example protein sequence (insulin)
    protein_sequence = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
    protein_id = "insulin"
    
    # Example SMILES (glucose)
    smiles = "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O"
    ligand_id = "glucose"
    
    # Run prediction
    results = predict_binding_sites_from_text(
        protein_sequence=protein_sequence,
        protein_id=protein_id,
        smiles=smiles,
        ligand_id=ligand_id,
        output_dir="./example_output/",
        batch_size=1,
        device_ids=[0],
        cluster=True
    )
    
    print("Prediction Results:")
    print(f"Protein: {results['protein_sequence'][:50]}...")
    print(f"Ligand SMILES: {results['ligand_smiles']}")
    print(f"Binding probabilities: {results['binding_probabilities'][:10]}...")  # First 10 residues
    print(f"Results saved to: {results['result_csv_path']}")
    if results['pocket_pdb_path']:
        print(f"Pocket PDB saved to: {results['pocket_pdb_path']}")


if __name__ == "__main__":
    example_usage()
