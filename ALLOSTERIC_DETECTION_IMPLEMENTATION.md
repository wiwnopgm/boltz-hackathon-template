# Allosteric Binding Detection Implementation

## Overview

This implementation adds automatic detection of allosteric vs orthosteric binding using **two complementary methods**:
1. **Variance-based detection**: Analyzes confidence metric variability across predictions
2. **Ligand similarity-based detection**: Compares query ligand to known allosteric binders

The system applies appropriate constraints and filtering for detected allosteric binders.

## Implementation Details

### 1. Dual Detection Strategy

#### Method 1: Variance-Based Detection

The system analyzes variance in confidence metrics from unconstrained predictions:

**Thresholds (based on `variance_comparison_ortho_allo.csv`):**
- **Confidence Score Std Dev**: > 0.005 (ortho: ~0.0023, allo: ~0.0115)
- **Ligand iPTM Std Dev**: > 0.01 (ortho: ~0.0025, allo: ~0.031)
- **Protein iPTM Std Dev**: > 0.01 (ortho: ~0.0025, allo: ~0.031)

If **any** of these standard deviations exceed their threshold → classified as allosteric

#### Method 2: Ligand Similarity-Based Detection

The system compares the query ligand to all known allosteric ligands in the reference dataset:

**Method:**
- Uses RDKit to compute Morgan fingerprints (ECFP4 equivalent, radius=2, 2048 bits)
- Calculates Tanimoto similarity between query and all reference allosteric ligands
- Finds maximum similarity score

**Threshold:**
- **Tanimoto Similarity**: > 0.80 (80% similarity)

If similarity to **any** allosteric ligand exceeds 80% → classified as allosteric

#### Combined Decision

The binding is classified as **ALLOSTERIC** if **EITHER** method indicates allosteric binding:
- `is_allosteric = is_allosteric_by_variance OR is_allosteric_by_similarity`

This provides more robust detection by combining structure-based and behavior-based signals.

### 2. Prediction Pipeline

The protein-ligand prediction workflow now follows these stages:

#### Stage 1: Unconstrained Prediction
- Run 5 unconstrained predictions
- Extract confidence metrics (confidence_score, ligand_iptm, iptm)
- Identify orthosteric pocket residues (within 4.5Å of ligand)

#### Stage 2: Allosteric Detection & Analysis
- Compute variance statistics from unconstrained predictions
- Check ligand similarity to known allosteric binders (using RDKit)
- Classify as orthosteric or allosteric using both methods
- Report which method(s) triggered allosteric classification
- Save analysis results to `allosteric_detection.json`

#### Stage 3: Constrained Prediction
For **allosteric binders only**:
- Add `negative_pocket` constraint to force binding away from orthosteric pocket
- Constraint parameters:
  - `binder`: "B" (ligand chain)
  - `contacts`: Orthosteric pocket residues
  - `min_distance`: 10.0 Å
  - `force`: True

For **orthosteric binders**:
- No negative pocket constraint added
- Model can freely predict binding in orthosteric site

### 3. Post-Processing & Filtering

For **allosteric binders only**:
- Filter structures based on pocket RMSD
- **Keep only structures with pocket RMSD > 4Å** from reference structure
  - This ensures predicted allosteric sites are far from orthosteric pocket
- If all structures filtered out, use unfiltered set as fallback

For **orthosteric binders**:
- No filtering applied
- All predicted structures used

### 4. Key Functions Added

**`load_reference_allosteric_ligands(reference_dataset_path)`**
- Loads SMILES strings of all allosteric ligands from reference dataset (asos_public.jsonl)
- Caches results for efficiency
- Auto-discovers dataset location if not specified

**`compute_tanimoto_similarity(smiles1, smiles2)`**
- Computes Tanimoto similarity between two molecules
- Uses Morgan fingerprints (ECFP4, radius=2, 2048 bits)
- Returns similarity score from 0.0 to 1.0
- Requires RDKit

**`check_ligand_similarity_to_allosteric(query_smiles, reference_dataset_path, threshold)`**
- Checks if query ligand is similar to any known allosteric ligands
- Returns (is_similar, max_similarity, most_similar_smiles)
- Compares against all allosteric ligands in reference dataset

**`extract_confidence_metrics(prediction_output_dir, datapoint_id)`**
- Extracts confidence_score, ligand_iptm, and iptm from prediction outputs
- Reads from `confidence_{datapoint_id}_model_{i}.json` files

**`is_allosteric_binder(metrics, ligand_smiles, reference_dataset_path)`**
- **Dual detection**: combines variance analysis and ligand similarity
- Computes standard deviations for confidence metrics
- Checks ligand similarity to known allosteric binders (if RDKit available)
- Returns (is_allosteric, detection_stats) with detailed information
- Detection stats include which method(s) triggered allosteric classification

**Modified Functions:**

**`_prefill_input_dict(..., is_allosteric=False)`**
- Adds `is_allosteric` parameter
- Only adds negative_pocket constraint when `is_allosteric=True`

**`prepare_protein_ligand(..., is_allosteric=False)`**
- Adds `is_allosteric` parameter for custom handling
- Receives input_dict with negative_pocket constraint already added if allosteric

**`post_process_protein_ligand(..., is_allosteric=False)`**
- Adds `is_allosteric` parameter
- Implements pocket RMSD filtering for allosteric binders
- Filters structures with pocket RMSD ≤ 4Å

### 5. Output Files

For each protein-ligand datapoint, the following files are generated:

1. **`orthosteric_pocket.txt`**: List of pocket residues
2. **`allosteric_detection.json`**: Comprehensive detection results from both methods
   ```json
   {
     "is_allosteric": true/false,
     "detection_stats": {
       "confidence_std": 0.012,
       "ligand_iptm_std": 0.025,
       "protein_iptm_std": 0.028,
       "allosteric_by_variance": true/false,
       "allosteric_by_similarity": true/false,
       "max_ligand_similarity": 0.85,
       "most_similar_allosteric_ligand": "SMILES_string...",
       "detection_method": ["variance", "similarity"]
     },
     "metrics": {
       "confidence_scores": [...],
       "ligand_iptm": [...],
       "protein_iptm": [...]
     },
     "ligand_smiles": "query_SMILES",
     "thresholds": {
       "confidence_std": 0.005,
       "ligand_iptm_std": 0.01,
       "protein_iptm_std": 0.01,
       "ligand_similarity": 0.80
     }
   }
   ```
3. **`unconstrained_for_pocket/model_*.pdb`**: Unconstrained predictions used for pocket identification

### 6. Potential Guidance Weights

Also updated the RepulsionContactPotential guidance weights in `potentials.py` to be much stronger:
- Previous: `[5.0, 10.0, 15.0]`
- New: `[50.0, 75.0, 100.0]`

This provides stronger guidance during diffusion to respect the negative pocket constraints.

## Usage

No changes needed to user-facing API. The system automatically:
1. Detects allosteric vs orthosteric binding
2. Applies appropriate constraints
3. Filters results accordingly

Simply run predictions as normal:
```bash
python hackathon/predict_hackathon.py --input-jsonl dataset.jsonl --msa-dir ./msa --submission-dir submission --intermediate-dir intermediate
```

## Benefits

1. **Dual Detection Methods**: Combines variance-based and structure-based detection for higher accuracy
2. **Automatic Classification**: No manual labeling or classification needed
3. **Robust Detection**: Catches allosteric binders even if one method fails
4. **Appropriate Constraints**: Allosteric binders automatically get negative pocket constraints
5. **Quality Filtering**: Allosteric predictions filtered to ensure binding far from orthosteric site
6. **Graceful Degradation**: Falls back to variance-only if RDKit unavailable
7. **Robust Fallback**: If all structures filtered, uses unfiltered set
8. **Full Transparency**: All analysis saved with detailed detection information

## Dependencies

- **Required**: NumPy, BioPython, PyYAML (already in base environment)
- **Optional (but recommended)**: RDKit
  - Used for ligand similarity-based detection
  - If not available, system falls back to variance-only detection
  - Install: `conda install -c conda-forge rdkit` or `pip install rdkit`

## Notes

- **Variance thresholds** based on empirical analysis of variance_comparison_ortho_allo.csv
- **Similarity threshold** set to 80% Tanimoto similarity (can be adjusted via LIGAND_SIMILARITY_THRESHOLD)
- **Detection uses OR logic**: Classification is allosteric if **either** method indicates it
- **Ligand similarity** computed using Morgan fingerprints (ECFP4 equivalent)
- Reference allosteric ligands automatically extracted from asos_public.jsonl
- RMSD filtering uses first predicted structure as reference
- Pocket RMSD calculated using CA atoms of pocket residues
- System gracefully handles missing RDKit by falling back to variance-only detection

