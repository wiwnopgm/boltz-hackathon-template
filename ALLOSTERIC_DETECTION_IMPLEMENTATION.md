# Allosteric Binding Detection Implementation

## Overview

This implementation adds automatic detection of allosteric vs orthosteric binding based on confidence variance, and applies appropriate constraints and filtering for allosteric binders.

## Implementation Details

### 1. Variance-Based Allosteric Detection

The system analyzes variance in confidence metrics from unconstrained predictions to determine binding type:

**Thresholds (based on `variance_comparison_ortho_allo.csv`):**
- **Confidence Score Std Dev**: > 0.005 (ortho: ~0.0023, allo: ~0.0115)
- **Ligand iPTM Std Dev**: > 0.01 (ortho: ~0.0025, allo: ~0.031)
- **Protein iPTM Std Dev**: > 0.01 (ortho: ~0.0025, allo: ~0.031)

**Detection Logic:**
If **any** of these standard deviations exceed their threshold, the binding is classified as allosteric.

### 2. Prediction Pipeline

The protein-ligand prediction workflow now follows these stages:

#### Stage 1: Unconstrained Prediction
- Run 5 unconstrained predictions
- Extract confidence metrics (confidence_score, ligand_iptm, iptm)
- Identify orthosteric pocket residues (within 4.5Å of ligand)

#### Stage 2: Allosteric Detection & Analysis
- Compute variance statistics from unconstrained predictions
- Classify as orthosteric or allosteric based on thresholds
- Save analysis results to `variance_analysis.json`

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

**`extract_confidence_metrics(prediction_output_dir, datapoint_id)`**
- Extracts confidence_score, ligand_iptm, and iptm from prediction outputs
- Reads from `confidence_{datapoint_id}_model_{i}.json` files

**`is_allosteric_binder(metrics)`**
- Computes standard deviations for confidence metrics
- Returns boolean flag and variance statistics
- Determines if binding is allosteric based on thresholds

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
2. **`variance_analysis.json`**: Detailed variance statistics and classification
   ```json
   {
     "is_allosteric": true/false,
     "variance_stats": {
       "confidence_std": 0.012,
       "ligand_iptm_std": 0.025,
       "protein_iptm_std": 0.028
     },
     "metrics": {
       "confidence_scores": [...],
       "ligand_iptm": [...],
       "protein_iptm": [...]
     },
     "thresholds": {...}
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

1. **Automatic Detection**: No manual classification needed
2. **Appropriate Constraints**: Allosteric binders get negative pocket constraints
3. **Quality Filtering**: Allosteric predictions filtered to ensure binding far from orthosteric site
4. **Robust Fallback**: If all structures filtered, uses unfiltered set
5. **Full Transparency**: All analysis saved for review

## Notes

- Thresholds based on empirical analysis of variance_comparison_ortho_allo.csv
- Detection uses OR logic (any metric exceeding threshold triggers allosteric classification)
- RMSD filtering uses first predicted structure as reference
- Pocket RMSD calculated using CA atoms of pocket residues

