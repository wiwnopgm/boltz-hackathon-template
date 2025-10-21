# Boltz Hackathon Prediction Flow

This document explains the prediction pipeline implemented in `predict_hackathon.py`, which handles both protein-complex and protein-ligand structure predictions with special support for allosteric binding detection.

## Overview

The pipeline uses a **two-stage prediction approach** for protein-ligand tasks to automatically detect and handle allosteric vs orthosteric binding. For protein-complex tasks, it uses a single-stage approach.

---

## Protein-Ligand Prediction Flow

### Stage 1: Unconstrained Predictions & Pocket Identification

**Purpose:** Establish a baseline and identify the orthosteric (primary) binding pocket.

1. **Run Unconstrained Boltz Predictions**
   - Generate 5 structural models without any constraints
   - Use standard Boltz parameters with `--use_potentials`
   - Save models as `model_0.pdb` through `model_4.pdb`

2. **Identify Orthosteric Pocket**
   ```
   Function: identify_orthosteric_pocket()
   ```
   - Load all 5 predicted structures
   - Superpose structures by aligning protein Cα atoms
   - Calculate pairwise ligand RMSD between all models
   - Find the **medoid** (most representative) structure
   - Identify contact residues within 4.5 Å of ligand in medoid structure
   - Save pocket residues as set of `(chain_id, resname, residue_number)`

### Stage 2: Allosteric Detection

**Purpose:** Determine if the ligand is an allosteric binder using two independent methods.

#### Method 1: Variance-Based Detection

```
Function: extract_confidence_metrics() → is_allosteric_binder()
```

**Rationale:** Allosteric binders show higher variance across models due to multiple possible binding sites.

Extract from confidence JSON files:
- **Confidence Score** (overall structure confidence)
- **Ligand iPTM** (interface predicted template modeling score)
- **Protein iPTM** (protein-only iPTM)

Calculate standard deviation for each metric across 5 models:

| Metric | Orthosteric Threshold | Allosteric Threshold | Decision Threshold |
|--------|----------------------|----------------------|-------------------|
| Confidence Score Std | ~0.0023 | ~0.0115 | **> 0.005** |
| Ligand iPTM Std | ~0.0025 | ~0.031 | **> 0.01** |
| Protein iPTM Std | ~0.0025 | ~0.031 | **> 0.01** |

**Classification:** If ANY metric exceeds threshold → **ALLOSTERIC**

#### Method 2: Ligand Similarity Detection

```
Function: check_ligand_similarity_to_allosteric()
```

**Rationale:** Ligands chemically similar to known allosteric binders are likely allosteric themselves.

Process:
1. Load reference allosteric ligand SMILES from training dataset
2. Compute Tanimoto similarity using Morgan fingerprints (ECFP4)
3. Compare query ligand to all reference allosteric ligands
4. Find maximum similarity score

**Classification:** If similarity **> 0.80** (80%) → **ALLOSTERIC**

#### Combined Decision

```python
is_allosteric = is_allosteric_by_variance OR is_allosteric_by_similarity
```

Results saved to: `{datapoint_id}/allosteric_detection.json`

### Stage 3: Automatic Pocket Scanning (Optional)

**Purpose:** Use ML-based binding site prediction to guide ligand placement.

**When enabled** (via `--use-auto-pocket-scanner` flag, enabled by default):

1. **Predict Binding Sites**
   ```
   Function: predict_binding_sites()
   ```
   - Analyzes protein sequence and ligand SMILES
   - Generates binding probability for each residue
   - Returns ranked list of potential binding sites

2. **Create Pocket Constraints**
   ```
   Function: get_top_binding_sites() → create_pocket_constraints()
   ```
   - Extract top N residues (default: 5) above threshold (default: 0.8)
   - Create pocket constraint to guide ligand placement
   
   ```yaml
   constraints:
     - pocket:
         binder: "B"  # Ligand chain ID
         contacts: [["A", 45], ["A", 67], ...]  # Predicted binding sites
         max_distance: 2.0  # Maximum distance in Angstroms
         force: true
   ```

**When disabled**: No automatic pocket constraints added

### Stage 4: Constrained Predictions

**Purpose:** Generate final predictions with appropriate constraints based on binding type.

#### For Orthosteric Binders (Default)

- Automatic pocket constraints added if pocket scanner enabled
- Ligand guided to predicted binding sites
- 10 diffusion samples with 300 sampling steps

#### For Allosteric Binders

Add **negative_pocket constraint** to force binding away from orthosteric site:

```yaml
constraints:
  - negative_pocket:
      binder: "B"  # Ligand chain ID
      contacts: [["A", 69], ["A", 70], ...]  # Orthosteric pocket residues
      min_distance: 10.0  # Minimum distance in Angstroms
      force: true
```

**Effect:** Ligand must stay ≥10 Å away from orthosteric pocket residues

**Additional pocket constraints may be added** if pocket scanner is enabled

### Stage 5: Post-Processing & Ranking

```
Function: post_process_protein_ligand()
```

#### For Allosteric Binders - Additional Filtering

**Filter 1: Ligand RMSD Filter**
- Compare predicted ligand positions to unconstrained predictions
- Align all structures to common reference frame
- Calculate minimum ligand RMSD to any unconstrained model
- **Keep only structures with ligand RMSD > 2 Å**
- Rationale: Ensures ligand is in different binding site

**Filter 2: Pocket RMSD Filter**
- Calculate Cα RMSD of orthosteric pocket residues between predicted structures
- **Keep only structures with pocket RMSD > 4 Å**
- Rationale: Ensures ligand doesn't distort orthosteric pocket (allosteric effect)

#### For All Predictions - Ranking

1. Extract ligand iPTM scores from confidence JSON files
2. Sort structures by ligand iPTM (descending - higher is better)
3. Return top 5 ranked structures as `model_0.pdb` through `model_4.pdb`

---

## Protein-Complex Prediction Flow

### Multi-Stage Prediction with Pocket Finding

**Purpose:** Predict antibody-antigen complexes with enhanced pocket detection.

1. **Initial 3D Structure Prediction**
   ```
   Function: predict_3d_structure()
   ```
   - Generate initial structural model for pocket analysis
   - Use standard Boltz parameters with 5 diffusion samples
   - Select best model based on confidence score

2. **Pocket Detection**
   ```
   Function: find_pockets()
   ```
   - Run fpocket on predicted structure
   - Identify potential binding pockets
   - Extract pocket residues (limit: 20 residues)
   - Returns list of (chain, resnum) tuples

3. **Prepare Input**
   - 3 chains: H (heavy), L (light), A (antigen)
   - May include MSA files for each chain
   - May include contact or other constraints based on pocket findings

4. **Run Final Boltz Predictions**
   - Generate 5 structural models (default)
   - Use `--diffusion_samples 5 --use_potentials`

5. **Post-Processing**
   ```
   Function: post_process_protein_complex()
   ```
   - Collect all predicted PDB files
   - Sort by model index
   - Return sorted paths

---

## Key Parameters & Thresholds

### Pocket Identification
```python
MODEL_COUNT = 5              # Number of models to generate
LIG_NAME = "LIG"            # Ligand residue name in PDB
RMSD_CUTOFF = 2.0           # Å - for ligand clustering
CONTACT_CUTOFF = 4.5        # Å - for pocket residue identification
```

### Allosteric Detection
```python
CONFIDENCE_STD_THRESHOLD = 0.005      # Confidence score variance
IPTM_STD_THRESHOLD = 0.01            # Protein iPTM variance
LIGAND_IPTM_STD_THRESHOLD = 0.01     # Ligand iPTM variance
LIGAND_SIMILARITY_THRESHOLD = 0.80    # Tanimoto similarity (80%)
```

### Allosteric Filtering
```python
MIN_LIGAND_RMSD = 2.0       # Å - minimum ligand displacement from orthosteric
MIN_POCKET_RMSD = 4.0       # Å - minimum pocket distortion
MIN_DISTANCE = 10.0         # Å - negative_pocket constraint distance
```

### Automatic Pocket Scanning
```python
TOP_N_PREDICTIONS = 5       # Number of top binding site predictions to use
BINDING_THRESHOLD = 0.8     # Minimum binding probability threshold
MAX_DISTANCE = 2.0          # Å - maximum distance for pocket constraint
POCKETS_LIMIT = 20          # Maximum number of pocket residues from fpocket
```

### Prediction Parameters
```python
# Protein-Ligand
DIFFUSION_SAMPLES = 10      # Number of models to generate
SAMPLING_STEPS = 300        # Number of sampling steps

# Protein-Complex
DIFFUSION_SAMPLES = 5       # Number of models to generate
```

---

## Output Files

### Per Datapoint Directory: `submission/{datapoint_id}/`

```
model_0.pdb                    # Top-ranked structure
model_1.pdb                    # Second-ranked structure
model_2.pdb                    # Third-ranked structure
model_3.pdb                    # Fourth-ranked structure
model_4.pdb                    # Fifth-ranked structure
orthosteric_pocket.txt         # Identified pocket residues (protein-ligand only)
allosteric_detection.json      # Detection analysis (protein-ligand only)
unconstrained_for_pocket/      # Unconstrained predictions (protein-ligand only)
  └── model_*.pdb
```

### Binding Site Prediction Output: `binding_output/{datapoint_id}/`

```
binding_predictions.csv        # Residue-level binding probabilities
protein_structure.pdb          # ESMFold predicted structure (if used)
ligand_conformer.pdb          # Generated ligand conformer
```

### Intermediate Files: `hackathon_intermediate/`

```
input/
  └── {datapoint_id}_config_*.yaml    # Input YAML files for Boltz
predictions/
  └── boltz_results_{datapoint_id}_*/  # Raw Boltz outputs
      └── predictions/
          └── {datapoint_id}_*/
              ├── {datapoint_id}_*_model_*.pdb
              └── confidence_{datapoint_id}_*_model_*.json
```

---

## Customization Points

Participants can modify four key functions:

### 1. `prepare_protein_complex()`
- Calls `predict_3d_structure()` to generate initial structure
- Calls `find_pockets()` to identify binding pockets using fpocket
- Modify input dictionary before prediction
- Add constraints (contact, distance, etc.) based on pocket findings
- Specify CLI arguments for Boltz
- Return multiple configurations to run

### 2. `prepare_protein_ligand()`
- Access to automatic pocket scanning via `args.use_auto_pocket_scanner`
- Calls `predict_binding_sites()` for ML-based pocket prediction
- Uses `get_top_binding_sites()` and `create_pocket_constraints()`
- Access to `is_allosteric` flag
- Can modify/augment negative_pocket constraints
- Can adjust TOP_N_PREDICTIONS and BINDING_THRESHOLD
- Return multiple configurations to run

### 3. `post_process_protein_complex()`
- Custom ranking logic for antibody-antigen predictions
- Can combine results from multiple configurations

### 4. `post_process_protein_ligand()`
- Custom ranking logic for protein-ligand predictions
- Access to pocket residues and allosteric information
- Can implement custom filtering/scoring
- Default ranking by ligand iPTM scores

---

## Example Usage

### Single Datapoint
```bash
python predict_hackathon.py \
  --input-json examples/specs/example_protein_ligand.json \
  --msa-dir ./msa \
  --submission-dir submission \
  --intermediate-dir intermediate \
  --result-folder evaluation_results
```

### Multiple Datapoints
```bash
python predict_hackathon.py \
  --input-jsonl hackathon_data/datasets/asos_public/asos_public.jsonl \
  --msa-dir ./msa \
  --submission-dir submission \
  --intermediate-dir intermediate \
  --result-folder evaluation_results
```

### With Automatic Pocket Scanning
```bash
# Enabled by default, to disable:
python predict_hackathon.py \
  --input-json example.json \
  --msa-dir ./msa \
  --submission-dir submission \
  --intermediate-dir intermediate \
  --use-auto-pocket-scanner false
```

### With GPU
According to project preferences, all commands should use GPU by default [[memory:5749486]]. The system automatically uses `--devices 1` for GPU acceleration.

---

## Dependencies

- **Boltz**: Structure prediction engine
- **BioPython**: PDB parsing and structure manipulation
- **RDKit** (optional): Ligand similarity calculations
- **NumPy**: Numerical operations
- **PyYAML**: Configuration file handling
- **fpocket**: Binding pocket detection tool (for protein-complex)
- **predict_binding_sites**: ML-based binding site predictor (from hackathon.contrib)

---

## Algorithm Details

### Medoid Selection

The medoid is the structure that minimizes the sum of distances to all other structures:

```python
def _medoid_index(dist, cutoff):
    # Count structures within RMSD cutoff for each model
    counts = [sum(d <= cutoff for d in row) for row in dist]
    
    # Find model(s) with maximum neighbors
    candidates = [i for i, c in enumerate(counts) if c == max(counts)]
    
    # Among candidates, select one with minimum average distance
    return min(candidates, key=lambda i: sum(dist[i]) / (n-1))
```

### Tanimoto Similarity

```python
similarity = (c / (a + b - c))
```
Where:
- `a` = number of bits set in fingerprint A
- `b` = number of bits set in fingerprint B  
- `c` = number of bits set in both fingerprints

Uses Morgan fingerprints (radius=2, 2048 bits) = ECFP4 equivalent

### Binding Site Prediction

The automatic pocket scanner uses ML-based methods to predict binding sites:

1. **Input Processing**
   - Protein sequence and ligand SMILES are provided
   - May use ESMFold for structure prediction if needed
   - Generates 3D ligand conformer

2. **Residue Scoring**
   - Each residue receives a binding probability score
   - Scores range from 0.0 (unlikely) to 1.0 (highly likely)

3. **Top-N Selection**
   ```python
   def get_top_binding_sites(binding_probs, top_n, threshold):
       # Filter by threshold
       valid_indices = where(probs >= threshold)
       
       # Get top N from valid indices
       top_indices = argsort(valid_probs)[-top_n:]
       
       # Convert to 1-based residue numbers
       return (top_indices + 1).tolist()
   ```

4. **Constraint Generation**
   - Creates pocket constraint with selected residues
   - Uses max_distance = 2.0 Å
   - Forces ligand to bind near predicted sites

---

## Troubleshooting

### "No confidence files found"
- Check that Boltz predictions completed successfully
- Verify confidence JSON files exist in prediction directories
- Confirm file naming matches expected pattern

### "All structures filtered out"
- For allosteric: RMSD thresholds may be too strict
- Pipeline falls back to unfiltered structures
- Consider adjusting `MIN_LIGAND_RMSD` or `MIN_POCKET_RMSD`

### "RDKit not available"
- Ligand similarity detection will be disabled
- Only variance-based allosteric detection will be used
- Install RDKit if ligand similarity is needed: `pip install rdkit`

### "Could not load reference dataset"
- Ligand similarity detection requires reference dataset
- Default path: `hackathon_data/datasets/asos_public/asos_public.jsonl`
- Provide path via code or ensure file exists

### "Binding site prediction failed"
- Check that `predict_binding_sites` is available from `hackathon.contrib`
- Verify protein sequence and ligand SMILES are valid
- Check `binding_output/{datapoint_id}/` directory for error logs
- Disable with `--use-auto-pocket-scanner false` if needed

### "fpocket command not found"
- fpocket must be installed for protein-complex predictions
- Install via package manager or from source
- Ensure fpocket is in system PATH

---

## Performance Notes

- **Unconstrained predictions** add 5 model runtime overhead for protein-ligand tasks
- **Allosteric detection** is fast (<1 second) once predictions complete
- **Filtering** requires loading and aligning structures (seconds per datapoint)
- **Binding site prediction** adds minimal overhead (~5-10 seconds per datapoint)
- **fpocket analysis** is fast (<5 seconds) for protein-complex predictions
- **Initial structure prediction** (for pocket finding) adds 1 complete prediction run for protein-complex
- **Constrained predictions** with pocket constraints may take longer due to increased sampling steps (300 vs default)
- **GPU** is strongly recommended for all predictions [[memory:5749486]]

---

## References

- Variance thresholds derived from `variance_comparison_ortho_allo.csv` analysis
- Morgan fingerprints: Rogers & Hahn, J. Chem. Inf. Model. 2010
- Tanimoto similarity: Jaccard index for binary fingerprints
- fpocket: Le Guilloux et al., BMC Bioinformatics 2009
- Binding site prediction: ML-based approach from hackathon.contrib

---

## Summary of Key Features

### Protein-Ligand Pipeline
1. **Unconstrained Predictions** (5 models) → Pocket Identification
2. **Allosteric Detection** via variance + ligand similarity
3. **Automatic Pocket Scanning** (ML-based, enabled by default)
4. **Constrained Predictions** (10 models, 300 steps)
   - Orthosteric: pocket constraints guide ligand placement
   - Allosteric: negative_pocket + optional pocket constraints
5. **Post-Processing & Ranking** by ligand iPTM

### Protein-Complex Pipeline
1. **Initial Structure Prediction** (5 models) → Select best
2. **fpocket Analysis** → Identify binding pockets
3. **Constrained Predictions** (5 models) with pocket-informed constraints
4. **Post-Processing** → Sort and return models

### Key Innovations
- **Automatic allosteric detection** using two independent methods
- **ML-based binding site prediction** for improved ligand placement
- **Structure-based pocket detection** using fpocket for protein complexes
- **Adaptive constraint generation** based on binding type
- **RMSD-based filtering** for allosteric predictions

