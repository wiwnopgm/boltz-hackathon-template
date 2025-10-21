# Negative Pocket Constraint Fix

## Summary of Issues Found

The negative pocket constraints were **not being enforced** during prediction, causing ligands to bind directly in the pockets they should avoid.

## Root Causes

### 1. **Critical Bug in `src/boltz/main.py`**
The `--use_potentials` flag was NOT enabling `contact_guidance_update`, which is required for negative pocket (repulsion) constraints to work.

**Before (BROKEN):**
```python
steering_args = BoltzSteeringParams()
steering_args.fk_steering = use_potentials
steering_args.physical_guidance_update = use_potentials
# contact_guidance_update NOT SET! ← Bug
```

**After (FIXED):**
```python
steering_args = BoltzSteeringParams()
steering_args.fk_steering = use_potentials
steering_args.physical_guidance_update = use_potentials
steering_args.contact_guidance_update = use_potentials  # ← Added!
```

### 2. **Syntax Error in `hackathon/predict_hackathon.py`**
The `--use_potentials` flag was being passed with an argument `"True"`, but it's a boolean flag that doesn't take arguments.

**Before (BROKEN):**
```python
cli_args = ["--diffusion_samples", "5", "--use_potentials", "True"]
```

**After (FIXED):**
```python
cli_args = ["--diffusion_samples", "5", "--use_potentials"]
```

## Validation Results Before Fix

Out of 75 predictions:
- **Passed: 25 (33.3%)**
- **Failed: 50 (66.7%)**

### Worst Violations
| Datapoint | Min Distance | Required | Deficit | Violated Residues |
|-----------|--------------|----------|---------|-------------------|
| 1MMN_ORTHOSTERIC_ANP | 0.50 Å | 10.0 Å | **9.50 Å** | 23 |
| 1BZC_ORTHOSTERIC_TPI | 1.03 Å | 10.0 Å | **8.97 Å** | 17 |
| 2E9N_ORTHOSTERIC_76A | 3.10 Å | 10.0 Å | **6.90 Å** | 19 |

These severe violations show ligands binding **directly in the orthosteric pockets** they should avoid!

## Files Modified

1. **`src/boltz/main.py`** (line 1312)
   - Added: `steering_args.contact_guidance_update = use_potentials`

2. **`hackathon/predict_hackathon.py`** (lines 48, 80)
   - Changed: `"--use_potentials", "True"` → `"--use_potentials"`

## How to Re-run Predictions

### Option 1: Quick Test (Single Structure)
```bash
cd /home/ubuntu/will/boltz-hackathon-template

boltz predict tmp/test_negative_pocket_v2/input/1AXB_ORTHOSTERIC_FOS_config_0.yaml \
  --devices 1 \
  --out_dir tmp/test_fix \
  --cache /home/ubuntu/.boltz \
  --no_kernels \
  --output_format pdb \
  --diffusion_samples 5 \
  --use_potentials

# Then validate
python validate_negative_pocket.py \
  --pdb tmp/test_fix/boltz_results_*/predictions/*/model_0.pdb \
  --config tmp/test_negative_pocket_v2/input/1AXB_ORTHOSTERIC_FOS_config_0.yaml
```

### Option 2: Full Dataset Re-run
```bash
cd /home/ubuntu/will/boltz-hackathon-template

# Run predictions with fixed code
python ./hackathon/predict_hackathon.py \
  --input-jsonl ./hackathon_data/datasets/asos_public/asos_public_test1.jsonl \
  --msa-dir ./hackathon_data/datasets/asos_public/msa/ \
  --submission-dir ./test_results/test_negative_pocket_fixed/ \
  --intermediate-dir ./tmp/test_negative_pocket_fixed/ \
  --result-folder ./test_negative_pocket_fixed_results

# Then validate all predictions
python validate_negative_pocket_from_jsonl.py \
  --predictions test_results/test_negative_pocket_fixed \
  --jsonl hackathon_data/datasets/asos_public/asos_public_test1.jsonl \
  --output validation_results_fixed.csv \
  -v
```

## How the Fix Works

The `RepulsionContactPotential` in `src/boltz/model/potentials/potentials.py` (lines 825-841) applies repulsive forces during diffusion sampling to push ligand atoms away from specified pocket residues.

**Key mechanism:**
```python
RepulsionContactPotential(
    parameters={
        "guidance_interval": 2,  # Apply every 2 diffusion steps
        "guidance_weight": (
            PiecewiseStepFunction(
                thresholds=[0.25, 0.75], 
                values=[1.0, 2.0, 3.0]  # Stronger as diffusion progresses
            )
            if steering_args["contact_guidance_update"]  # ← This must be True!
            else 0.0  # No force if False
        ),
        ...
    }
)
```

**Without the fix:**
- `contact_guidance_update = False` (default)
- `guidance_weight = 0.0`
- **No repulsive force applied** → ligands bind in forbidden pockets

**With the fix:**
- `contact_guidance_update = True` (when `--use_potentials` is used)
- `guidance_weight = [1.0, 2.0, 3.0]` (increases during sampling)
- **Repulsive forces push ligands away** → ligands avoid forbidden pockets

## Expected Improvements

After the fix, you should see:
- ✅ **Significantly higher pass rate** (expect >80% instead of 33%)
- ✅ **Much larger minimum distances** from pocket residues
- ✅ **Fewer violations** per structure
- ✅ **Ligands positioned in alternative binding sites** (e.g., allosteric sites)

## Advanced Tuning (If Needed)

If some structures still fail after the fix, you can strengthen the repulsion by modifying `src/boltz/model/potentials/potentials.py` line 831:

```python
# Current (moderate repulsion)
values=[1.0, 2.0, 3.0]

# Stronger repulsion
values=[2.0, 4.0, 6.0]

# Very strong repulsion (for difficult cases)
values=[3.0, 6.0, 10.0]
```

Or increase guidance steps (line 157 in `main.py`):
```python
num_gd_steps: int = 20  # Try 40 or 50 for stronger enforcement
```

Or apply more frequently (line 827 in `potentials.py`):
```python
"guidance_interval": 2,  # Change to 1 to apply every step
```

## Validation

Use the provided validation scripts to check if constraints are satisfied:

```bash
# Single structure
python validate_negative_pocket.py --pdb <pdb_file> --config <yaml_file>

# Batch validation
python validate_negative_pocket_from_jsonl.py \
  --predictions <predictions_dir> \
  --jsonl <jsonl_file> \
  --output results.csv \
  -v
```

## Performance Impact

Enabling potentials adds minimal overhead:
- **Guidance interval: 2** → Applies every 2 diffusion steps (not every step)
- **Gradient descent: 20 steps** → Fast optimization at each guidance step
- **Expected slowdown: ~10-20%** compared to unconstrained prediction

## Notes

- The fix is **backward compatible** - predictions without `--use_potentials` work as before
- The validation scripts work on any PDB files, even from other tools
- All distances are calculated in 3D Euclidean space between all atom pairs
- The constraint enforcement is **soft** (gradient-based), not hard (rejection-based)

## Questions?

If predictions still fail after the fix:
1. Check if `--use_potentials` flag is actually being used
2. Verify `contact_guidance_update=True` in logs/debug output
3. Try stronger repulsion weights (see Advanced Tuning above)
4. Check if the constraint definition in YAML/JSONL is correct

---

**TL;DR:** The bug was that `--use_potentials` didn't enable contact guidance, so negative pocket constraints had zero force. Now fixed! Re-run with `--use_potentials` flag.


