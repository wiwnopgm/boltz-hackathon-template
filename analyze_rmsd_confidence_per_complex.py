#!/usr/bin/env python3
"""
Analyze RMSD vs confidence metrics per complex (aggregated across models)
Each point represents one complex with average/min/max RMSD
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

# Set up paths
BASE_DIR = Path("/home/ubuntu/will/boltz-hackathon-template/hackathon_data")
PREDICTIONS_DIR = BASE_DIR / "intermediate_files/asos_public/predictions"
EVAL_DIR = BASE_DIR / "evaluation/asos_public"
RMSD_CSV = EVAL_DIR / "combined_results.csv"

# Load RMSD data
print("Loading RMSD data...")
rmsd_df = pd.read_csv(RMSD_CSV)
print(f"Loaded {len(rmsd_df)} datapoints")

# Collect confidence metrics for all models
print("\nCollecting confidence metrics from predictions...")
confidence_data = []

for idx, row in rmsd_df.iterrows():
    datapoint_id = row['datapoint_id']
    ligand_type = row['type']
    
    # Find the prediction directory
    pred_dir = PREDICTIONS_DIR / f"boltz_results_{datapoint_id}_config_0" / "predictions" / f"{datapoint_id}_config_0"
    
    if not pred_dir.exists():
        print(f"Warning: Prediction directory not found for {datapoint_id}")
        continue
    
    # Collect metrics for all 5 models
    model_rmsds = []
    model_ligand_iptms = []
    model_iptms = []
    model_ptms = []
    model_conf_scores = []
    model_plddt = []
    
    for model_idx in range(5):
        confidence_file = pred_dir / f"confidence_{datapoint_id}_config_0_model_{model_idx}.json"
        
        if not confidence_file.exists():
            print(f"Warning: Confidence file not found: {confidence_file}")
            continue
        
        with open(confidence_file, 'r') as f:
            conf_data = json.load(f)
        
        # Get RMSD for this specific model
        rmsd_col = f"rmsd_model_{model_idx}"
        rmsd = row[rmsd_col]
        
        model_rmsds.append(rmsd)
        model_ligand_iptms.append(conf_data.get('ligand_iptm', np.nan))
        model_iptms.append(conf_data.get('iptm', np.nan))
        model_ptms.append(conf_data.get('ptm', np.nan))
        model_conf_scores.append(conf_data.get('confidence_score', np.nan))
        model_plddt.append(conf_data.get('complex_plddt', np.nan))
    
    # Calculate aggregated statistics per complex
    if model_rmsds:
        confidence_data.append({
            'datapoint_id': datapoint_id,
            'type': ligand_type,
            'rmsd_mean': np.mean(model_rmsds),
            'rmsd_min': np.min(model_rmsds),
            'rmsd_max': np.max(model_rmsds),
            'rmsd_std': np.std(model_rmsds),
            'ligand_iptm_mean': np.nanmean(model_ligand_iptms),
            'ligand_iptm_max': np.nanmax(model_ligand_iptms),
            'ligand_iptm_min': np.nanmin(model_ligand_iptms),
            'iptm_mean': np.nanmean(model_iptms),
            'iptm_max': np.nanmax(model_iptms),
            'iptm_min': np.nanmin(model_iptms),
            'ptm_mean': np.nanmean(model_ptms),
            'confidence_score_mean': np.nanmean(model_conf_scores),
            'confidence_score_max': np.nanmax(model_conf_scores),
            'complex_plddt_mean': np.nanmean(model_plddt),
            'n_models': len(model_rmsds)
        })

# Create DataFrame
df = pd.DataFrame(confidence_data)
print(f"\nAggregated metrics for {len(df)} complexes")
print(f"Orthosteric: {len(df[df['type'] == 'orthosteric'])}")
print(f"Allosteric: {len(df[df['type'] == 'allosteric'])}")

# Save the aggregated data
output_file = EVAL_DIR / "rmsd_confidence_per_complex.csv"
df.to_csv(output_file, index=False)
print(f"\nSaved aggregated data to {output_file}")

# Calculate correlations
print("\n" + "="*80)
print("CORRELATION ANALYSIS (PER COMPLEX)")
print("="*80)

def calculate_correlations(data, name):
    """Calculate and display correlations"""
    print(f"\n{name} ({len(data)} complexes)")
    print("-" * 80)
    
    # Test different RMSD aggregations vs confidence metrics
    results = []
    
    # RMSD aggregations
    rmsd_types = [
        ('rmsd_mean', 'Mean RMSD'),
        ('rmsd_min', 'Min RMSD'),
        ('rmsd_max', 'Max RMSD')
    ]
    
    # Confidence metrics
    conf_metrics = [
        ('ligand_iptm_mean', 'Mean Ligand iPTM'),
        ('ligand_iptm_max', 'Max Ligand iPTM'),
        ('iptm_mean', 'Mean iPTM'),
        ('iptm_max', 'Max iPTM'),
        ('confidence_score_mean', 'Mean Confidence'),
        ('confidence_score_max', 'Max Confidence'),
    ]
    
    for rmsd_col, rmsd_label in rmsd_types:
        print(f"\n{rmsd_label}:")
        for conf_col, conf_label in conf_metrics:
            valid_data = data[[rmsd_col, conf_col]].dropna()
            if len(valid_data) > 1:
                pearson_r, pearson_p = pearsonr(valid_data[rmsd_col], valid_data[conf_col])
                spearman_r, spearman_p = spearmanr(valid_data[rmsd_col], valid_data[conf_col])
                
                print(f"  vs {conf_label}:")
                print(f"    Pearson r:  {pearson_r:7.4f}  (p={pearson_p:.4e})")
                print(f"    Spearman r: {spearman_r:7.4f}  (p={spearman_p:.4e})")
                
                results.append({
                    'rmsd_type': rmsd_label,
                    'confidence_metric': conf_label,
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'n': len(valid_data)
                })
    
    return pd.DataFrame(results)

# Overall correlations
overall_corr = calculate_correlations(df, "OVERALL (All complexes)")

# Orthosteric only
ortho_df = df[df['type'] == 'orthosteric']
ortho_corr = calculate_correlations(ortho_df, "ORTHOSTERIC")

# Allosteric only
allo_df = df[df['type'] == 'allosteric']
allo_corr = calculate_correlations(allo_df, "ALLOSTERIC")

# Save correlation results
corr_results = pd.concat([
    overall_corr.assign(dataset='overall'),
    ortho_corr.assign(dataset='orthosteric'),
    allo_corr.assign(dataset='allosteric')
])
corr_file = EVAL_DIR / "correlation_results_per_complex.csv"
corr_results.to_csv(corr_file, index=False)
print(f"\nSaved correlation results to {corr_file}")

# Create visualizations
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# Set up plotting style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

def create_scatter_plot(data, x_col, y_col, title, filename, color_by_type=True):
    """Create scatter plot with regression line"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if color_by_type:
        # Plot orthosteric and allosteric separately
        ortho = data[data['type'] == 'orthosteric']
        allo = data[data['type'] == 'allosteric']
        
        ax.scatter(ortho[x_col], ortho[y_col], alpha=0.7, s=100, 
                   label=f'Orthosteric (n={len(ortho)})', color='#2E86AB', 
                   edgecolors='black', linewidths=1)
        ax.scatter(allo[x_col], allo[y_col], alpha=0.7, s=100, 
                   label=f'Allosteric (n={len(allo)})', color='#A23B72', 
                   edgecolors='black', linewidths=1)
    else:
        ax.scatter(data[x_col], data[y_col], alpha=0.7, s=100, 
                  edgecolors='black', linewidths=1)
    
    # Calculate and add trend line
    valid_data = data[[x_col, y_col]].dropna()
    if len(valid_data) > 1:
        z = np.polyfit(valid_data[x_col], valid_data[y_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid_data[x_col].min(), valid_data[x_col].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2.5, label='Linear fit')
        
        # Calculate correlation
        pearson_r, pearson_p = pearsonr(valid_data[x_col], valid_data[y_col])
        spearman_r, spearman_p = spearmanr(valid_data[x_col], valid_data[y_col])
        
        # Add correlation text
        text_str = f'Pearson r = {pearson_r:.3f} (p={pearson_p:.2e})\n'
        text_str += f'Spearman ρ = {spearman_r:.3f} (p={spearman_p:.2e})\n'
        text_str += f'N = {len(valid_data)} complexes'
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, 
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=13, fontweight='bold')
    ax.set_ylabel(y_col.replace('_', ' ').upper().replace('RMSD', 'RMSD (Å)'), 
                  fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

# Create comprehensive comparison plots
output_dir = EVAL_DIR

print("\nGenerating comparison plots for different RMSD aggregations...")

# Create 3x2 grid: rows = Mean/Min/Max RMSD, columns = Ligand iPTM / iPTM
fig, axes = plt.subplots(3, 2, figsize=(16, 20))

rmsd_types = [
    ('rmsd_mean', 'Mean RMSD'),
    ('rmsd_min', 'Min RMSD'),
    ('rmsd_max', 'Max RMSD')
]

conf_types = [
    ('ligand_iptm_mean', 'Mean Ligand iPTM'),
    ('iptm_mean', 'Mean iPTM')
]

for row_idx, (rmsd_col, rmsd_label) in enumerate(rmsd_types):
    for col_idx, (conf_col, conf_label) in enumerate(conf_types):
        ax = axes[row_idx, col_idx]
        
        # Plot orthosteric and allosteric
        ortho = df[df['type'] == 'orthosteric']
        allo = df[df['type'] == 'allosteric']
        
        ax.scatter(ortho[conf_col], ortho[rmsd_col], alpha=0.7, s=100, 
                   color='#2E86AB', edgecolors='black', linewidths=1,
                   label='Orthosteric')
        ax.scatter(allo[conf_col], allo[rmsd_col], alpha=0.7, s=100, 
                   color='#A23B72', edgecolors='black', linewidths=1,
                   label='Allosteric')
        
        # Add trend line
        valid_data = df[[conf_col, rmsd_col]].dropna()
        if len(valid_data) > 1:
            z = np.polyfit(valid_data[conf_col], valid_data[rmsd_col], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid_data[conf_col].min(), valid_data[conf_col].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
            
            pearson_r, pearson_p = pearsonr(valid_data[conf_col], valid_data[rmsd_col])
            spearman_r, spearman_p = spearmanr(valid_data[conf_col], valid_data[rmsd_col])
            
            text_str = f'r = {pearson_r:.3f} (p={pearson_p:.2e})\n'
            text_str += f'ρ = {spearman_r:.3f} (p={spearman_p:.2e})'
            ax.text(0.05, 0.95, text_str, transform=ax.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        ax.set_xlabel(conf_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{rmsd_label} (Å)', fontsize=12, fontweight='bold')
        ax.set_title(f'{rmsd_label} vs {conf_label}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if row_idx == 0 and col_idx == 1:
            ax.legend(loc='upper right', framealpha=0.9)

plt.suptitle('RMSD vs Confidence Metrics (Per Complex Aggregation)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'rmsd_vs_confidence_per_complex_grid.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'rmsd_vs_confidence_per_complex_grid.png'}")
plt.close()

# Create separate plots for orthosteric vs allosteric with mean RMSD
print("\nGenerating separate orthosteric vs allosteric plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Orthosteric - ligand_iptm_mean vs rmsd_mean
ax = axes[0, 0]
valid = ortho_df[['ligand_iptm_mean', 'rmsd_mean']].dropna()
ax.scatter(valid['ligand_iptm_mean'], valid['rmsd_mean'], 
           alpha=0.7, s=120, color='#2E86AB', edgecolors='black', linewidths=1)
if len(valid) > 1:
    z = np.polyfit(valid['ligand_iptm_mean'], valid['rmsd_mean'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid['ligand_iptm_mean'].min(), valid['ligand_iptm_mean'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2.5)
    pearson_r, pearson_p = pearsonr(valid['ligand_iptm_mean'], valid['rmsd_mean'])
    spearman_r, spearman_p = spearmanr(valid['ligand_iptm_mean'], valid['rmsd_mean'])
    ax.text(0.05, 0.95, f'r = {pearson_r:.3f} (p={pearson_p:.2e})\nρ = {spearman_r:.3f} (p={spearman_p:.2e})', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
ax.set_xlabel('Mean Ligand iPTM', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean RMSD (Å)', fontsize=12, fontweight='bold')
ax.set_title('Orthosteric: Mean RMSD vs Mean Ligand iPTM', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Orthosteric - iptm_mean vs rmsd_mean
ax = axes[0, 1]
valid = ortho_df[['iptm_mean', 'rmsd_mean']].dropna()
ax.scatter(valid['iptm_mean'], valid['rmsd_mean'], 
           alpha=0.7, s=120, color='#2E86AB', edgecolors='black', linewidths=1)
if len(valid) > 1:
    z = np.polyfit(valid['iptm_mean'], valid['rmsd_mean'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid['iptm_mean'].min(), valid['iptm_mean'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2.5)
    pearson_r, pearson_p = pearsonr(valid['iptm_mean'], valid['rmsd_mean'])
    spearman_r, spearman_p = spearmanr(valid['iptm_mean'], valid['rmsd_mean'])
    ax.text(0.05, 0.95, f'r = {pearson_r:.3f} (p={pearson_p:.2e})\nρ = {spearman_r:.3f} (p={spearman_p:.2e})', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
ax.set_xlabel('Mean iPTM', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean RMSD (Å)', fontsize=12, fontweight='bold')
ax.set_title('Orthosteric: Mean RMSD vs Mean iPTM', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Allosteric - ligand_iptm_mean vs rmsd_mean
ax = axes[1, 0]
valid = allo_df[['ligand_iptm_mean', 'rmsd_mean']].dropna()
ax.scatter(valid['ligand_iptm_mean'], valid['rmsd_mean'], 
           alpha=0.7, s=120, color='#A23B72', edgecolors='black', linewidths=1)
if len(valid) > 1:
    z = np.polyfit(valid['ligand_iptm_mean'], valid['rmsd_mean'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid['ligand_iptm_mean'].min(), valid['ligand_iptm_mean'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2.5)
    pearson_r, pearson_p = pearsonr(valid['ligand_iptm_mean'], valid['rmsd_mean'])
    spearman_r, spearman_p = spearmanr(valid['ligand_iptm_mean'], valid['rmsd_mean'])
    ax.text(0.05, 0.95, f'r = {pearson_r:.3f} (p={pearson_p:.2e})\nρ = {spearman_r:.3f} (p={spearman_p:.2e})', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
ax.set_xlabel('Mean Ligand iPTM', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean RMSD (Å)', fontsize=12, fontweight='bold')
ax.set_title('Allosteric: Mean RMSD vs Mean Ligand iPTM', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Allosteric - iptm_mean vs rmsd_mean
ax = axes[1, 1]
valid = allo_df[['iptm_mean', 'rmsd_mean']].dropna()
ax.scatter(valid['iptm_mean'], valid['rmsd_mean'], 
           alpha=0.7, s=120, color='#A23B72', edgecolors='black', linewidths=1)
if len(valid) > 1:
    z = np.polyfit(valid['iptm_mean'], valid['rmsd_mean'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid['iptm_mean'].min(), valid['iptm_mean'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2.5)
    pearson_r, pearson_p = pearsonr(valid['iptm_mean'], valid['rmsd_mean'])
    spearman_r, spearman_p = spearmanr(valid['iptm_mean'], valid['rmsd_mean'])
    ax.text(0.05, 0.95, f'r = {pearson_r:.3f} (p={pearson_p:.2e})\nρ = {spearman_r:.3f} (p={spearman_p:.2e})', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
ax.set_xlabel('Mean iPTM', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean RMSD (Å)', fontsize=12, fontweight='bold')
ax.set_title('Allosteric: Mean RMSD vs Mean iPTM', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.suptitle('Mean RMSD vs Confidence: Orthosteric vs Allosteric Comparison', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'rmsd_mean_vs_confidence_ortho_allo.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'rmsd_mean_vs_confidence_ortho_allo.png'}")
plt.close()

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nGenerated files:")
print(f"  - {output_file}")
print(f"  - {corr_file}")
print(f"  - Multiple PNG plots in {output_dir}/")
print(f"\nKey findings:")
print(f"  - Analyzed {len(df)} complexes (20 orthosteric, 20 allosteric)")
print(f"  - Each complex has 5 model predictions aggregated")

