#!/usr/bin/env python3
"""
Analyze RMSD vs confidence metrics (ligand_iptm, iptm) for ASOS predictions
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
        
        confidence_data.append({
            'datapoint_id': datapoint_id,
            'type': ligand_type,
            'model_idx': model_idx,
            'rmsd': rmsd,
            'ligand_iptm': conf_data.get('ligand_iptm', np.nan),
            'iptm': conf_data.get('iptm', np.nan),
            'ptm': conf_data.get('ptm', np.nan),
            'confidence_score': conf_data.get('confidence_score', np.nan),
            'complex_plddt': conf_data.get('complex_plddt', np.nan),
        })

# Create DataFrame
df = pd.DataFrame(confidence_data)
print(f"\nCollected metrics for {len(df)} predictions")
print(f"Unique datapoints: {df['datapoint_id'].nunique()}")
print(f"Orthosteric: {len(df[df['type'] == 'orthosteric'])}")
print(f"Allosteric: {len(df[df['type'] == 'allosteric'])}")

# Save the combined data
output_file = EVAL_DIR / "rmsd_confidence_combined.csv"
df.to_csv(output_file, index=False)
print(f"\nSaved combined data to {output_file}")

# Calculate correlations
print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)

def calculate_correlations(data, name):
    """Calculate and display correlations"""
    print(f"\n{name} ({len(data)} predictions)")
    print("-" * 80)
    
    metrics = ['ligand_iptm', 'iptm', 'ptm', 'confidence_score', 'complex_plddt']
    results = []
    
    for metric in metrics:
        valid_data = data[['rmsd', metric]].dropna()
        if len(valid_data) > 1:
            pearson_r, pearson_p = pearsonr(valid_data['rmsd'], valid_data[metric])
            spearman_r, spearman_p = spearmanr(valid_data['rmsd'], valid_data[metric])
            
            print(f"\n{metric.upper()}:")
            print(f"  Pearson r:  {pearson_r:7.4f}  (p={pearson_p:.4e})")
            print(f"  Spearman r: {spearman_r:7.4f}  (p={spearman_p:.4e})")
            print(f"  N = {len(valid_data)}")
            
            results.append({
                'metric': metric,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'n': len(valid_data)
            })
    
    return pd.DataFrame(results)

# Overall correlations
overall_corr = calculate_correlations(df, "OVERALL (All predictions)")

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
corr_file = EVAL_DIR / "correlation_results.csv"
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
        
        ax.scatter(ortho[x_col], ortho[y_col], alpha=0.6, s=60, 
                   label=f'Orthosteric (n={len(ortho)})', color='#2E86AB', edgecolors='black', linewidths=0.5)
        ax.scatter(allo[x_col], allo[y_col], alpha=0.6, s=60, 
                   label=f'Allosteric (n={len(allo)})', color='#A23B72', edgecolors='black', linewidths=0.5)
    else:
        ax.scatter(data[x_col], data[y_col], alpha=0.6, s=60, edgecolors='black', linewidths=0.5)
    
    # Calculate and add trend line
    valid_data = data[[x_col, y_col]].dropna()
    if len(valid_data) > 1:
        z = np.polyfit(valid_data[x_col], valid_data[y_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid_data[x_col].min(), valid_data[x_col].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Linear fit')
        
        # Calculate correlation
        pearson_r, pearson_p = pearsonr(valid_data[x_col], valid_data[y_col])
        spearman_r, spearman_p = spearmanr(valid_data[x_col], valid_data[y_col])
        
        # Add correlation text
        text_str = f'Pearson r = {pearson_r:.3f} (p={pearson_p:.2e})\n'
        text_str += f'Spearman ρ = {spearman_r:.3f} (p={spearman_p:.2e})'
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, 
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=13, fontweight='bold')
    ax.set_ylabel(y_col.replace('_', ' ').upper(), fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

# Create plots for main metrics
output_dir = EVAL_DIR

# 1. Overall plots
print("\nGenerating overall plots...")
create_scatter_plot(df, 'ligand_iptm', 'rmsd', 
                   'RMSD vs Ligand iPTM - All Predictions',
                   output_dir / 'rmsd_vs_ligand_iptm_all.png')

create_scatter_plot(df, 'iptm', 'rmsd', 
                   'RMSD vs iPTM - All Predictions',
                   output_dir / 'rmsd_vs_iptm_all.png')

# 2. Orthosteric only
print("\nGenerating orthosteric plots...")
create_scatter_plot(ortho_df, 'ligand_iptm', 'rmsd', 
                   'RMSD vs Ligand iPTM - Orthosteric',
                   output_dir / 'rmsd_vs_ligand_iptm_orthosteric.png',
                   color_by_type=False)

create_scatter_plot(ortho_df, 'iptm', 'rmsd', 
                   'RMSD vs iPTM - Orthosteric',
                   output_dir / 'rmsd_vs_iptm_orthosteric.png',
                   color_by_type=False)

# 3. Allosteric only
print("\nGenerating allosteric plots...")
create_scatter_plot(allo_df, 'ligand_iptm', 'rmsd', 
                   'RMSD vs Ligand iPTM - Allosteric',
                   output_dir / 'rmsd_vs_ligand_iptm_allosteric.png',
                   color_by_type=False)

create_scatter_plot(allo_df, 'iptm', 'rmsd', 
                   'RMSD vs iPTM - Allosteric',
                   output_dir / 'rmsd_vs_iptm_allosteric.png',
                   color_by_type=False)

# 4. Create combined comparison plot
print("\nGenerating combined comparison plots...")
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Orthosteric - ligand_iptm
ax = axes[0, 0]
ortho_valid = ortho_df[['ligand_iptm', 'rmsd']].dropna()
ax.scatter(ortho_valid['ligand_iptm'], ortho_valid['rmsd'], 
           alpha=0.6, s=60, color='#2E86AB', edgecolors='black', linewidths=0.5)
if len(ortho_valid) > 1:
    z = np.polyfit(ortho_valid['ligand_iptm'], ortho_valid['rmsd'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(ortho_valid['ligand_iptm'].min(), ortho_valid['ligand_iptm'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    pearson_r, pearson_p = pearsonr(ortho_valid['ligand_iptm'], ortho_valid['rmsd'])
    ax.text(0.05, 0.95, f'r = {pearson_r:.3f}\np = {pearson_p:.2e}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.set_xlabel('Ligand iPTM', fontsize=12, fontweight='bold')
ax.set_ylabel('RMSD (Å)', fontsize=12, fontweight='bold')
ax.set_title('Orthosteric - Ligand iPTM', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Orthosteric - iptm
ax = axes[0, 1]
ortho_valid = ortho_df[['iptm', 'rmsd']].dropna()
ax.scatter(ortho_valid['iptm'], ortho_valid['rmsd'], 
           alpha=0.6, s=60, color='#2E86AB', edgecolors='black', linewidths=0.5)
if len(ortho_valid) > 1:
    z = np.polyfit(ortho_valid['iptm'], ortho_valid['rmsd'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(ortho_valid['iptm'].min(), ortho_valid['iptm'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    pearson_r, pearson_p = pearsonr(ortho_valid['iptm'], ortho_valid['rmsd'])
    ax.text(0.05, 0.95, f'r = {pearson_r:.3f}\np = {pearson_p:.2e}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.set_xlabel('iPTM', fontsize=12, fontweight='bold')
ax.set_ylabel('RMSD (Å)', fontsize=12, fontweight='bold')
ax.set_title('Orthosteric - iPTM', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Allosteric - ligand_iptm
ax = axes[1, 0]
allo_valid = allo_df[['ligand_iptm', 'rmsd']].dropna()
ax.scatter(allo_valid['ligand_iptm'], allo_valid['rmsd'], 
           alpha=0.6, s=60, color='#A23B72', edgecolors='black', linewidths=0.5)
if len(allo_valid) > 1:
    z = np.polyfit(allo_valid['ligand_iptm'], allo_valid['rmsd'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(allo_valid['ligand_iptm'].min(), allo_valid['ligand_iptm'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    pearson_r, pearson_p = pearsonr(allo_valid['ligand_iptm'], allo_valid['rmsd'])
    ax.text(0.05, 0.95, f'r = {pearson_r:.3f}\np = {pearson_p:.2e}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.set_xlabel('Ligand iPTM', fontsize=12, fontweight='bold')
ax.set_ylabel('RMSD (Å)', fontsize=12, fontweight='bold')
ax.set_title('Allosteric - Ligand iPTM', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Allosteric - iptm
ax = axes[1, 1]
allo_valid = allo_df[['iptm', 'rmsd']].dropna()
ax.scatter(allo_valid['iptm'], allo_valid['rmsd'], 
           alpha=0.6, s=60, color='#A23B72', edgecolors='black', linewidths=0.5)
if len(allo_valid) > 1:
    z = np.polyfit(allo_valid['iptm'], allo_valid['rmsd'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(allo_valid['iptm'].min(), allo_valid['iptm'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    pearson_r, pearson_p = pearsonr(allo_valid['iptm'], allo_valid['rmsd'])
    ax.text(0.05, 0.95, f'r = {pearson_r:.3f}\np = {pearson_p:.2e}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.set_xlabel('iPTM', fontsize=12, fontweight='bold')
ax.set_ylabel('RMSD (Å)', fontsize=12, fontweight='bold')
ax.set_title('Allosteric - iPTM', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.suptitle('RMSD vs Confidence Metrics: Orthosteric vs Allosteric', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'rmsd_vs_confidence_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'rmsd_vs_confidence_comparison.png'}")
plt.close()

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nGenerated files:")
print(f"  - {output_file}")
print(f"  - {corr_file}")
print(f"  - Multiple PNG plots in {output_dir}/")

