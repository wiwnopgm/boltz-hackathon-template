#!/usr/bin/env python3
"""
Analyze variance in confidence metrics across multiple predictions per complex
Compare orthosteric vs allosteric binding modes
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

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
variance_data = []

for idx, row in rmsd_df.iterrows():
    datapoint_id = row['datapoint_id']
    ligand_type = row['type']
    
    # Find the prediction directory
    pred_dir = PREDICTIONS_DIR / f"boltz_results_{datapoint_id}_config_0" / "predictions" / f"{datapoint_id}_config_0"
    
    if not pred_dir.exists():
        print(f"Warning: Prediction directory not found for {datapoint_id}")
        continue
    
    # Collect metrics for all 5 models
    model_ligand_iptms = []
    model_iptms = []
    model_ptms = []
    model_conf_scores = []
    model_plddt = []
    model_rmsds = []
    
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
    
    # Calculate variance/std statistics per complex
    if model_rmsds:
        variance_data.append({
            'datapoint_id': datapoint_id,
            'type': ligand_type,
            # RMSD statistics
            'rmsd_mean': np.mean(model_rmsds),
            'rmsd_std': np.std(model_rmsds, ddof=1),  # sample std
            'rmsd_var': np.var(model_rmsds, ddof=1),  # sample variance
            'rmsd_cv': np.std(model_rmsds, ddof=1) / np.mean(model_rmsds) if np.mean(model_rmsds) > 0 else 0,  # coefficient of variation
            'rmsd_range': np.max(model_rmsds) - np.min(model_rmsds),
            # Ligand iPTM statistics
            'ligand_iptm_mean': np.nanmean(model_ligand_iptms),
            'ligand_iptm_std': np.nanstd(model_ligand_iptms, ddof=1),
            'ligand_iptm_var': np.nanvar(model_ligand_iptms, ddof=1),
            'ligand_iptm_cv': np.nanstd(model_ligand_iptms, ddof=1) / np.nanmean(model_ligand_iptms) if np.nanmean(model_ligand_iptms) > 0 else 0,
            'ligand_iptm_range': np.nanmax(model_ligand_iptms) - np.nanmin(model_ligand_iptms),
            # iPTM statistics
            'iptm_mean': np.nanmean(model_iptms),
            'iptm_std': np.nanstd(model_iptms, ddof=1),
            'iptm_var': np.nanvar(model_iptms, ddof=1),
            'iptm_cv': np.nanstd(model_iptms, ddof=1) / np.nanmean(model_iptms) if np.nanmean(model_iptms) > 0 else 0,
            'iptm_range': np.nanmax(model_iptms) - np.nanmin(model_iptms),
            # Confidence score statistics
            'confidence_score_mean': np.nanmean(model_conf_scores),
            'confidence_score_std': np.nanstd(model_conf_scores, ddof=1),
            'confidence_score_var': np.nanvar(model_conf_scores, ddof=1),
            'confidence_score_cv': np.nanstd(model_conf_scores, ddof=1) / np.nanmean(model_conf_scores) if np.nanmean(model_conf_scores) > 0 else 0,
            'confidence_score_range': np.nanmax(model_conf_scores) - np.nanmin(model_conf_scores),
            # PTM statistics
            'ptm_mean': np.nanmean(model_ptms),
            'ptm_std': np.nanstd(model_ptms, ddof=1),
            'ptm_var': np.nanvar(model_ptms, ddof=1),
            # pLDDT statistics
            'complex_plddt_mean': np.nanmean(model_plddt),
            'complex_plddt_std': np.nanstd(model_plddt, ddof=1),
            'complex_plddt_var': np.nanvar(model_plddt, ddof=1),
            'n_models': len(model_rmsds)
        })

# Create DataFrame
df = pd.DataFrame(variance_data)
print(f"\nComputed variance statistics for {len(df)} complexes")
print(f"Orthosteric: {len(df[df['type'] == 'orthosteric'])}")
print(f"Allosteric: {len(df[df['type'] == 'allosteric'])}")

# Save the data
output_file = EVAL_DIR / "confidence_variance_analysis.csv"
df.to_csv(output_file, index=False)
print(f"\nSaved variance data to {output_file}")

# Statistical comparison
print("\n" + "="*80)
print("STATISTICAL COMPARISON: ORTHOSTERIC vs ALLOSTERIC")
print("="*80)

ortho_df = df[df['type'] == 'orthosteric']
allo_df = df[df['type'] == 'allosteric']

# Compare variance in confidence metrics
metrics = [
    ('ligand_iptm_std', 'Ligand iPTM Std Dev'),
    ('iptm_std', 'iPTM Std Dev'),
    ('confidence_score_std', 'Confidence Score Std Dev'),
    ('ligand_iptm_var', 'Ligand iPTM Variance'),
    ('iptm_var', 'iPTM Variance'),
    ('confidence_score_var', 'Confidence Score Variance'),
    ('ligand_iptm_cv', 'Ligand iPTM CV'),
    ('iptm_cv', 'iPTM CV'),
    ('confidence_score_cv', 'Confidence Score CV'),
    ('rmsd_std', 'RMSD Std Dev'),
    ('rmsd_cv', 'RMSD CV'),
]

print("\nComparison of variance metrics:")
print("-" * 80)

comparison_results = []

for metric_col, metric_label in metrics:
    ortho_vals = ortho_df[metric_col].dropna()
    allo_vals = allo_df[metric_col].dropna()
    
    ortho_mean = ortho_vals.mean()
    ortho_median = ortho_vals.median()
    allo_mean = allo_vals.mean()
    allo_median = allo_vals.median()
    
    # T-test (parametric)
    t_stat, t_pval = stats.ttest_ind(ortho_vals, allo_vals)
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_pval = stats.mannwhitneyu(ortho_vals, allo_vals, alternative='two-sided')
    
    print(f"\n{metric_label}:")
    print(f"  Orthosteric:  mean={ortho_mean:.6f}, median={ortho_median:.6f}")
    print(f"  Allosteric:   mean={allo_mean:.6f}, median={allo_median:.6f}")
    print(f"  T-test:       t={t_stat:.4f}, p={t_pval:.4f}")
    print(f"  Mann-Whitney: U={u_stat:.1f}, p={u_pval:.4f}")
    
    significance = "***" if u_pval < 0.001 else "**" if u_pval < 0.01 else "*" if u_pval < 0.05 else "ns"
    
    comparison_results.append({
        'metric': metric_label,
        'ortho_mean': ortho_mean,
        'ortho_median': ortho_median,
        'allo_mean': allo_mean,
        'allo_median': allo_median,
        't_stat': t_stat,
        't_pval': t_pval,
        'u_stat': u_stat,
        'u_pval': u_pval,
        'significance': significance
    })

# Save comparison results
comparison_df = pd.DataFrame(comparison_results)
comparison_file = EVAL_DIR / "variance_comparison_ortho_allo.csv"
comparison_df.to_csv(comparison_file, index=False)
print(f"\nSaved comparison results to {comparison_file}")

# Create visualizations
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

# Color palette
colors = {'orthosteric': '#2E86AB', 'allosteric': '#A23B72'}

# 1. Box plots for variance comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

variance_metrics = [
    ('ligand_iptm_std', 'Ligand iPTM\nStd Dev'),
    ('iptm_std', 'iPTM\nStd Dev'),
    ('confidence_score_std', 'Confidence Score\nStd Dev'),
    ('ligand_iptm_var', 'Ligand iPTM\nVariance'),
    ('iptm_var', 'iPTM\nVariance'),
    ('confidence_score_var', 'Confidence Score\nVariance'),
]

for idx, (metric_col, metric_label) in enumerate(variance_metrics):
    ax = axes[idx // 3, idx % 3]
    
    # Prepare data for boxplot
    plot_data = []
    plot_labels = []
    for binding_type in ['orthosteric', 'allosteric']:
        vals = df[df['type'] == binding_type][metric_col].dropna()
        plot_data.append(vals)
        plot_labels.append(binding_type.capitalize())
    
    # Create box plot
    bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True,
                    widths=0.6, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=8),
                    medianprops=dict(color='black', linewidth=2))
    
    # Color boxes
    for patch, binding_type in zip(bp['boxes'], ['orthosteric', 'allosteric']):
        patch.set_facecolor(colors[binding_type])
        patch.set_alpha(0.7)
    
    # Add individual points
    for i, (vals, binding_type) in enumerate(zip(plot_data, ['orthosteric', 'allosteric'])):
        x = np.random.normal(i+1, 0.04, size=len(vals))
        ax.scatter(x, vals, alpha=0.4, s=40, color=colors[binding_type], edgecolors='black', linewidths=0.5)
    
    # Get p-value for this metric
    result = comparison_df[comparison_df['metric'] == metric_label.replace('\n', ' ')].iloc[0]
    sig = result['significance']
    
    # Add significance annotation
    if sig != 'ns':
        y_max = max([max(d) for d in plot_data])
        y_range = y_max - min([min(d) for d in plot_data])
        ax.plot([1, 2], [y_max + 0.1*y_range, y_max + 0.1*y_range], 'k-', linewidth=1.5)
        ax.text(1.5, y_max + 0.12*y_range, sig, ha='center', fontsize=12, fontweight='bold')
    
    ax.set_ylabel(metric_label, fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', labelsize=11)

plt.suptitle('Variance in Confidence Metrics: Orthosteric vs Allosteric', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(EVAL_DIR / 'confidence_variance_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved: {EVAL_DIR / 'confidence_variance_comparison.png'}")
plt.close()

# 2. Coefficient of Variation comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

cv_metrics = [
    ('ligand_iptm_cv', 'Ligand iPTM CV'),
    ('iptm_cv', 'iPTM CV'),
    ('confidence_score_cv', 'Confidence Score CV'),
]

for idx, (metric_col, metric_label) in enumerate(cv_metrics):
    ax = axes[idx]
    
    # Prepare data
    plot_data = []
    plot_labels = []
    for binding_type in ['orthosteric', 'allosteric']:
        vals = df[df['type'] == binding_type][metric_col].dropna()
        plot_data.append(vals)
        plot_labels.append(binding_type.capitalize())
    
    # Create box plot
    bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True,
                    widths=0.6, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=10),
                    medianprops=dict(color='black', linewidth=2.5))
    
    # Color boxes
    for patch, binding_type in zip(bp['boxes'], ['orthosteric', 'allosteric']):
        patch.set_facecolor(colors[binding_type])
        patch.set_alpha(0.7)
    
    # Add individual points
    for i, (vals, binding_type) in enumerate(zip(plot_data, ['orthosteric', 'allosteric'])):
        x = np.random.normal(i+1, 0.05, size=len(vals))
        ax.scatter(x, vals, alpha=0.5, s=60, color=colors[binding_type], edgecolors='black', linewidths=0.7)
    
    # Get p-value
    result = comparison_df[comparison_df['metric'] == metric_label].iloc[0]
    sig = result['significance']
    p_val = result['u_pval']
    
    # Add significance annotation
    y_max = max([max(d) for d in plot_data])
    y_range = y_max - min([min(d) for d in plot_data])
    if sig != 'ns':
        ax.plot([1, 2], [y_max + 0.1*y_range, y_max + 0.1*y_range], 'k-', linewidth=2)
        ax.text(1.5, y_max + 0.13*y_range, sig, ha='center', fontsize=14, fontweight='bold')
    
    ax.text(0.5, 0.95, f'p = {p_val:.4f}', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_ylabel('Coefficient of Variation', fontsize=13, fontweight='bold')
    ax.set_title(metric_label, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', labelsize=12)

plt.suptitle('Coefficient of Variation (CV): Orthosteric vs Allosteric', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(EVAL_DIR / 'cv_comparison_ortho_allo.png', dpi=300, bbox_inches='tight')
print(f"Saved: {EVAL_DIR / 'cv_comparison_ortho_allo.png'}")
plt.close()

# 3. Scatter plot: Variance vs Mean RMSD
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

variance_cols = [
    ('ligand_iptm_std', 'Ligand iPTM Std Dev'),
    ('iptm_std', 'iPTM Std Dev'),
    ('confidence_score_std', 'Confidence Score Std Dev'),
]

for idx, (var_col, var_label) in enumerate(variance_cols):
    ax = axes[idx]
    
    # Plot orthosteric and allosteric
    for binding_type in ['orthosteric', 'allosteric']:
        subset = df[df['type'] == binding_type]
        ax.scatter(subset['rmsd_mean'], subset[var_col],
                  alpha=0.7, s=100, color=colors[binding_type],
                  label=binding_type.capitalize(), edgecolors='black', linewidths=1)
    
    # Add trend line for all data
    valid_data = df[['rmsd_mean', var_col]].dropna()
    if len(valid_data) > 1:
        from scipy.stats import pearsonr
        z = np.polyfit(valid_data['rmsd_mean'], valid_data[var_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid_data['rmsd_mean'].min(), valid_data['rmsd_mean'].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Linear fit')
        
        r, pval = pearsonr(valid_data['rmsd_mean'], valid_data[var_col])
        ax.text(0.05, 0.95, f'r = {r:.3f}\np = {pval:.3e}',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Mean RMSD (Å)', fontsize=13, fontweight='bold')
    ax.set_ylabel(var_label, fontsize=13, fontweight='bold')
    ax.set_title(f'RMSD vs {var_label}', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=11)
    ax.grid(True, alpha=0.3)

plt.suptitle('Confidence Metric Variance vs Mean RMSD', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(EVAL_DIR / 'variance_vs_rmsd.png', dpi=300, bbox_inches='tight')
print(f"Saved: {EVAL_DIR / 'variance_vs_rmsd.png'}")
plt.close()

# 4. Distribution histograms
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx, (metric_col, metric_label) in enumerate(variance_metrics):
    ax = axes[idx // 3, idx % 3]
    
    # Get data for both types
    ortho_vals = ortho_df[metric_col].dropna()
    allo_vals = allo_df[metric_col].dropna()
    
    # Plot histograms
    ax.hist(ortho_vals, bins=10, alpha=0.6, color=colors['orthosteric'], 
            label='Orthosteric', edgecolor='black', linewidth=1.2)
    ax.hist(allo_vals, bins=10, alpha=0.6, color=colors['allosteric'], 
            label='Allosteric', edgecolor='black', linewidth=1.2)
    
    # Add vertical lines for means
    ax.axvline(ortho_vals.mean(), color=colors['orthosteric'], 
              linestyle='--', linewidth=2.5, alpha=0.8)
    ax.axvline(allo_vals.mean(), color=colors['allosteric'], 
              linestyle='--', linewidth=2.5, alpha=0.8)
    
    ax.set_xlabel(metric_label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Distribution of Variance Metrics: Orthosteric vs Allosteric', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(EVAL_DIR / 'variance_distributions.png', dpi=300, bbox_inches='tight')
print(f"Saved: {EVAL_DIR / 'variance_distributions.png'}")
plt.close()

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nGenerated files:")
print(f"  - {output_file}")
print(f"  - {comparison_file}")
print(f"  - confidence_variance_comparison.png")
print(f"  - cv_comparison_ortho_allo.png")
print(f"  - variance_vs_rmsd.png")
print(f"  - variance_distributions.png")

