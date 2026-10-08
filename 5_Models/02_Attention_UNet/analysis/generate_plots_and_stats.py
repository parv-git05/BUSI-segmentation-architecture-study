import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

models_root = r'f:\BUSI-segmentation-architecture-study\5_Models'
att_root = os.path.join(models_root, '02_Attention_UNet')
audit_dir = os.path.join(att_root, 'analysis', 'per_image_validation_audit')

df = pd.read_csv(os.path.join(audit_dir, 'validation_per_image_metrics.csv'))

# 1. Distribution Plot: validation_dice_distribution.png
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
fig.patch.set_facecolor('#ffffff')

# Histogram + KDE
sns.histplot(df['dice'], bins=20, kde=True, ax=axes[0], color='#1f77b4', edgecolor='black')
axes[0].axvline(df['dice'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {df['dice'].mean():.4f}")
axes[0].axvline(df['dice'].median(), color='green', linestyle='-', linewidth=2, label=f"Median: {df['dice'].median():.4f}")
axes[0].set_title('Validation Dice Distribution (N=125)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Dice Similarity Coefficient')
axes[0].set_ylabel('Sample Count')
axes[0].legend()
axes[0].grid(True, linestyle=':', alpha=0.6)

# Cumulative Distribution
sorted_dice = np.sort(df['dice'])
yvals = np.arange(1, len(sorted_dice) + 1) / len(sorted_dice)
axes[1].plot(sorted_dice, yvals, marker='.', color='#2ca02c', linewidth=2)
axes[1].axvline(0.25, color='orange', linestyle=':', label='Dice < 0.25 (Catastrophic)')
axes[1].axvline(0.50, color='purple', linestyle=':', label='Dice < 0.50 (Severe Failure)')
axes[1].axvline(0.75, color='blue', linestyle=':', label='Dice < 0.75 (Sub-optimal)')
axes[1].set_title('Empirical Cumulative Distribution Function (ECDF)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Dice Similarity Coefficient')
axes[1].set_ylabel('Cumulative Probability')
axes[1].legend()
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
p1 = os.path.join(audit_dir, 'validation_dice_distribution.png')
plt.savefig(p1, bbox_inches='tight')
plt.close(fig)
print('Saved', p1)

# 2. Lesion Size Analysis Plot: validation_lesion_size_analysis.png
# Quartiles on GT area fraction
q1, q2, q3 = np.percentile(df['ground_truth_lesion_area_fraction'], [25, 50, 75])
def get_size_tier(val):
    if val <= q1:
        return f'Q1: Small (<= {q1*100:.2f}%)'
    elif val <= q2:
        return f'Q2: Medium-Low ({q1*100:.2f}-{q2*100:.2f}%)'
    elif val <= q3:
        return f'Q3: Medium-High ({q2*100:.2f}-{q3*100:.2f}%)'
    else:
        return f'Q4: Large (> {q3*100:.2f}%)'

df['size_tier'] = df['ground_truth_lesion_area_fraction'].apply(get_size_tier)
tier_order = [
    f'Q1: Small (<= {q1*100:.2f}%)',
    f'Q2: Medium-Low ({q1*100:.2f}-{q2*100:.2f}%)',
    f'Q3: Medium-High ({q2*100:.2f}-{q3*100:.2f}%)',
    f'Q4: Large (> {q3*100:.2f}%)'
]

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=150)
fig.patch.set_facecolor('#ffffff')

# Scatter Plot
sns.scatterplot(
    data=df,
    x='ground_truth_lesion_area_fraction',
    y='dice',
    hue='class_label',
    style='class_label',
    s=70,
    ax=axes[0],
    palette={'benign': '#1f77b4', 'malignant': '#d62728'}
)
axes[0].set_title('Lesion Size vs. Dice Performance', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Ground Truth Lesion Area Fraction (relative to 256x256)')
axes[0].set_ylabel('Dice Similarity Coefficient')
axes[0].grid(True, linestyle=':', alpha=0.6)

# Boxplot by Quartiles
sns.boxplot(data=df, x='size_tier', y='dice', order=tier_order, ax=axes[1], palette='Blues')
sns.stripplot(data=df, x='size_tier', y='dice', order=tier_order, color='black', alpha=0.5, jitter=0.2, ax=axes[1])
axes[1].set_title('Dice Performance by Lesion Size Quartile', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Lesion Size Quartile Group')
axes[1].set_ylabel('Dice Similarity Coefficient')
axes[1].tick_params(axis='x', rotation=15)
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
p2 = os.path.join(audit_dir, 'validation_lesion_size_analysis.png')
plt.savefig(p2, bbox_inches='tight')
plt.close(fig)
print('Saved', p2)

# 3. Class Analysis Plot: validation_class_analysis.png
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
fig.patch.set_facecolor('#ffffff')

sns.boxplot(data=df, x='class_label', y='dice', ax=axes[0], palette=['#6baed6', '#fb6a4a'])
sns.stripplot(data=df, x='class_label', y='dice', color='black', alpha=0.6, jitter=0.2, ax=axes[0])
axes[0].set_title('Dice Comparison: Benign vs. Malignant', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Pathology Class')
axes[0].set_ylabel('Dice Similarity Coefficient')
axes[0].grid(True, linestyle=':', alpha=0.6)

# Subgroup failure rates
class_stats = []
for cl in ['benign', 'malignant']:
    sub = df[df['class_label'] == cl]
    class_stats.append({
        'Class': cl.capitalize(),
        'Count': len(sub),
        'Mean Dice': sub['dice'].mean(),
        'Median Dice': sub['dice'].median(),
        'Failure Rate (<0.50)': (sub['dice'] < 0.50).mean() * 100
    })
df_cl_stats = pd.DataFrame(class_stats)

bars = axes[1].bar(df_cl_stats['Class'], df_cl_stats['Failure Rate (<0.50)'], color=['#3182bd', '#de2d26'], width=0.4)
axes[1].set_title('Failure Rate (% with Dice < 0.50) by Class', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Percentage of Samples (%)')
axes[1].set_ylim(0, 30)
for bar in bars:
    yval = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
p3 = os.path.join(audit_dir, 'validation_class_analysis.png')
plt.savefig(p3, bbox_inches='tight')
plt.close(fig)
print('Saved', p3)

# Print detailed quantitative breakdown for the report
print('--- QUANTITATIVE BREAKDOWN ---')
print('Overall Metrics:')
print('Mean:', df['dice'].mean(), 'Std (ddof=0):', np.std(df['dice'].values), 'Std (ddof=1):', df['dice'].std())
print('Median:', df['dice'].median(), 'IQR:', np.percentile(df['dice'], 75) - np.percentile(df['dice'], 25))
print('Min:', df['dice'].min(), 'Max:', df['dice'].max())
print('Percentiles (10, 25, 75, 90):', np.percentile(df['dice'], [10, 25, 75, 90]))
print('Dice < 0.25:', (df['dice'] < 0.25).sum(), f"({(df['dice'] < 0.25).mean()*100:.2f}%)")
print('Dice < 0.50:', (df['dice'] < 0.50).sum(), f"({(df['dice'] < 0.50).mean()*100:.2f}%)")
print('Dice < 0.75:', (df['dice'] < 0.75).sum(), f"({(df['dice'] < 0.75).mean()*100:.2f}%)")
print('Empty Preds:', df['prediction_empty'].sum())
print('Empty GT:', df['ground_truth_empty'].sum())
print('Invalid HD95:', (df['hd95_valid_status'] == 'invalid').sum())

print('\nSize Quartile Breakdown:')
for t in tier_order:
    sub = df[df['size_tier'] == t]
    print(f"{t}: Count={len(sub)}, Mean={sub['dice'].mean():.4f}, Std(ddof=1)={sub['dice'].std():.4f}, Median={sub['dice'].median():.4f}, Fail(<0.50)={((sub['dice']<0.5).mean()*100):.2f}%")

print('\nClass Breakdown:')
for cl in ['benign', 'malignant']:
    sub = df[df['class_label'] == cl]
    print(f"{cl}: Count={len(sub)}, Mean={sub['dice'].mean():.4f}, Std(ddof=1)={sub['dice'].std():.4f}, Median={sub['dice'].median():.4f}, Fail(<0.50)={((sub['dice']<0.5).mean()*100):.2f}%")

print('\nPrediction Behavioral Breakdown:')
print('Empty or near empty (pred_area < 50 px):', (df['predicted_lesion_area_pixels'] < 50).sum())
print('Under-segmentation (ratio < 0.70):', (df['predicted_to_ground_truth_area_ratio'] < 0.70).sum())
print('Over-segmentation (ratio > 1.30):', (df['predicted_to_ground_truth_area_ratio'] > 1.30).sum())
print('Balanced area (0.70 <= ratio <= 1.30):', ((df['predicted_to_ground_truth_area_ratio'] >= 0.70) & (df['predicted_to_ground_truth_area_ratio'] <= 1.30)).sum())
print('Low Dice (<0.50) but High Recall (>0.70):', ((df['dice'] < 0.50) & (df['recall'] > 0.70)).sum())
print('High Recall (>0.80) but Low Precision (<0.50):', ((df['recall'] > 0.80) & (df['precision'] < 0.50)).sum())
print('Poor overlap (Dice < 0.60) despite similar area (0.8 <= ratio <= 1.2):', ((df['dice'] < 0.60) & (df['predicted_to_ground_truth_area_ratio'] >= 0.80) & (df['predicted_to_ground_truth_area_ratio'] <= 1.20)).sum())
print('High HD95 (> 50 px):', (df['hd95_pixels'] > 50).sum())
print('Low NSD@2 (< 0.20):', (df['nsd_at_2'] < 0.20).sum())
