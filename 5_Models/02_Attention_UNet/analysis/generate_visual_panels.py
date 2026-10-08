import os
import sys
import json
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

models_root = r'f:\BUSI-segmentation-architecture-study\5_Models'
common_root = os.path.join(models_root, '00_Common')
att_root = os.path.join(models_root, '02_Attention_UNet')
sys.path.insert(0, common_root)
sys.path.insert(0, att_root)

from datasets.busi_dataset import BUSIDataset
from datasets.transforms import get_val_transforms
from model import AttentionUNet
from utils.checkpoint import load_checkpoint

audit_dir = os.path.join(att_root, 'analysis', 'per_image_validation_audit')
panels_dir = os.path.join(audit_dir, 'case_overlays')
os.makedirs(panels_dir, exist_ok=True)

df_all = pd.read_csv(os.path.join(audit_dir, 'validation_per_image_metrics.csv'))
worst20_ids = df_all.head(20)['image_id'].tolist()

ckpt_path = os.path.join(att_root, 'experiments', 'EXP-ARCH-02_ATTENTION_UNET', 'checkpoints', 'best_model.pth')
val_manifest = os.path.join(models_root, 'MASTER_SPLIT', 'val_20.csv')
val_transform = get_val_transforms(input_size=(256, 256))
val_dataset = BUSIDataset(manifest_path=val_manifest, transform=val_transform)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AttentionUNet(in_channels=1, out_channels=1, base_channels=64, bilinear=False).to(device)
ckpt = load_checkpoint(ckpt_path, model, device=device)
model.eval()

# Store worst 20 visual items
worst20_data = []

with torch.no_grad():
    for idx in range(len(val_dataset)):
        sample = val_dataset[idx]
        sid = sample['sample_id']
        if sid in worst20_ids:
            img_t = sample['image']
            mask_t = sample['mask']
            img_in = img_t.unsqueeze(0).to(device)
            out = model(img_in)
            prob = torch.sigmoid(out)
            pred_b = (prob > 0.5).squeeze().cpu().numpy().astype(bool)
            gt_b = (mask_t.squeeze().cpu().numpy() > 0.5).astype(bool)
            img_np = img_t.squeeze().cpu().numpy()
            
            row = df_all[df_all['image_id'] == sid].iloc[0].to_dict()
            worst20_data.append({
                'image_id': sid,
                'img': img_np,
                'gt': gt_b,
                'pred': pred_b,
                'row': row
            })

# Sort according to rank in df_all
worst20_data.sort(key=lambda x: worst20_ids.index(x['image_id']))

# Color overlay creation helper
def make_overlay(img_np, gt_b, pred_b):
    rgb = np.stack([img_np, img_np, img_np], axis=-1)
    
    # GT only: Green (0, 1, 0)
    # Pred only: Red (1, 0, 0)
    # Overlap: Yellow (1, 1, 0)
    gt_only = np.logical_and(gt_b, ~pred_b)
    pred_only = np.logical_and(~gt_b, pred_b)
    overlap = np.logical_and(gt_b, pred_b)
    
    overlay = rgb.copy()
    alpha = 0.5
    
    overlay[gt_only, 0] = overlay[gt_only, 0] * (1 - alpha) + 0.0 * alpha
    overlay[gt_only, 1] = overlay[gt_only, 1] * (1 - alpha) + 1.0 * alpha
    overlay[gt_only, 2] = overlay[gt_only, 2] * (1 - alpha) + 0.0 * alpha
    
    overlay[pred_only, 0] = overlay[pred_only, 0] * (1 - alpha) + 1.0 * alpha
    overlay[pred_only, 1] = overlay[pred_only, 1] * (1 - alpha) + 0.0 * alpha
    overlay[pred_only, 2] = overlay[pred_only, 2] * (1 - alpha) + 0.0 * alpha
    
    overlay[overlap, 0] = overlay[overlap, 0] * (1 - alpha) + 1.0 * alpha
    overlay[overlap, 1] = overlay[overlap, 1] * (1 - alpha) + 1.0 * alpha
    overlay[overlap, 2] = overlay[overlap, 2] * (1 - alpha) + 0.0 * alpha
    
    return np.clip(overlay, 0.0, 1.0)

# Generate individual case overlays
for rank, item in enumerate(worst20_data, start=1):
    row = item['row']
    sid = item['image_id']
    img_np = item['img']
    gt_b = item['gt']
    pred_b = item['pred']
    overlay = make_overlay(img_np, gt_b, pred_b)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=150)
    fig.patch.set_facecolor('#1e1e1e')
    
    for ax in axes:
        ax.axis('off')
        
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title('Original Ultrasound', color='white', fontsize=11, pad=6)
    
    axes[1].imshow(gt_b, cmap='gray')
    gt_area = int(row['ground_truth_lesion_area_pixels'])
    axes[1].set_title(f'Ground Truth ({gt_area} px)', color='white', fontsize=11, pad=6)
    
    axes[2].imshow(pred_b, cmap='gray')
    pred_area = int(row['predicted_lesion_area_pixels'])
    axes[2].set_title(f'Predicted Mask ({pred_area} px)', color='white', fontsize=11, pad=6)
    
    axes[3].imshow(overlay)
    axes[3].set_title('Composite Overlay', color='white', fontsize=11, pad=6)
    
    # Legend
    patch_gt = mpatches.Patch(color='#00FF00', label='Ground Truth Only (FN)')
    patch_pred = mpatches.Patch(color='#FF0000', label='Prediction Only (FP)')
    patch_overlap = mpatches.Patch(color='#FFFF00', label='Overlap (TP)')
    axes[3].legend(handles=[patch_gt, patch_pred, patch_overlap], loc='lower right', facecolor='#2b2b2b', edgecolor='none', labelcolor='white', fontsize=8)
    
    hd95_val = row['hd95_pixels']
    hd95_str = f"{hd95_val:.1f} px" if row['hd95_valid_status'] == 'valid' and not np.isnan(hd95_val) else 'Invalid/NaN'
    title_text = (
        f"Rank #{rank:02d} | Case: {sid} | Class: {str(row['class_label']).upper()} | "
        f"Dice: {row['dice']:.4f} | IoU: {row['iou']:.4f} | HD95: {hd95_str}\n"
        f"Category: {row['failure_category']} | Recall: {row['recall']:.3f} | Precision: {row['precision']:.3f}"
    )
    fig.suptitle(title_text, color='#00e5ff', fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    clean_sid = sid.replace('.png', '').replace(' ', '_').replace('(', '').replace(')', '')
    save_fig_path = os.path.join(panels_dir, f'rank_{rank:02d}_{clean_sid}.png')
    plt.savefig(save_fig_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)

print(f"Saved 20 individual case panels in {panels_dir}")

# Generate 5x4 Contact Sheet
fig, axes = plt.subplots(5, 4, figsize=(20, 24), dpi=150)
fig.patch.set_facecolor('#121212')

for idx, item in enumerate(worst20_data):
    r = idx // 4
    c = idx % 4
    ax = axes[r, c]
    row = item['row']
    sid = item['image_id']
    overlay = make_overlay(item['img'], item['gt'], item['pred'])
    
    ax.imshow(overlay)
    ax.axis('off')
    hd95_val = row['hd95_pixels']
    hd95_str = f"{hd95_val:.1f}" if row['hd95_valid_status'] == 'valid' and not np.isnan(hd95_val) else "NaN"
    ax.set_title(
        f"#{idx+1:02d} {sid} ({str(row['class_label'])[0].upper()})\n"
        f"DSC: {row['dice']:.3f} | IoU: {row['iou']:.3f} | HD95: {hd95_str}\n"
        f"GT: {int(row['ground_truth_lesion_area_pixels'])}px | PR: {int(row['predicted_lesion_area_pixels'])}px",
        color='white', fontsize=9, pad=3
    )

patch_gt = mpatches.Patch(color='#00FF00', label='GT Only (FN)')
patch_pred = mpatches.Patch(color='#FF0000', label='Pred Only (FP)')
patch_overlap = mpatches.Patch(color='#FFFF00', label='Overlap (TP)')
fig.legend(handles=[patch_gt, patch_pred, patch_overlap], loc='upper right', facecolor='#2b2b2b', edgecolor='none', labelcolor='white', fontsize=12)

fig.suptitle('BUSI Attention U-Net Validation Failure Audit: Worst 20 Cases (Ascending Dice)', color='#00e5ff', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.985])
contact_sheet_path = os.path.join(audit_dir, 'worst_20_contact_sheet.png')
plt.savefig(contact_sheet_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close(fig)
print('Saved contact sheet to', contact_sheet_path)
