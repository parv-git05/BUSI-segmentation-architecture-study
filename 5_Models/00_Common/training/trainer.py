"""
Reusable Training Engine for BUSI Phase-1 Segmentation Architecture Benchmark.
Decoupled from model architectures so U-Net, Attention U-Net, ResUNet, AttResUNet, and U-Net++
can be trained using identical protocol, optimization, metric computation, and checkpointing logic.
"""

import os
import sys
import time
from typing import Dict, Any, Tuple, Optional, List, Union
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure 00_Common root is in sys.path for relative/absolute import compatibility
common_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if common_root not in sys.path:
    sys.path.insert(0, common_root)

from configs.experiment_config import ExperimentConfig
from utils.seed import seed_everything
from utils.logger import setup_logger
from utils.checkpoint import save_checkpoint
from utils.experiment_tracker import ExperimentTracker
from metrics.segmentation_metrics import calculate_batch_metrics, aggregate_metrics
from visualization.visualizer import plot_training_curves, plot_qualitative_predictions, plot_metric_distributions


class SegmentationTrainer:
    """
    Architecture-agnostic segmentation trainer.
    Receives any PyTorch nn.Module architecture and executes training, validation,
    metric calculation, checkpointing, and visualization according to ExperimentConfig.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        config: ExperimentConfig,
        scheduler: Optional[Any] = None,
        tracker: Optional[ExperimentTracker] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        self.config = config
        
        # Enforce global reproducibility seed
        seed_everything(self.config.seed)

        # Device assignment
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        # Setup tracker and logging
        if tracker is not None:
            self.tracker = tracker
        else:
            self.tracker = ExperimentTracker(
                output_root=self.config.output_dir,
                experiment_name=self.config.experiment_name
            )

        self.logger = self.tracker.logger
        self.tracker.save_config(self.config.to_dict())

        # History Tracking
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "learning_rate": []
        }
        self.best_metric_val = -float("inf")
        self.best_epoch = 0

    def train_epoch(self, epoch: int) -> float:
        """Runs one full training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)

        for batch_idx, sample in enumerate(self.train_loader):
            images = sample["image"].to(self.device)
            masks = sample["mask"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, n_batches)
        return avg_loss

    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> Tuple[float, Dict[str, float], pd.DataFrame, List[Dict[str, Any]]]:
        """
        Runs full validation evaluation.
        Computes validation loss and full metric suite (Dice, IoU, Precision, Recall, Specificity, HD95, NSD@1,2,3).
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = len(self.val_loader)

        all_sample_metrics: List[Dict[str, Any]] = []
        visual_images, visual_gts, visual_preds, visual_ids = [], [], [], []

        for batch_idx, sample in enumerate(self.val_loader):
            images = sample["image"].to(self.device)
            masks = sample["mask"].to(self.device)
            sample_ids = sample.get("sample_id", [f"val_{batch_idx}_{i}" for i in range(images.size(0))])

            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            total_loss += loss.item()

            # Apply sigmoid thresholding for metric calculation
            preds_prob = torch.sigmoid(outputs)
            preds_binary = (preds_prob > 0.5).to(torch.float32)

            # Compute batch metrics
            batch_metrics = calculate_batch_metrics(
                preds=preds_binary,
                gts=masks,
                threshold=0.5,
                sample_ids=sample_ids
            )
            all_sample_metrics.extend(batch_metrics)

            # Collect visualization samples from the first batch
            if batch_idx == 0:
                visual_images = images.cpu().numpy()
                visual_gts = masks.cpu().numpy()
                visual_preds = preds_binary.cpu().numpy()
                visual_ids = sample_ids

        avg_loss = total_loss / max(1, n_batches)
        agg_metrics = aggregate_metrics(all_sample_metrics)
        df_per_sample = pd.DataFrame(all_sample_metrics)

        # Generate qualitative plot for first batch
        if len(visual_images) > 0:
            qual_path = os.path.join(self.tracker.qualitative_dir, f"epoch_{epoch:03d}_qualitative.png")
            plot_qualitative_predictions(
                images=visual_images,
                gt_masks=visual_gts,
                pred_masks=visual_preds,
                sample_ids=visual_ids,
                metrics_list=all_sample_metrics[:len(visual_images)],
                save_path=qual_path,
                max_samples=6
            )

        return avg_loss, agg_metrics, df_per_sample, all_sample_metrics

    def fit(self) -> Dict[str, Any]:
        """
        Executes the full training loop for config.epochs.
        Monitors target validation metric, updates learning rate scheduler,
        and saves latest and best model checkpoints.
        """
        self.logger.info(f"Starting training run: {self.config.experiment_name}")
        self.logger.info(f"Model Architecture: {self.config.architecture_name}")
        self.logger.info(f"Device: {self.device} | Epochs: {self.config.epochs} | Batch Size: {self.config.batch_size}")
        self.logger.info(f"Optimizer: {self.config.optimizer_name} | LR: {self.config.learning_rate} | Loss: {self.config.loss_fn_name}")

        start_time = time.time()

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()

            # 1. Train Epoch
            train_loss = self.train_epoch(epoch)

            # 2. Validation Epoch
            val_loss, agg_metrics, df_per_sample, _ = self.validate_epoch(epoch)

            # 3. Update Learning Rate Scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 4. Record History
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(current_lr)

            for key, val in agg_metrics.items():
                hist_key = f"val_{key}"
                if hist_key not in self.history:
                    self.history[hist_key] = []
                self.history[hist_key].append(val)

            # Log Progress
            epoch_time = time.time() - epoch_start
            dice_mean = agg_metrics.get("dice_mean", 0.0)
            iou_mean = agg_metrics.get("iou_mean", 0.0)
            hd95_mean = agg_metrics.get("hd95_mean", np.nan)
            hd95_str = f"{hd95_mean:.2f}px" if not np.isnan(hd95_mean) else "N/A"

            self.logger.info(
                f"Epoch [{epoch:03d}/{self.config.epochs:03d}] ({epoch_time:.1f}s) - "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Dice: {dice_mean:.4f} | Val IoU: {iou_mean:.4f} | Val HD95: {hd95_str} | LR: {current_lr:.6f}"
            )

            # 5. Checkpointing (Latest & Best)
            is_best = False
            target_metric_key = self.config.target_metric.replace("val_", "")
            current_target_val = agg_metrics.get(f"{target_metric_key}_mean", dice_mean)

            if current_target_val > self.best_metric_val:
                self.best_metric_val = current_target_val
                self.best_epoch = epoch
                is_best = True

            # Save Latest Checkpoint
            latest_path = os.path.join(self.tracker.checkpoints_dir, "latest.pth")
            save_checkpoint(
                filepath=latest_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_metrics=agg_metrics,
                config=self.config.to_dict(),
                seed=self.config.seed,
                experiment_id=self.config.experiment_name
            )

            # Save Best Checkpoint
            if is_best:
                best_path = os.path.join(self.tracker.checkpoints_dir, "best_model.pth")
                save_checkpoint(
                    filepath=best_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_metrics=agg_metrics,
                    config=self.config.to_dict(),
                    seed=self.config.seed,
                    experiment_id=self.config.experiment_name
                )
                self.tracker.save_per_sample_metrics(df_per_sample, prefix="best_val")
                self.logger.info(f"  >>> New Best Model saved! ({self.config.target_metric}: {self.best_metric_val:.4f} at epoch {epoch})")

        total_time = time.time() - start_time
        self.logger.info(f"Training Complete in {total_time/60.0:.2f} minutes. Best Epoch: {self.best_epoch} ({self.best_metric_val:.4f})")

        # 6. Save Artifacts & Visualizations
        self.tracker.save_history_csv(self.history)
        plot_training_curves(
            history=self.history,
            save_path=os.path.join(self.tracker.training_curves_dir, "learning_curves.png"),
            title_suffix=f"({self.config.architecture_name})"
        )

        summary_report = {
            "experiment_name": self.config.experiment_name,
            "architecture_name": self.config.architecture_name,
            "best_epoch": self.best_epoch,
            "best_target_metric": float(self.best_metric_val),
            "total_epochs_trained": self.config.epochs,
            "total_time_seconds": total_time,
            "final_val_metrics": agg_metrics
        }
        self.tracker.save_summary_report(summary_report)

        return summary_report
