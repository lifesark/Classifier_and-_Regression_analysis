"""
Main training script for violence detection models.

This script provides a comprehensive training pipeline with support for
different model architectures, optimizers, schedulers, and training techniques.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import os
import time
import logging
from typing import Dict, Any, Tuple, Optional, List
import json
from tqdm import tqdm
import wandb

# Import local modules
from ..models import ModelFactory, save_model_checkpoint, load_model_checkpoint
from ..data import create_data_loaders, get_dataset_stats
from .config import TrainingConfig
from .evaluate import ModelEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ViolenceDetectionTrainer:
    """Main trainer class for violence detection models."""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = config.device
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('-inf') if config.monitor_mode == 'max' else float('inf')
        self.patience_counter = 0
        self.training_history = []
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Evaluator
        self.evaluator = ModelEvaluator()
        
        # Set random seeds for reproducibility
        self._set_seeds()
        
        logger.info(f"Trainer initialized with device: {self.device}")
    
    def _set_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
        
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def setup_data(self) -> None:
        """Set up data loaders."""
        logger.info("Setting up data loaders...")
        
        data_loaders = create_data_loaders(
            data_dir=self.config.data_dir,
            batch_size=self.config.batch_size,
            frames_per_clip=self.config.frames_per_clip,
            target_size=self.config.target_size,
            num_workers=self.config.num_workers,
            cache_data=self.config.cache_data,
            train_split=self.config.train_split,
            val_split=self.config.val_split
        )
        
        self.train_loader = data_loaders['train']
        self.val_loader = data_loaders['val']
        self.test_loader = data_loaders['test']
        
        # Print dataset statistics
        for split, loader in data_loaders.items():
            stats = get_dataset_stats(loader)
            logger.info(f"{split.upper()} - Samples: {stats['total_samples']}, "
                       f"Classes: {stats['class_distribution']}")
    
    def setup_model(self) -> None:
        """Set up the model."""
        logger.info(f"Setting up {self.config.model_type} model...")
        
        # Create model
        self.model = ModelFactory.create_model(
            self.config.model_type, 
            self.config.get_model_config()
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Multi-GPU support
        if self.config.num_gpus > 1:
            self.model = nn.DataParallel(self.model)
            logger.info(f"Using {self.config.num_gpus} GPUs for training")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model parameters: {trainable_params:,} trainable, {total_params:,} total")
    
    def setup_criterion(self) -> None:
        """Set up the loss criterion."""
        if self.config.loss_type == "cross_entropy":
            if self.config.class_weights is not None:
                weights = torch.tensor(self.config.class_weights).to(self.device)
            else:
                weights = None
            
            self.criterion = nn.CrossEntropyLoss(
                weight=weights,
                label_smoothing=self.config.label_smoothing
            )
        
        elif self.config.loss_type == "focal":
            from .losses import FocalLoss
            self.criterion = FocalLoss(
                alpha=self.config.class_weights,
                gamma=2.0,
                reduction='mean'
            )
        
        else:
            raise ValueError(f"Unsupported loss type: {self.config.loss_type}")
        
        logger.info(f"Set up {self.config.loss_type} loss criterion")
    
    def setup_optimizer(self) -> None:
        """Set up the optimizer."""
        optimizer_config = self.config.get_optimizer_config()
        
        if self.config.optimizer_type == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                **optimizer_config
            )
        elif self.config.optimizer_type == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                **optimizer_config
            )
        elif self.config.optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                momentum=0.9,
                **optimizer_config
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
        
        logger.info(f"Set up {self.config.optimizer_type} optimizer")
    
    def setup_scheduler(self) -> None:
        """Set up the learning rate scheduler."""
        scheduler_config = self.config.get_scheduler_config()
        
        if self.config.scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                **scheduler_config
            )
        elif self.config.scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                **scheduler_config
            )
        elif self.config.scheduler_type == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                **scheduler_config
            )
        else:
            self.scheduler = None
        
        if self.scheduler is not None:
            logger.info(f"Set up {self.config.scheduler_type} scheduler")
    
    def setup_mixed_precision(self) -> None:
        """Set up mixed precision training."""
        if self.config.mixed_precision and torch.cuda.is_available():
            self.scaler = GradScaler()
            logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (videos, labels) in enumerate(progress_bar):
            videos = videos.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(videos)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip_value > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_value
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            else:
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_value
                    )
                
                # Optimizer step
                self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Update progress bar
            if batch_idx % self.config.log_frequency == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.0 * correct_predictions / total_samples:.2f}%',
                    'LR': f'{current_lr:.2e}'
                })
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(self.train_loader)
        epoch_accuracy = 100.0 * correct_predictions / total_samples
        
        return {
            'train_loss': epoch_loss,
            'train_accuracy': epoch_accuracy
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for videos, labels in tqdm(self.val_loader, desc="Validation"):
                videos = videos.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass
                if self.scaler is not None:
                    with autocast():
                        outputs = self.model(videos)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(videos)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                # Store predictions and labels for metrics calculation
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = total_loss / len(self.val_loader)
        val_metrics = self.evaluator.calculate_metrics(all_labels, all_predictions)
        val_metrics['val_loss'] = val_loss
        
        return val_metrics
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            metrics: Current metrics
            is_best: Whether this is the best model so far
        """
        checkpoint_dir = self.config.save_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{self.current_epoch + 1}.pt")
        
        save_model_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch + 1,
            loss=metrics.get('val_loss', 0.0),
            metrics=metrics,
            save_path=checkpoint_path,
            model_config=self.config.get_model_config()
        )
        
        # Save best model
        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            save_model_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=self.current_epoch + 1,
                loss=metrics.get('val_loss', 0.0),
                metrics=metrics,
                save_path=best_path,
                model_config=self.config.get_model_config()
            )
            logger.info(f"Saved best model with {self.config.monitor_metric}: {metrics[self.config.monitor_metric]:.4f}")
    
    def should_stop_early(self, current_metric: float) -> bool:
        """
        Check if training should stop early.
        
        Args:
            current_metric: Current value of the monitored metric
            
        Returns:
            True if training should stop
        """
        if self.config.monitor_mode == 'max':
            improved = current_metric > self.best_metric + self.config.early_stopping_min_delta
        else:
            improved = current_metric < self.best_metric - self.config.early_stopping_min_delta
        
        if improved:
            self.best_metric = current_metric
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.early_stopping_patience
    
    def train(self) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Returns:
            Training history dictionary
        """
        logger.info("Starting training...")
        
        # Setup all components
        self.setup_data()
        self.setup_model()
        self.setup_criterion()
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_mixed_precision()
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate epoch
            if epoch % self.config.eval_frequency == 0:
                val_metrics = self.validate_epoch()
            else:
                val_metrics = {}
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(epoch_metrics.get('val_loss', 0))
                else:
                    self.scheduler.step()
            
            # Log metrics
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                       f"Time: {epoch_time:.2f}s - "
                       f"Train Loss: {train_metrics['train_loss']:.4f} - "
                       f"Train Acc: {train_metrics['train_accuracy']:.2f}%")
            
            if val_metrics:
                logger.info(f"Val Loss: {val_metrics['val_loss']:.4f} - "
                           f"Val Acc: {val_metrics.get('val_accuracy', 0):.2f}% - "
                           f"Val F1: {val_metrics.get('val_f1', 0):.4f}")
            
            # Save training history
            self.training_history.append(epoch_metrics)
            
            # Save checkpoint
            is_best = False
            if val_metrics and self.config.monitor_metric in val_metrics:
                current_metric = val_metrics[self.config.monitor_metric]
                
                if self.config.monitor_mode == 'max':
                    is_best = current_metric > self.best_metric
                else:
                    is_best = current_metric < self.best_metric
                
                if is_best:
                    self.best_metric = current_metric
            
            # Save checkpoint
            if (self.config.save_best_only and is_best) or \
               (not self.config.save_best_only and (epoch + 1) % self.config.save_frequency == 0):
                self.save_checkpoint(epoch_metrics, is_best)
            
            # Early stopping
            if val_metrics and self.config.monitor_metric in val_metrics:
                if self.should_stop_early(val_metrics[self.config.monitor_metric]):
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Final evaluation on test set
        if self.config.test_at_end and self.test_loader is not None:
            logger.info("Running final evaluation on test set...")
            test_metrics = self.evaluate_test_set()
            logger.info(f"Test Results - Accuracy: {test_metrics.get('test_accuracy', 0):.2f}% - "
                       f"F1: {test_metrics.get('test_f1', 0):.4f}")
        
        logger.info("Training completed!")
        
        return self.training_history
    
    def evaluate_test_set(self) -> Dict[str, float]:
        """
        Evaluate the model on the test set.
        
        Returns:
            Test metrics dictionary
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for videos, labels in tqdm(self.test_loader, desc="Testing"):
                videos = videos.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                if self.scaler is not None:
                    with autocast():
                        outputs = self.model(videos)
                else:
                    outputs = self.model(videos)
                
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_metrics = self.evaluator.calculate_metrics(all_labels, all_predictions)
        
        # Add 'test_' prefix to metrics
        test_metrics = {f'test_{k}': v for k, v in test_metrics.items()}
        
        return test_metrics

def main():
    """Main function for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train violence detection model")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--preset", type=str, choices=["fast_training", "high_accuracy", "3d_cnn_training", "transformer_training", "colab_optimized"],
                       help="Use a configuration preset")
    parser.add_argument("--data_dir", type=str, help="Path to dataset directory")
    parser.add_argument("--save_dir", type=str, help="Directory to save checkpoints")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = TrainingConfig.load(args.config)
    elif args.preset:
        from .config import get_config_preset
        config = get_config_preset(args.preset)
    else:
        config = TrainingConfig()
    
    # Override with command line arguments
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.save_dir:
        config.save_dir = args.save_dir
    
    # Create trainer and train
    trainer = ViolenceDetectionTrainer(config)
    training_history = trainer.train()
    
    # Save training history
    history_path = os.path.join(config.save_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()