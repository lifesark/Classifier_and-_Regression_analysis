"""
Model evaluation and metrics calculation for violence detection.

This module provides comprehensive evaluation tools including various
metrics, visualizations, and analysis functions.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from tqdm import tqdm
import json
import os

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            class_names: List of class names for visualization
        """
        self.class_names = class_names or ["Non-Violence", "Violence"]
    
    def calculate_metrics(self, 
                         y_true: List[int], 
                         y_pred: List[int],
                         y_proba: Optional[List[List[float]]] = None) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary containing various metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            if i < len(precision_per_class):
                metrics[f'precision_{class_name.lower()}'] = precision_per_class[i]
                metrics[f'recall_{class_name.lower()}'] = recall_per_class[i]
                metrics[f'f1_{class_name.lower()}'] = f1_per_class[i]
        
        # AUC metrics (if probabilities are provided)
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                y_proba_binary = np.array(y_proba)
                if y_proba_binary.ndim == 2 and y_proba_binary.shape[1] == 2:
                    # For binary classification, use probabilities of positive class
                    y_proba_pos = y_proba_binary[:, 1]
                    metrics['auc_roc'] = roc_auc_score(y_true, y_proba_pos)
                    metrics['auc_pr'] = average_precision_score(y_true, y_proba_pos)
            except Exception as e:
                logger.warning(f"Could not calculate AUC metrics: {e}")
        
        return metrics
    
    def plot_confusion_matrix(self,
                            y_true: List[int],
                            y_pred: List[int],
                            save_path: Optional[str] = None,
                            normalize: bool = True) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
            normalize: Whether to normalize the matrix
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_roc_curve(self,
                      y_true: List[int],
                      y_proba: List[List[float]],
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curve for binary classification.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if len(np.unique(y_true)) != 2:
            raise ValueError("ROC curve is only applicable for binary classification")
        
        y_proba_binary = np.array(y_proba)
        if y_proba_binary.ndim == 2 and y_proba_binary.shape[1] == 2:
            y_proba_pos = y_proba_binary[:, 1]
        else:
            y_proba_pos = y_proba_binary
        
        fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
        roc_auc = roc_auc_score(y_true, y_proba_pos)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2,
               label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC)')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        return fig
    
    def plot_precision_recall_curve(self,
                                   y_true: List[int],
                                   y_proba: List[List[float]],
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Precision-Recall curve for binary classification.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if len(np.unique(y_true)) != 2:
            raise ValueError("PR curve is only applicable for binary classification")
        
        y_proba_binary = np.array(y_proba)
        if y_proba_binary.ndim == 2 and y_proba_binary.shape[1] == 2:
            y_proba_pos = y_proba_binary[:, 1]
        else:
            y_proba_pos = y_proba_binary
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba_pos)
        avg_precision = average_precision_score(y_true, y_proba_pos)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='darkorange', lw=2,
               label=f'PR curve (AP = {avg_precision:.2f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PR curve saved to {save_path}")
        
        return fig
    
    def plot_training_history(self,
                            history: List[Dict[str, float]],
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training history.
        
        Args:
            history: List of dictionaries containing training metrics
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        epochs = list(range(1, len(history) + 1))
        
        # Extract metrics
        train_loss = [h.get('train_loss', 0) for h in history]
        val_loss = [h.get('val_loss', 0) for h in history]
        train_acc = [h.get('train_accuracy', 0) for h in history]
        val_acc = [h.get('val_accuracy', 0) for h in history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(epochs, train_loss, 'b-', label='Training Loss')
        if any(val_loss):
            ax1.plot(epochs, val_loss, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy')
        if any(val_acc):
            ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        return fig
    
    def generate_classification_report(self,
                                     y_true: List[int],
                                     y_pred: List[int],
                                     save_path: Optional[str] = None) -> str:
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the report
            
        Returns:
            Classification report as string
        """
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            digits=4
        )
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Classification report saved to {save_path}")
        
        return report
    
    def evaluate_model(self,
                      model: nn.Module,
                      data_loader: torch.utils.data.DataLoader,
                      device: torch.device,
                      return_predictions: bool = False) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: PyTorch model to evaluate
            data_loader: Data loader for evaluation
            device: Device to run evaluation on
            return_predictions: Whether to return predictions and probabilities
            
        Returns:
            Dictionary containing evaluation results
        """
        model.eval()
        
        all_labels = []
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for videos, labels in tqdm(data_loader, desc="Evaluating"):
                videos = videos.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(videos)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_labels, all_predictions, all_probabilities)
        
        # Generate classification report
        classification_rep = self.generate_classification_report(all_labels, all_predictions)
        
        results = {
            'metrics': metrics,
            'classification_report': classification_rep,
            'confusion_matrix': confusion_matrix(all_labels, all_predictions).tolist()
        }
        
        if return_predictions:
            results['predictions'] = {
                'labels': all_labels,
                'predictions': all_predictions,
                'probabilities': all_probabilities
            }
        
        return results
    
    def compare_models(self,
                      results_dict: Dict[str, Dict[str, Any]],
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare multiple model results.
        
        Args:
            results_dict: Dictionary with model names as keys and results as values
            save_path: Path to save the comparison plot
            
        Returns:
            Matplotlib figure
        """
        models = list(results_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        # Extract metric values
        metric_values = {metric: [] for metric in metrics}
        for model_name in models:
            model_metrics = results_dict[model_name]['metrics']
            for metric in metrics:
                metric_values[metric].append(model_metrics.get(metric, 0))
        
        # Create comparison plot
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i * width, metric_values[metric], width, 
                  label=metric.capitalize())
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, metric in enumerate(metrics):
            for j, value in enumerate(metric_values[metric]):
                ax.text(j + i * width, value + 0.01, f'{value:.3f}',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        return fig
    
    def save_evaluation_report(self,
                             results: Dict[str, Any],
                             save_dir: str,
                             model_name: str = "model") -> None:
        """
        Save comprehensive evaluation report.
        
        Args:
            results: Evaluation results dictionary
            save_dir: Directory to save the report
            model_name: Name of the model for file naming
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metrics as JSON
        metrics_path = os.path.join(save_dir, f"{model_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        
        # Save classification report
        report_path = os.path.join(save_dir, f"{model_name}_classification_report.txt")
        with open(report_path, 'w') as f:
            f.write(results['classification_report'])
        
        # Save confusion matrix plot
        if 'predictions' in results:
            pred_data = results['predictions']
            cm_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
            self.plot_confusion_matrix(
                pred_data['labels'],
                pred_data['predictions'],
                save_path=cm_path
            )
            
            # Save ROC curve for binary classification
            if len(np.unique(pred_data['labels'])) == 2:
                roc_path = os.path.join(save_dir, f"{model_name}_roc_curve.png")
                self.plot_roc_curve(
                    pred_data['labels'],
                    pred_data['probabilities'],
                    save_path=roc_path
                )
                
                pr_path = os.path.join(save_dir, f"{model_name}_pr_curve.png")
                self.plot_precision_recall_curve(
                    pred_data['labels'],
                    pred_data['probabilities'],
                    save_path=pr_path
                )
        
        logger.info(f"Evaluation report saved to {save_dir}")

def cross_validate_model(model_class: type,
                        model_config: Dict[str, Any],
                        data_loader: torch.utils.data.DataLoader,
                        device: torch.device,
                        k_folds: int = 5,
                        epochs: int = 10) -> Dict[str, List[float]]:
    """
    Perform k-fold cross-validation.
    
    Args:
        model_class: Model class to instantiate
        model_config: Model configuration
        data_loader: Data loader
        device: Device for training
        k_folds: Number of folds
        epochs: Number of training epochs per fold
        
    Returns:
        Dictionary with cross-validation results
    """
    from sklearn.model_selection import KFold
    import torch.optim as optim
    
    evaluator = ModelEvaluator()
    fold_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    # Get all data
    all_data = []
    all_labels = []
    for videos, labels in data_loader:
        all_data.append(videos)
        all_labels.extend(labels.numpy())
    
    all_data = torch.cat(all_data, dim=0)
    all_labels = np.array(all_labels)
    
    # K-fold split
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_data)):
        logger.info(f"Training fold {fold + 1}/{k_folds}")
        
        # Create model
        model = model_class(**model_config).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Split data
        train_data = all_data[train_idx]
        train_labels = torch.tensor(all_labels[train_idx])
        val_data = all_data[val_idx]
        val_labels = all_labels[val_idx]
        
        # Train model
        model.train()
        for epoch in range(epochs):
            # Simple training loop
            optimizer.zero_grad()
            outputs = model(train_data.to(device))
            loss = criterion(outputs, train_labels.to(device))
            loss.backward()
            optimizer.step()
        
        # Evaluate model
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_data.to(device))
            _, val_predictions = torch.max(val_outputs, 1)
            val_predictions = val_predictions.cpu().numpy()
        
        # Calculate metrics
        fold_metrics = evaluator.calculate_metrics(val_labels, val_predictions)
        
        for metric in fold_results:
            fold_results[metric].append(fold_metrics[metric])
    
    # Calculate mean and std
    cv_results = {}
    for metric, values in fold_results.items():
        cv_results[f'{metric}_mean'] = np.mean(values)
        cv_results[f'{metric}_std'] = np.std(values)
    
    return cv_results

if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator()
    
    # Dummy data for testing
    y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
    y_pred = [0, 1, 0, 0, 0, 1, 1, 1, 1, 0]
    y_proba = [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.6, 0.4], [0.7, 0.3],
               [0.2, 0.8], [0.1, 0.9], [0.4, 0.6], [0.3, 0.7], [0.8, 0.2]]
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_true, y_pred, y_proba)
    print("Metrics:", metrics)
    
    # Generate plots
    evaluator.plot_confusion_matrix(y_true, y_pred)
    evaluator.plot_roc_curve(y_true, y_proba)
    
    plt.show()