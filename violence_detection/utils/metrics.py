"""
Custom metrics for violence detection evaluation.

This module provides specialized metrics and evaluation functions
for the violence detection task.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef
)
from typing import List, Dict, Any, Tuple, Optional
import warnings

class ViolenceDetectionMetrics:
    """Specialized metrics for violence detection evaluation."""
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize metrics calculator.
        
        Args:
            class_names: Names of the classes
        """
        self.class_names = class_names or ["Non-Violence", "Violence"]
        self.violence_class_idx = 1  # Assuming violence is class 1
    
    def calculate_all_metrics(self, 
                            y_true: List[int], 
                            y_pred: List[int],
                            y_proba: Optional[List[List[float]]] = None) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for violence detection.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics.update(self._calculate_basic_metrics(y_true, y_pred))
        
        # Violence-specific metrics
        metrics.update(self._calculate_violence_metrics(y_true, y_pred))
        
        # Probability-based metrics
        if y_proba is not None:
            metrics.update(self._calculate_probability_metrics(y_true, y_proba))
        
        # Additional metrics
        metrics.update(self._calculate_additional_metrics(y_true, y_pred))
        
        return metrics
    
    def _calculate_basic_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """Calculate basic classification metrics."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
                'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
                'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
                'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
                'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            }
    
    def _calculate_violence_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """Calculate violence-specific metrics."""
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics = {}
        
        # Non-violence metrics
        if len(precision_per_class) > 0:
            metrics['precision_non_violence'] = precision_per_class[0]
            metrics['recall_non_violence'] = recall_per_class[0]
            metrics['f1_non_violence'] = f1_per_class[0]
        
        # Violence metrics (most important for this task)
        if len(precision_per_class) > 1:
            metrics['precision_violence'] = precision_per_class[1]
            metrics['recall_violence'] = recall_per_class[1]
            metrics['f1_violence'] = f1_per_class[1]
        
        # Violence detection specific metrics
        tn, fp, fn, tp = self._calculate_confusion_matrix_values(y_true, y_pred)
        
        # Sensitivity and Specificity
        if tp + fn > 0:
            metrics['sensitivity'] = tp / (tp + fn)  # Same as recall for violence class
        else:
            metrics['sensitivity'] = 0.0
            
        if tn + fp > 0:
            metrics['specificity'] = tn / (tn + fp)
        else:
            metrics['specificity'] = 0.0
        
        # False positive rate and False negative rate
        if fp + tn > 0:
            metrics['false_positive_rate'] = fp / (fp + tn)
        else:
            metrics['false_positive_rate'] = 0.0
            
        if fn + tp > 0:
            metrics['false_negative_rate'] = fn / (fn + tp)
        else:
            metrics['false_negative_rate'] = 0.0
        
        # Positive and Negative Predictive Values
        if tp + fp > 0:
            metrics['positive_predictive_value'] = tp / (tp + fp)  # Same as precision
        else:
            metrics['positive_predictive_value'] = 0.0
            
        if tn + fn > 0:
            metrics['negative_predictive_value'] = tn / (tn + fn)
        else:
            metrics['negative_predictive_value'] = 0.0
        
        return metrics
    
    def _calculate_probability_metrics(self, y_true: List[int], y_proba: List[List[float]]) -> Dict[str, float]:
        """Calculate probability-based metrics."""
        metrics = {}
        
        try:
            y_proba_array = np.array(y_proba)
            
            # For binary classification
            if y_proba_array.shape[1] == 2:
                violence_proba = y_proba_array[:, self.violence_class_idx]
                
                # AUC metrics
                metrics['auc_roc'] = roc_auc_score(y_true, violence_proba)
                metrics['auc_pr'] = average_precision_score(y_true, violence_proba)
                
                # Brier Score (lower is better)
                y_true_binary = np.array(y_true)
                brier_score = np.mean((violence_proba - y_true_binary) ** 2)
                metrics['brier_score'] = brier_score
                
                # Log Loss
                epsilon = 1e-15  # To avoid log(0)
                violence_proba_clipped = np.clip(violence_proba, epsilon, 1 - epsilon)
                log_loss = -np.mean(y_true_binary * np.log(violence_proba_clipped) + 
                                  (1 - y_true_binary) * np.log(1 - violence_proba_clipped))
                metrics['log_loss'] = log_loss
                
        except Exception as e:
            print(f"Warning: Could not calculate probability-based metrics: {e}")
        
        return metrics
    
    def _calculate_additional_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """Calculate additional specialized metrics."""
        metrics = {}
        
        # Matthews Correlation Coefficient
        try:
            metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        except:
            metrics['matthews_corrcoef'] = 0.0
        
        # Violence Detection Rate (how many violence instances were correctly detected)
        violence_true = np.array(y_true) == self.violence_class_idx
        violence_pred = np.array(y_pred) == self.violence_class_idx
        
        if np.sum(violence_true) > 0:
            violence_detection_rate = np.sum(violence_true & violence_pred) / np.sum(violence_true)
            metrics['violence_detection_rate'] = violence_detection_rate
        else:
            metrics['violence_detection_rate'] = 0.0
        
        # False Violence Alarm Rate
        non_violence_true = np.array(y_true) == 0
        if np.sum(non_violence_true) > 0:
            false_violence_rate = np.sum(non_violence_true & violence_pred) / np.sum(non_violence_true)
            metrics['false_violence_alarm_rate'] = false_violence_rate
        else:
            metrics['false_violence_alarm_rate'] = 0.0
        
        return metrics
    
    def _calculate_confusion_matrix_values(self, y_true: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
        """Calculate confusion matrix values (tn, fp, fn, tp)."""
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle cases where not all classes are present
            tn = fp = fn = tp = 0
            for i, true_label in enumerate(y_true):
                pred_label = y_pred[i]
                if true_label == 0 and pred_label == 0:
                    tn += 1
                elif true_label == 0 and pred_label == 1:
                    fp += 1
                elif true_label == 1 and pred_label == 0:
                    fn += 1
                elif true_label == 1 and pred_label == 1:
                    tp += 1
        
        return tn, fp, fn, tp
    
    def calculate_temporal_metrics(self, 
                                 temporal_predictions: List[List[int]],
                                 temporal_labels: List[List[int]]) -> Dict[str, float]:
        """
        Calculate metrics for temporal sequences.
        
        Args:
            temporal_predictions: List of prediction sequences
            temporal_labels: List of label sequences
            
        Returns:
            Dictionary containing temporal metrics
        """
        metrics = {}
        
        # Flatten sequences for overall metrics
        all_preds = [pred for seq in temporal_predictions for pred in seq]
        all_labels = [label for seq in temporal_labels for label in seq]
        
        # Overall sequence metrics
        sequence_metrics = self.calculate_all_metrics(all_labels, all_preds)
        metrics.update({f'temporal_{k}': v for k, v in sequence_metrics.items()})
        
        # Sequence-level accuracy
        sequence_accuracies = []
        for pred_seq, label_seq in zip(temporal_predictions, temporal_labels):
            if len(pred_seq) == len(label_seq):
                seq_acc = accuracy_score(label_seq, pred_seq)
                sequence_accuracies.append(seq_acc)
        
        if sequence_accuracies:
            metrics['sequence_accuracy_mean'] = np.mean(sequence_accuracies)
            metrics['sequence_accuracy_std'] = np.std(sequence_accuracies)
        
        # Violence episode detection
        violence_episodes_true = self._count_violence_episodes(temporal_labels)
        violence_episodes_pred = self._count_violence_episodes(temporal_predictions)
        
        metrics['violence_episodes_true'] = violence_episodes_true
        metrics['violence_episodes_predicted'] = violence_episodes_pred
        
        if violence_episodes_true > 0:
            episode_detection_rate = min(violence_episodes_pred / violence_episodes_true, 1.0)
            metrics['episode_detection_rate'] = episode_detection_rate
        
        return metrics
    
    def _count_violence_episodes(self, sequences: List[List[int]]) -> int:
        """Count violence episodes across sequences."""
        total_episodes = 0
        
        for sequence in sequences:
            in_episode = False
            for label in sequence:
                if label == self.violence_class_idx and not in_episode:
                    total_episodes += 1
                    in_episode = True
                elif label != self.violence_class_idx:
                    in_episode = False
        
        return total_episodes

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in violence detection.
    
    Reference: Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal loss for dense object detection.
    """
    
    def __init__(self, alpha: Optional[List[float]] = None, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (list of weights for each class)
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            self.alpha = torch.tensor(alpha)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Focal Loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Focal loss value
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            at = self.alpha.gather(0, targets)
            ce_loss = at * ce_loss
        
        # Apply focal term
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross Entropy Loss for imbalanced datasets."""
    
    def __init__(self, weights: List[float]):
        """
        Initialize weighted cross entropy loss.
        
        Args:
            weights: List of weights for each class
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = torch.tensor(weights)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.weights.device != targets.device:
            self.weights = self.weights.to(targets.device)
        
        return nn.functional.cross_entropy(inputs, targets, weight=self.weights)

def calculate_class_weights(labels: List[int], method: str = 'balanced') -> List[float]:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        labels: List of class labels
        method: Method to calculate weights ('balanced', 'inverse_freq')
        
    Returns:
        List of class weights
    """
    from collections import Counter
    
    class_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    if method == 'balanced':
        weights = []
        for class_id in sorted(class_counts.keys()):
            weight = total_samples / (num_classes * class_counts[class_id])
            weights.append(weight)
    
    elif method == 'inverse_freq':
        weights = []
        for class_id in sorted(class_counts.keys()):
            weight = 1.0 / class_counts[class_id]
            weights.append(weight)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return weights

def evaluate_model_robustness(predictions: List[int], 
                            confidence_scores: List[float],
                            threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluate model robustness based on prediction confidence.
    
    Args:
        predictions: Model predictions
        confidence_scores: Confidence scores for predictions
        threshold: Confidence threshold
        
    Returns:
        Dictionary containing robustness metrics
    """
    confident_predictions = np.array(confidence_scores) >= threshold
    
    metrics = {
        'confident_predictions_ratio': np.mean(confident_predictions),
        'mean_confidence': np.mean(confidence_scores),
        'std_confidence': np.std(confidence_scores),
        'min_confidence': np.min(confidence_scores),
        'max_confidence': np.max(confidence_scores)
    }
    
    # Confidence distribution by class
    predictions_array = np.array(predictions)
    for class_id in np.unique(predictions_array):
        class_mask = predictions_array == class_id
        class_confidences = np.array(confidence_scores)[class_mask]
        
        if len(class_confidences) > 0:
            metrics[f'mean_confidence_class_{class_id}'] = np.mean(class_confidences)
            metrics[f'std_confidence_class_{class_id}'] = np.std(class_confidences)
    
    return metrics

if __name__ == "__main__":
    # Example usage
    
    # Dummy data
    y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0] * 10
    y_pred = [0, 1, 0, 0, 0, 1, 1, 1, 1, 0] * 10
    y_proba = [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.6, 0.4], [0.7, 0.3],
               [0.2, 0.8], [0.1, 0.9], [0.4, 0.6], [0.3, 0.7], [0.8, 0.2]] * 10
    
    # Calculate metrics
    metrics_calculator = ViolenceDetectionMetrics()
    metrics = metrics_calculator.calculate_all_metrics(y_true, y_pred, y_proba)
    
    print("Violence Detection Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test Focal Loss
    inputs = torch.randn(10, 2)
    targets = torch.randint(0, 2, (10,))
    
    focal_loss = FocalLoss(alpha=[0.25, 0.75], gamma=2.0)
    loss = focal_loss(inputs, targets)
    print(f"\nFocal Loss: {loss.item():.4f}")
    
    # Test class weights calculation
    weights = calculate_class_weights(y_true)
    print(f"Class weights: {weights}")