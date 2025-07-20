import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class MultiLabelMetrics:
    """Metrics for multi-label classification"""    
    @staticmethod
    def hamming_accuracy(y_true, y_pred):
        """Hamming accuracy (element-wise accuracy)"""
        return (y_true == y_pred).float().mean()



    @staticmethod
    def subset_accuracy(y_true, y_pred):
        """Exact match accuracy"""
        return (y_true == y_pred).all(dim=1).float().mean()



    @staticmethod
    def compute_metrics(y_true, y_pred, threshold=0.5):
        """Compute comprehensive metrics"""
        # binary predictions
        y_pred_binary = (y_pred > threshold).float()
        
        # to numpy for sklearn
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred_binary.cpu().numpy()



        metrics = {
            'hamming_accuracy': MultiLabelMetrics.hamming_accuracy(y_true, y_pred_binary).item(),
            'subset_accuracy': MultiLabelMetrics.subset_accuracy(y_true, y_pred_binary).item(),
            'micro_f1': f1_score(y_true_np, y_pred_np, average='micro'),
            'macro_f1': f1_score(y_true_np, y_pred_np, average='macro'),
            'weighted_f1': f1_score(y_true_np, y_pred_np, average='weighted'),
            'micro_precision': precision_score(y_true_np, y_pred_np, average='micro'),
            'macro_precision': precision_score(y_true_np, y_pred_np, average='macro'),
            'micro_recall': recall_score(y_true_np, y_pred_np, average='micro'),
            'macro_recall': recall_score(y_true_np, y_pred_np, average='macro'),
        }
        return metrics