import os
import sys
import numpy as np
import torch
from sklearn.metrics import classification_report
try:
    from .visualize import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from visualize import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve


def evaluate_model(model, val_loader, class_names, config):
    """在验证集上进行全面评估"""
    device = config["device"]
    output_dir = config["output_dir"]
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    accuracy = np.mean(all_preds == all_labels)
    print(f"\n{'='*60}\n分类评估报告\n{'='*60}")
    print(report)
    print(f"总体准确率: {accuracy:.4f}")

    plot_confusion_matrix(all_labels, all_preds, class_names, output_dir)

    unhealthy_idx = class_names.index('Unhealthy') if 'Unhealthy' in class_names else 1
    y_bin = (all_labels == unhealthy_idx).astype(int)
    roc_auc = plot_roc_curve(y_bin, all_probs[:, unhealthy_idx], output_dir)
    ap = plot_precision_recall_curve(y_bin, all_probs[:, unhealthy_idx], output_dir)
    print(f"AUC-ROC: {roc_auc:.4f} | AP: {ap:.4f}")

    result_path = os.path.join(output_dir, "evaluation_results.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"总体准确率: {accuracy:.4f}\nAUC-ROC: {roc_auc:.4f}\nAP: {ap:.4f}\n\n")
        f.write(report)
