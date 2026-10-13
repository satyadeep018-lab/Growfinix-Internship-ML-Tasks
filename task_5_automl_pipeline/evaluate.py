import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, log_loss, confusion_matrix, classification_report
from dataset import load_dataset_by_name, load_custom_csv
from pipeline import AutoMLPipeline

def print_banner():
    banner = '\n================================================================================\n   GROWFINIX TECHNOLOGY — MACHINE LEARNING INTERNSHIP (MONTH 2)\n   Task 5 Evaluation Suite: Champion Model Rigorous Validation\n================================================================================\n    '
    print(banner)

def main():
    parser = argparse.ArgumentParser(description='Evaluate Champion Model from AutoML Pipeline')
    parser.add_argument('--model', type=str, default=os.path.join(CURRENT_DIR, 'models', 'champion_model.pkl'), help='Path to serialized champion model')
    parser.add_argument('--dataset', type=str, default='customer_churn', help='Benchmark dataset name or custom CSV')
    parser.add_argument('--test-size', type=float, default=0.25, help='Holdout test split ratio (default: 0.25)')
    parser.add_argument('--save-plots', action='store_true', help='Generate and save evaluation plots into outputs/')
    args = parser.parse_args()
    print_banner()
    print(f"[*] Step 1/3: Loading dataset '{args.dataset}' with {int(args.test_size * 100)}% holdout test split...")
    if os.path.isfile(args.dataset):
        (X, y, meta) = load_custom_csv(args.dataset)
    else:
        (X, y, meta) = load_dataset_by_name(args.dataset, n_samples=1200)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)
    print(f'    - Train Samples: {X_train.shape[0]}')
    print(f'    - Holdout Test:  {X_test.shape[0]}')
    if os.path.isfile(args.model):
        print(f"\n[*] Step 2/3: Loading serialized champion model from '{args.model}'...")
        automl = AutoMLPipeline.load(args.model)
    else:
        print(f"\n[!] Model file '{args.model}' not found. Auto-training champion model on training set...")
        automl = AutoMLPipeline(random_state=42)
        automl.fit(X_train, y_train, primary_metric='F1', cv_folds=5)
    champion_name = automl.champion_name or 'Champion Estimator'
    print(f'    --> Evaluating Champion: [{champion_name}]')
    print(f'\n[*] Step 3/3: Evaluating holdout test split...')
    y_pred = automl.predict(X_test)
    y_proba = automl.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    loss = log_loss(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    print('\n' + '=' * 65)
    print('               HOLDOUT TEST SET METRICS')
    print('=' * 65)
    print(f'  Accuracy:             {acc * 100:.2f}%')
    print(f'  ROC-AUC Score:        {auc:.4f}')
    print(f'  F1-Score:             {f1:.4f}')
    print(f'  Precision:            {prec:.4f}')
    print(f'  Recall (Sensitivity): {rec:.4f}')
    print(f'  Binary Log-Loss:      {loss:.4f}')
    print('=' * 65)
    print('\nDetailed Scikit-Learn Classification Report:')
    target_names = [meta['negative_label'], meta['positive_label']]
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))
    print('\nConfusion Matrix:')
    print(f'  TN (True Negative):  {cm[0, 0]:>4}   |   FP (False Positive): {cm[0, 1]:>4}')
    print(f'  FN (False Negative): {cm[1, 0]:>4}   |   TP (True Positive):  {cm[1, 1]:>4}')
    if args.save_plots:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.metrics import roc_curve
            out_dir = os.path.join(CURRENT_DIR, 'outputs')
            os.makedirs(out_dir, exist_ok=True)
            (fig, ax) = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax, xticklabels=target_names, yticklabels=target_names)
            ax.set_title(f'Confusion Matrix - {champion_name}', fontsize=11, fontweight='bold')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            plt.tight_layout()
            cm_path = os.path.join(out_dir, 'confusion_matrix.png')
            fig.savefig(cm_path, dpi=150)
            plt.close(fig)
            print(f'\n[+] Saved Confusion Matrix to: {cm_path}')
            (fpr, tpr, _) = roc_curve(y_test, y_proba)
            (fig, ax) = plt.subplots(figsize=(5, 4))
            ax.plot(fpr, tpr, color='#2563eb', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - {champion_name}', fontsize=11, fontweight='bold')
            ax.legend(loc='lower right')
            plt.tight_layout()
            roc_path = os.path.join(out_dir, 'roc_curve.png')
            fig.savefig(roc_path, dpi=150)
            plt.close(fig)
            print(f'[+] Saved ROC Curve to: {roc_path}')
        except Exception as e:
            print(f'[!] Warning: Plot generation skipped: {e}')
    print('\n[✓] Evaluation completed successfully!')
if __name__ == '__main__':
    main()
