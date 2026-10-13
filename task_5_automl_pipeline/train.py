import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
import argparse
import time
import pandas as pd
import numpy as np
from dataset import load_dataset_by_name, load_custom_csv
from pipeline import AutoMLPipeline

def print_banner():
    banner = '\n================================================================================\n   GROWFINIX TECHNOLOGY — MACHINE LEARNING INTERNSHIP (MONTH 2)\n   Task 5: Automating Machine Learning (AutoML Pipeline)\n================================================================================\n    '
    print(banner)

def main():
    parser = argparse.ArgumentParser(description='Growfinix AutoML Pipeline CLI Training Script')
    parser.add_argument('--dataset', type=str, default='customer_churn', help="Benchmark dataset ('customer_churn', 'credit_risk', 'employee_attrition') or path to custom CSV")
    parser.add_argument('--metric', type=str, default='F1', choices=['F1', 'ROC-AUC', 'Accuracy', 'Precision', 'Recall'], help='Primary ranking metric for champion selection (default: F1)')
    parser.add_argument('--cv', type=int, default=5, help='Stratified K-Fold cross validation splits (default: 5)')
    parser.add_argument('--samples', type=int, default=1200, help='Sample count for synthetic benchmark generation (default: 1200)')
    parser.add_argument('--output', type=str, default=os.path.join(CURRENT_DIR, 'models', 'champion_model.pkl'), help='Path where serialized champion model will be saved')
    parser.add_argument('--no-tuning', action='store_true', help='Skip hyperparameter refinement on champion')
    args = parser.parse_args()
    print_banner()
    print(f"[*] Step 1/4: Ingesting dataset '{args.dataset}'...")
    if os.path.isfile(args.dataset):
        (X, y, meta) = load_custom_csv(args.dataset)
    else:
        (X, y, meta) = load_dataset_by_name(args.dataset, n_samples=args.samples)
    pos_pct = y.sum() / len(y) * 100
    print(f"    - Title:        {meta['title']}")
    print(f'    - Total Rows:   {X.shape[0]}')
    print(f'    - Total Feats:  {X.shape[1]}')
    print(f"    - Target Name:  {meta['target_col']}")
    print(f"    - Class Balance: {meta['positive_label']}: {pos_pct:.1f}% | {meta['negative_label']}: {100 - pos_pct:.1f}%")
    print(f'\n[*] Step 2/4: Initializing AutoML Engine ({args.cv}-Fold CV, Primary Metric: {args.metric})...')
    automl = AutoMLPipeline(random_state=42)
    t_start = time.time()
    leaderboard = automl.fit(X, y, primary_metric=args.metric, cv_folds=args.cv, tune_champion=not args.no_tuning)
    total_time = time.time() - t_start
    print(f'\n[*] Step 3/4: Benchmarking Complete in {total_time:.2f}s!')
    print('\n' + '=' * 80)
    print('                     MODEL LEADERBOARD BENCHMARK')
    print('=' * 80)
    print(leaderboard.to_string())
    print('=' * 80)
    champion_name = automl.champion_name
    print(f'\n🏆 CHAMPION MODEL SELECTED: [{champion_name}]')
    print(f'    - Best CV {args.metric}: {leaderboard.iloc[0][args.metric]:.4f}')
    if automl.champion_params:
        print(f'    - Tuned Parameters:')
        for (k, v) in list(automl.champion_params.items())[:4]:
            print(f'      • {k}: {v}')
    print('\n[*] Top 5 Predictive Features (Relative Weight %):')
    df_imp = automl.get_feature_importances()
    if not df_imp.empty:
        for (idx, r) in df_imp.head(5).iterrows():
            print(f"    {idx + 1}. {r['Feature']:<30} | {r.get('Relative_Weight_%', 0):>6.2f}%")
    save_path = automl.save(args.output)
    print(f'\n[*] Step 4/4: Champion Pipeline successfully serialized to:')
    print(f'    --> {save_path}')
    print('\n[✓] Task 5 Training Pipeline Completed Successfully!')
if __name__ == '__main__':
    main()
