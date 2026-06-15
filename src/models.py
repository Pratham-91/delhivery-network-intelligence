"""
models.py — ETA prediction models: baseline vs graph-enhanced.

Models:
1. Baseline XGBoost: trip-level features only
2. Graph-Enhanced XGBoost: baseline + graph centrality + corridor history
3. Graph+Node2Vec XGBoost: graph-enhanced + node2vec embeddings

Evaluation:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- 15%-accuracy: % of predictions within 15% of actual
- Statistical significance via bootstrap confidence intervals
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Any
import json
import pickle


OUTPUT_DIR = Path('outputs')


def accuracy_within_pct(y_true, y_pred, pct: float = 0.15) -> float:
    """Compute % of predictions within pct of actual value."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Avoid division by zero
    mask = y_true > 0
    if mask.sum() == 0:
        return 0.0
    relative_error = np.abs(y_pred[mask] - y_true[mask]) / y_true[mask]
    return float((relative_error < pct).mean() * 100)


def train_xgboost(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    model_name: str = "model",
) -> xgb.XGBRegressor:
    """Train an XGBoost regressor with tuned hyperparameters."""
    print(f"  Training {model_name} on {X_train.shape[0]:,} samples, "
          f"{X_train.shape[1]} features...")
    
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False,
    )
    
    return model


def evaluate_model(
    model: xgb.XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "model",
) -> Dict[str, float]:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    acc_15 = accuracy_within_pct(y_test, y_pred, 0.15)
    acc_20 = accuracy_within_pct(y_test, y_pred, 0.20)
    
    # Median absolute error
    med_ae = float(np.median(np.abs(y_test - y_pred)))
    
    metrics = {
        'model': model_name,
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'MedAE': round(med_ae, 4),
        'Accuracy_15pct': round(acc_15, 2),
        'Accuracy_20pct': round(acc_20, 2),
    }
    
    print(f"\n  {model_name} Results:")
    for k, v in metrics.items():
        if k != 'model':
            print(f"    {k}: {v}")
    
    return metrics


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot: int = 1000, ci: float = 0.95):
    """Compute bootstrap confidence interval for a metric."""
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        scores.append(metric_fn(y_true.iloc[idx], y_pred[idx]))
    
    alpha = (1 - ci) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)
    return lower, upper


def run_model_comparison(model_data: Dict) -> pd.DataFrame:
    """
    Train and compare all model variants.
    
    Returns a DataFrame with model comparison metrics.
    """
    print("=" * 60)
    print("ETA PREDICTION MODEL TRAINING & BENCHMARKING")
    print("=" * 60)
    
    results = []
    models = {}
    predictions = {}
    
    for model_name in ['baseline', 'graph_enhanced', 'graph_n2v']:
        X_train, y_train = model_data['train'][model_name]
        X_test, y_test = model_data['test'][model_name]
        
        if X_train.shape[1] == 0:
            print(f"  Skipping {model_name}: no features")
            continue
        
        # Train
        model = train_xgboost(X_train, y_train, model_name)
        models[model_name] = model
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test, model_name)
        results.append(metrics)
        
        # Store predictions for comparison
        predictions[model_name] = model.predict(X_test)
        
        # Save model
        model_path = OUTPUT_DIR / 'models' / f'{model_name}_xgb.json'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            model.get_booster().save_model(str(model_path))
        except Exception:
            # Fallback to pickle if XGBoost save fails
            import pickle
            pkl_path = OUTPUT_DIR / 'models' / f'{model_name}_xgb.pkl'
            with open(pkl_path, 'wb') as f:
                pickle.dump(model, f)
    
    results_df = pd.DataFrame(results)
    
    # ── Compute graph advantage ──
    if 'baseline' in models and 'graph_enhanced' in models:
        baseline_metrics = results_df[results_df['model'] == 'baseline'].iloc[0]
        graph_metrics = results_df[results_df['model'] == 'graph_enhanced'].iloc[0]
        
        print("\n" + "=" * 40)
        print("GRAPH ADVANTAGE ANALYSIS")
        print("=" * 40)
        
        mae_improvement = baseline_metrics['MAE'] - graph_metrics['MAE']
        mae_pct_improvement = (mae_improvement / baseline_metrics['MAE']) * 100
        acc_improvement = graph_metrics['Accuracy_15pct'] - baseline_metrics['Accuracy_15pct']
        
        print(f"  MAE improvement: {mae_improvement:.4f} ({mae_pct_improvement:.2f}% reduction)")
        print(f"  15%-accuracy improvement: +{acc_improvement:.2f} percentage points")
        
        # Bootstrap CI for the improvement
        X_test_b, y_test_b = model_data['test']['baseline']
        X_test_g, y_test_g = model_data['test']['graph_enhanced']
        
        pred_b = predictions['baseline']
        pred_g = predictions['graph_enhanced']
        
        def mae_diff(y_true, idx):
            return mean_absolute_error(y_true, pred_b[idx]) - mean_absolute_error(y_true, pred_g[idx])
        
        n = len(y_test_b)
        diffs = []
        for _ in range(500):
            idx = np.random.choice(n, n, replace=True)
            diff = mean_absolute_error(y_test_b.iloc[idx], pred_b[idx]) - \
                   mean_absolute_error(y_test_g.iloc[idx], pred_g[idx])
            diffs.append(diff)
        
        ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
        print(f"  MAE improvement 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
        
        if ci_low > 0:
            print("  ✓ Graph advantage is STATISTICALLY SIGNIFICANT")
        else:
            print("  ⚠ Graph advantage includes zero — marginal significance")
    
    # Save results
    results_df.to_csv(OUTPUT_DIR / 'model_comparison.csv', index=False)
    
    # ── Generate comparison plots ──
    _plot_model_comparison(results_df, predictions, model_data)
    _plot_feature_importance(models, model_data)
    
    return results_df, models, predictions


def _plot_model_comparison(results_df, predictions, model_data):
    """Generate model comparison visualizations."""
    print("\nGenerating model comparison plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = ['#1a73e8', '#e53935', '#43a047']
    model_names = results_df['model'].tolist()
    
    # MAE comparison
    mae_vals = results_df['MAE'].tolist()
    bars = axes[0].bar(model_names, mae_vals, color=colors[:len(model_names)], alpha=0.8)
    axes[0].set_ylabel('MAE (minutes)')
    axes[0].set_title('Mean Absolute Error ↓', fontweight='bold')
    for bar, val in zip(bars, mae_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f'{val:.2f}', ha='center', fontweight='bold')
    
    # 15%-accuracy comparison
    acc_vals = results_df['Accuracy_15pct'].tolist()
    bars = axes[1].bar(model_names, acc_vals, color=colors[:len(model_names)], alpha=0.8)
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('% Predictions Within 15% of Actual ↑', fontweight='bold')
    for bar, val in zip(bars, acc_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}%', ha='center', fontweight='bold')
    
    # RMSE comparison
    rmse_vals = results_df['RMSE'].tolist()
    bars = axes[2].bar(model_names, rmse_vals, color=colors[:len(model_names)], alpha=0.8)
    axes[2].set_ylabel('RMSE (minutes)')
    axes[2].set_title('Root Mean Squared Error ↓', fontweight='bold')
    for bar, val in zip(bars, rmse_vals):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f'{val:.2f}', ha='center', fontweight='bold')
    
    fig.suptitle('ETA Prediction Model Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    fig.savefig(OUTPUT_DIR / 'graphs' / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: outputs/graphs/model_comparison.png")
    
    # ── Actual vs Predicted scatter ──
    _, y_test = model_data['test']['baseline']
    
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
    if n_models == 1:
        axes = [axes]
    
    for ax, (name, y_pred) in zip(axes, predictions.items()):
        # Sample for readability
        n = min(5000, len(y_test))
        idx = np.random.choice(len(y_test), n, replace=False)
        
        ax.scatter(y_test.iloc[idx], y_pred[idx], alpha=0.2, s=10, color='#1a73e8')
        
        # Perfect prediction line
        max_val = max(y_test.iloc[idx].max(), y_pred[idx].max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.7, label='Perfect')
        
        # ±15% bands
        x_range = np.linspace(0, max_val, 100)
        ax.fill_between(x_range, x_range * 0.85, x_range * 1.15,
                        alpha=0.1, color='green', label='±15%')
        
        ax.set_xlabel('Actual Time (min)')
        ax.set_ylabel('Predicted Time (min)')
        ax.set_title(f'{name}\nMAE={mean_absolute_error(y_test, y_pred):.2f}',
                     fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_xlim(0, np.percentile(y_test, 99))
        ax.set_ylim(0, np.percentile(y_pred, 99))
    
    fig.suptitle('Actual vs Predicted ETA', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'graphs' / 'actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: outputs/graphs/actual_vs_predicted.png")


def _plot_feature_importance(models, model_data):
    """Plot feature importance for each model."""
    print("Generating feature importance plots...")
    
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 10))
    if n_models == 1:
        axes = [axes]
    
    for ax, (name, model) in zip(axes, models.items()):
        importance = model.feature_importances_
        feature_names = model_data['feature_names'][name]
        
        # Top 25 features
        idx = np.argsort(importance)[-25:]
        
        ax.barh(
            [feature_names[i] for i in idx],
            importance[idx],
            color='#1a73e8', alpha=0.8
        )
        ax.set_xlabel('Feature Importance (Gain)')
        ax.set_title(f'{name}', fontweight='bold')
    
    fig.suptitle('Top 25 Features by Importance', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'graphs' / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: outputs/graphs/feature_importance.png")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_pipeline import run_pipeline
    from graph_builder import build_graph
    from bottleneck_analysis import run_bottleneck_analysis
    from feature_engineering import prepare_model_data
    
    df = run_pipeline()
    G = build_graph(df)
    node_metrics, corridor_df = run_bottleneck_analysis(G)
    model_data = prepare_model_data(df, G, node_metrics, corridor_df, include_node2vec=True)
    results_df, models, predictions = run_model_comparison(model_data)
    
    print("\n── Final Model Comparison ──")
    print(results_df.to_string(index=False))
