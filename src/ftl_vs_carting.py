"""
ftl_vs_carting.py — FTL vs Carting route-type decision framework.

Analyzes when Full Truck Load (FTL) outperforms Carting and vice versa,
accounting for:
- Distance (FTL typically better for long-haul)
- Time of day (congestion patterns)
- Source facility's graph position (centrality, reliability)
- Historical corridor delay patterns

Outputs:
- Corridor-level recommendations
- ML classifier for route-type selection
- Time-cost trade-off quantification
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from pathlib import Path
from typing import Tuple, Dict


OUTPUT_DIR = Path('outputs')


def analyze_route_type_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each corridor, compare FTL vs Carting performance.
    
    Computes:
    - Median delay ratio for each route type
    - Time savings of switching
    - Volume split
    """
    print("Analyzing FTL vs Carting performance by corridor...")
    
    # Only corridors with BOTH route types
    corridor_rt = df.groupby(
        ['source_center', 'destination_center', 'route_type']
    ).agg(
        trip_count=('trip_uuid', 'count'),
        median_delay=('delay_ratio', 'median'),
        mean_delay=('delay_ratio', 'mean'),
        pct_delayed=('is_delayed', 'mean'),
        median_actual_time=('segment_actual_time', 'median'),
        median_osrm_time=('segment_osrm_time', 'median'),
        median_distance=('segment_osrm_distance', 'median'),
    ).reset_index()
    
    # Pivot by route type
    ftl = corridor_rt[corridor_rt['route_type'] == 'FTL'].copy()
    carting = corridor_rt[corridor_rt['route_type'] == 'Carting'].copy()
    
    ftl = ftl.rename(columns={
        'trip_count': 'ftl_trips', 'median_delay': 'ftl_delay',
        'mean_delay': 'ftl_mean_delay', 'pct_delayed': 'ftl_pct_delayed',
        'median_actual_time': 'ftl_actual_time', 'median_osrm_time': 'ftl_osrm_time',
        'median_distance': 'ftl_distance',
    })
    carting = carting.rename(columns={
        'trip_count': 'carting_trips', 'median_delay': 'carting_delay',
        'mean_delay': 'carting_mean_delay', 'pct_delayed': 'carting_pct_delayed',
        'median_actual_time': 'carting_actual_time', 'median_osrm_time': 'carting_osrm_time',
        'median_distance': 'carting_distance',
    })
    
    # Join corridors that have both types
    comparison = ftl[['source_center', 'destination_center', 'ftl_trips', 'ftl_delay',
                       'ftl_mean_delay', 'ftl_pct_delayed', 'ftl_actual_time', 
                       'ftl_osrm_time', 'ftl_distance']].merge(
        carting[['source_center', 'destination_center', 'carting_trips', 'carting_delay',
                 'carting_mean_delay', 'carting_pct_delayed', 'carting_actual_time',
                 'carting_osrm_time', 'carting_distance']],
        on=['source_center', 'destination_center'],
        how='inner'
    )
    
    # Compute comparison metrics
    comparison['delay_diff'] = comparison['carting_delay'] - comparison['ftl_delay']
    comparison['time_saved_by_ftl'] = comparison['carting_actual_time'] - comparison['ftl_actual_time']
    comparison['better_route'] = np.where(
        comparison['ftl_delay'] < comparison['carting_delay'], 'FTL', 'Carting'
    )
    comparison['delay_improvement'] = np.where(
        comparison['better_route'] == 'FTL',
        (comparison['carting_delay'] - comparison['ftl_delay']) / comparison['carting_delay'] * 100,
        (comparison['ftl_delay'] - comparison['carting_delay']) / comparison['ftl_delay'] * 100,
    )
    comparison['total_trips'] = comparison['ftl_trips'] + comparison['carting_trips']
    comparison['avg_distance'] = (comparison['ftl_distance'] + comparison['carting_distance']) / 2
    
    print(f"  Corridors with both FTL and Carting: {len(comparison)}")
    print(f"  FTL is better: {(comparison['better_route'] == 'FTL').sum()}")
    print(f"  Carting is better: {(comparison['better_route'] == 'Carting').sum()}")
    
    return comparison


def build_route_classifier(
    df: pd.DataFrame,
    G: nx.DiGraph,
    node_metrics: pd.DataFrame,
) -> Tuple[GradientBoostingClassifier, pd.DataFrame]:
    """
    Build an ML classifier that predicts the optimal route type.
    
    Target: which route type achieves lower delay ratio on a trip
    Features: distance, time of day, source/dest centrality, corridor history
    """
    print("\nBuilding route-type classifier...")
    
    # Only use corridors where we can compare
    both_types = df.groupby(['source_center', 'destination_center'])['route_type'].nunique()
    dual_corridors = both_types[both_types == 2].index
    
    # For corridors with both types, determine which is better
    corridor_perf = df.groupby(
        ['source_center', 'destination_center', 'route_type']
    )['delay_ratio'].median().reset_index()
    
    corridor_perf_pivot = corridor_perf.pivot_table(
        index=['source_center', 'destination_center'],
        columns='route_type',
        values='delay_ratio'
    )
    
    # Filter to corridors with both types
    corridor_perf_pivot = corridor_perf_pivot.dropna()
    
    if len(corridor_perf_pivot) == 0:
        print("  Not enough dual-type corridors for classification")
        return None, pd.DataFrame()
    
    corridor_perf_pivot['better'] = np.where(
        corridor_perf_pivot.get('FTL', 999) < corridor_perf_pivot.get('Carting', 999),
        'FTL', 'Carting'
    )
    
    better_map = {}
    for (src, dst), row in corridor_perf_pivot.iterrows():
        better_map[(src, dst)] = row['better']
    
    # Prepare training data from individual trips on these corridors
    mask = df.set_index(['source_center', 'destination_center']).index.isin(corridor_perf_pivot.index)
    subset = df[mask].copy()
    
    # Target: optimal route type for this corridor
    subset['optimal_route'] = subset.apply(
        lambda r: better_map.get((r['source_center'], r['destination_center']), 'Carting'),
        axis=1
    )
    subset['target'] = (subset['optimal_route'] == 'FTL').astype(int)
    
    # Features
    node_lookup = node_metrics.set_index('facility_code')
    
    feat_df = pd.DataFrame({
        'segment_osrm_distance': subset['segment_osrm_distance'],
        'segment_osrm_time': subset['segment_osrm_time'],
        'hour_of_day': subset['hour_of_day'],
        'day_of_week': subset['day_of_week'],
        'is_weekend': subset['is_weekend'],
    })
    
    # Add source/dest graph features
    for prefix, center_col in [('src', 'source_center'), ('dst', 'destination_center')]:
        for metric in ['betweenness_centrality', 'pagerank', 'in_degree', 'out_degree',
                       'pct_delayed_out', 'total_volume']:
            if metric in node_lookup.columns:
                feat_df[f'{prefix}_{metric}'] = subset[center_col].map(
                    node_lookup[metric].to_dict()
                ).fillna(0)
    
    feat_df = feat_df.fillna(0)
    target = subset['target']
    
    # Train/test split using the data column
    train_mask = subset['data'] == 'training'
    test_mask = subset['data'] == 'test'
    
    X_train, y_train = feat_df[train_mask], target[train_mask]
    X_test, y_test = feat_df[test_mask], target[test_mask]
    
    if len(X_train) < 50 or len(X_test) < 10:
        print("  Not enough samples for classification")
        return None, pd.DataFrame()
    
    # Train classifier
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Route-type classifier accuracy: {acc:.3f}")
    print(f"  Classification report:")
    print(classification_report(y_test, y_pred, target_names=['Carting', 'FTL']))
    
    # Feature importance
    importances = pd.DataFrame({
        'feature': feat_df.columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    print(f"\n  Top features for route-type decision:")
    print(importances.head(10).to_string(index=False))
    
    return clf, importances


def generate_decision_matrix(comparison: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a decision matrix for FTL vs Carting recommendations.
    
    Segments corridors by:
    - Distance category (short: <50km, medium: 50-150km, long: >150km)
    - Time of day
    - Recommendation with quantified trade-off
    """
    print("\nGenerating decision matrix...")
    
    # Distance categories
    comparison['distance_category'] = pd.cut(
        comparison['avg_distance'],
        bins=[0, 50, 150, float('inf')],
        labels=['Short (<50km)', 'Medium (50-150km)', 'Long (>150km)']
    )
    
    # Aggregate by distance category
    decision = comparison.groupby('distance_category').agg(
        n_corridors=('source_center', 'count'),
        ftl_better_pct=('better_route', lambda x: (x == 'FTL').mean() * 100),
        avg_delay_diff=('delay_diff', 'mean'),
        avg_time_saved=('time_saved_by_ftl', 'mean'),
        avg_delay_improvement=('delay_improvement', 'mean'),
    ).reset_index()
    
    print("\n── Decision Matrix by Distance ──")
    print(decision.to_string(index=False))
    
    # Time-of-day analysis
    time_analysis = df.groupby(['route_type', 'time_bin']).agg(
        median_delay=('delay_ratio', 'median'),
        pct_delayed=('is_delayed', 'mean'),
        trip_count=('trip_uuid', 'count'),
    ).reset_index()
    
    print("\n── Route Type Performance by Time of Day ──")
    time_pivot = time_analysis.pivot_table(
        index='time_bin', columns='route_type', values='median_delay'
    )
    print(time_pivot.to_string())
    
    return decision


def plot_ftl_vs_carting(comparison: pd.DataFrame, df: pd.DataFrame):
    """Generate FTL vs Carting visualizations."""
    print("\nGenerating FTL vs Carting plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Delay comparison scatter
    ax = axes[0, 0]
    ax.scatter(comparison['ftl_delay'], comparison['carting_delay'],
               s=comparison['total_trips'].clip(upper=200) * 2,
               alpha=0.5, c=comparison['avg_distance'],
               cmap='viridis', edgecolors='grey', linewidths=0.3)
    
    max_val = max(comparison['ftl_delay'].quantile(0.95), comparison['carting_delay'].quantile(0.95))
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.7, label='Equal performance')
    ax.set_xlabel('FTL Median Delay Ratio')
    ax.set_ylabel('Carting Median Delay Ratio')
    ax.set_title('FTL vs Carting: Delay Ratio Comparison\n'
                 'Points below line = FTL is better', fontweight='bold')
    ax.legend()
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    
    # 2. Distribution of delay difference
    ax = axes[0, 1]
    ax.hist(comparison['delay_diff'].clip(-3, 3), bins=40, color='#1a73e8', alpha=0.7,
            edgecolor='white')
    ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Equal')
    ax.set_xlabel('Carting Delay − FTL Delay')
    ax.set_ylabel('Number of Corridors')
    ax.set_title('Route Type Delay Difference Distribution\n'
                 'Positive = FTL is better', fontweight='bold')
    ax.legend()
    
    # 3. By distance category
    ax = axes[1, 0]
    if 'distance_category' not in comparison.columns:
        comparison['distance_category'] = pd.cut(
            comparison['avg_distance'],
            bins=[0, 50, 150, float('inf')],
            labels=['Short', 'Medium', 'Long']
        )
    
    dist_summary = comparison.groupby('distance_category').agg(
        ftl_better=('better_route', lambda x: (x == 'FTL').mean() * 100),
        n=('source_center', 'count')
    ).reset_index()
    
    bars = ax.bar(dist_summary['distance_category'].astype(str), dist_summary['ftl_better'],
                  color=['#43a047', '#fb8c00', '#e53935'], alpha=0.8)
    ax.axhline(50, color='grey', linestyle='--', alpha=0.5)
    ax.set_ylabel('% Corridors Where FTL is Better')
    ax.set_xlabel('Distance Category')
    ax.set_title('FTL Advantage by Distance\n'
                 'Above 50% = FTL generally better', fontweight='bold')
    for bar, n in zip(bars, dist_summary['n']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'n={n}', ha='center', fontsize=10)
    
    # 4. Time-of-day comparison
    ax = axes[1, 1]
    time_perf = df.groupby(['route_type', 'time_bin'])['delay_ratio'].median().reset_index()
    time_pivot = time_perf.pivot(index='time_bin', columns='route_type', values='delay_ratio')
    
    col_order = [c for c in ['night', 'morning', 'afternoon', 'evening'] if c in time_pivot.index]
    if col_order:
        time_pivot = time_pivot.loc[col_order]
    
    time_pivot.plot(kind='bar', ax=ax, color=['#e53935', '#1a73e8'], alpha=0.8)
    ax.axhline(1.2, color='red', linestyle=':', alpha=0.5, label='SLA Threshold')
    ax.set_ylabel('Median Delay Ratio')
    ax.set_xlabel('Time of Day')
    ax.set_title('FTL vs Carting by Time of Day', fontweight='bold')
    ax.legend()
    ax.tick_params(axis='x', rotation=0)
    
    fig.suptitle('FTL vs Carting Decision Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'graphs' / 'ftl_vs_carting.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: outputs/graphs/ftl_vs_carting.png")


def run_ftl_analysis(
    df: pd.DataFrame,
    G: nx.DiGraph,
    node_metrics: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full FTL vs Carting analysis."""
    print("=" * 60)
    print("FTL vs CARTING DECISION FRAMEWORK")
    print("=" * 60)
    
    comparison = analyze_route_type_performance(df)
    clf, importances = build_route_classifier(df, G, node_metrics)
    decision = generate_decision_matrix(comparison, df)
    plot_ftl_vs_carting(comparison, df)
    
    # Save results
    comparison.to_csv(OUTPUT_DIR / 'ftl_vs_carting_comparison.csv', index=False)
    decision.to_csv(OUTPUT_DIR / 'route_type_decision_matrix.csv', index=False)
    
    return comparison, decision


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_pipeline import run_pipeline
    from graph_builder import build_graph
    from bottleneck_analysis import run_bottleneck_analysis
    
    df = run_pipeline()
    G = build_graph(df)
    node_metrics, corridor_df = run_bottleneck_analysis(G)
    comparison, decision = run_ftl_analysis(df, G, node_metrics)
