"""
main.py -- Master orchestration script for the Delhivery Graph-Based Network Intelligence project.

Runs the full pipeline end-to-end:
1. Data pipeline (load, clean, feature extraction)
2. Graph construction
3. Bottleneck & corridor audit
4. Visualizations
5. ETA prediction models (baseline vs graph-enhanced)
6. FTL vs Carting decision framework
7. Exports all results and generates strategy memo data
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_pipeline import run_pipeline
from graph_builder import build_graph, save_graph
from bottleneck_analysis import run_bottleneck_analysis
from visualizations import generate_all_visualizations
from feature_engineering import prepare_model_data
from models import run_model_comparison
from ftl_vs_carting import run_ftl_analysis


def main():
    start_time = time.time()
    
    print("=" * 60)
    print("  DELHIVERY GRAPH-BASED NETWORK INTELLIGENCE")
    print("  Full Pipeline Execution")
    print("=" * 60)
    print()
    
    # -- Phase 1: Data Pipeline --
    print("\n[1/6] DATA PIPELINE")
    df = run_pipeline()
    
    # -- Phase 2: Graph Construction --
    print("\n[2/6] GRAPH CONSTRUCTION")
    G = build_graph(df)
    save_graph(G)
    
    # -- Phase 3: Bottleneck Analysis --
    print("\n[3/6] BOTTLENECK & CORRIDOR AUDIT")
    node_metrics, corridor_df = run_bottleneck_analysis(G)
    
    # -- Phase 4: Visualizations --
    print("\n[4/6] VISUALIZATIONS")
    generate_all_visualizations(G, df, node_metrics, corridor_df)
    
    # -- Phase 5: ETA Prediction Models --
    print("\n[5/6] ETA PREDICTION MODELS")
    model_data = prepare_model_data(
        df, G, node_metrics, corridor_df, 
        include_node2vec=True, n2v_dimensions=32
    )
    results_df, models, predictions = run_model_comparison(model_data)
    
    # -- Phase 6: FTL vs Carting --
    print("\n[6/6] FTL vs CARTING FRAMEWORK")
    comparison, decision = run_ftl_analysis(df, G, node_metrics)
    
    # -- Summary --
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"\n  Outputs generated:")
    print(f"    - outputs/logistics_graph.graphml")
    print(f"    - outputs/hub_metrics.csv")
    print(f"    - outputs/corridor_metrics.csv")
    print(f"    - outputs/model_comparison.csv")
    print(f"    - outputs/ftl_vs_carting_comparison.csv")
    print(f"    - outputs/route_type_decision_matrix.csv")
    print(f"    - outputs/graphs/ (7 visualization PNGs)")
    print(f"    - outputs/models/ (trained XGBoost models)")
    
    print(f"\n  Final Model Comparison:")
    print(results_df.to_string(index=False))
    
    # Export memo data
    _export_memo_data(node_metrics, corridor_df, results_df, comparison, df)
    
    return df, G, node_metrics, corridor_df, results_df, models


def _export_memo_data(node_metrics, corridor_df, results_df, comparison, df):
    """Export data needed for the strategy memo."""
    import json
    
    memo_data = {}
    
    # Top 5 bottleneck hubs
    top5 = node_metrics.nsmallest(5, 'bottleneck_rank')
    memo_data['top_5_hubs'] = top5[[
        'facility_name', 'state', 'bottleneck_score', 'bottleneck_rank',
        'betweenness_centrality', 'pct_delayed_out', 'total_volume',
        'sla_contribution_pct'
    ]].to_dict('records')
    
    # Top 10 delay corridors
    top_corridors = corridor_df.nlargest(10, 'breach_impact')
    memo_data['top_10_corridors'] = top_corridors[[
        'source_name', 'dest_name', 'trip_count', 'median_delay_ratio',
        'pct_delayed', 'breach_impact'
    ]].to_dict('records')
    
    # Model comparison
    memo_data['model_comparison'] = results_df.to_dict('records')
    
    # Overall stats
    memo_data['overall_stats'] = {
        'total_trips': int(df['trip_uuid'].nunique()),
        'total_segments': len(df),
        'sla_breach_rate': float(df['is_delayed'].mean() * 100),
        'avg_delay_ratio': float(df['delay_ratio'].mean()),
        'states_covered': int(df['source_state'].nunique()),
    }
    
    with open('outputs/memo_data.json', 'w') as f:
        json.dump(memo_data, f, indent=2, default=str)
    
    print("  Memo data exported to outputs/memo_data.json")


if __name__ == "__main__":
    main()
