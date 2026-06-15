"""
bottleneck_analysis.py — Identify critical chokepoint hubs and chronically delayed corridors.

Computes:
1. Node centrality metrics (betweenness, degree, PageRank, clustering)
2. Composite bottleneck score
3. Corridor-level SLA breach analysis
4. Hub ranking for the strategy memo
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Tuple


# ─── Node-Level Metrics ──────────────────────────────────────────────────────

def compute_node_centrality(G: nx.DiGraph) -> pd.DataFrame:
    """
    Compute centrality and structural metrics for every node (facility).
    
    Metrics:
    - betweenness_centrality: fraction of shortest paths passing through node
    - in_degree / out_degree: incoming/outgoing corridor count
    - weighted_in_degree / weighted_out_degree: by trip volume
    - clustering_coefficient: local connectivity density
    - pagerank: importance in the flow network
    """
    print("Computing node centrality metrics...")
    
    metrics = {}
    
    # Betweenness centrality (weighted by median_delay_ratio for realism)
    try:
        bc = nx.betweenness_centrality(G, weight='median_delay_ratio', normalized=True)
    except Exception:
        bc = nx.betweenness_centrality(G, normalized=True)
    
    # PageRank
    try:
        pr = nx.pagerank(G, weight='trip_count', max_iter=200)
    except Exception:
        pr = {n: 1.0 / G.number_of_nodes() for n in G.nodes()}
    
    # Clustering coefficient (on undirected version)
    undirected = G.to_undirected()
    cc = nx.clustering(undirected)
    
    # Degree metrics
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    w_in_deg = dict(G.in_degree(weight='trip_count'))
    w_out_deg = dict(G.out_degree(weight='trip_count'))
    
    for node in G.nodes():
        node_data = G.nodes[node]
        metrics[node] = {
            'facility_code': node,
            'facility_name': node_data.get('facility_name', 'Unknown'),
            'state': node_data.get('state', 'Unknown'),
            'total_volume': node_data.get('total_volume', 0),
            'betweenness_centrality': round(bc.get(node, 0), 6),
            'pagerank': round(pr.get(node, 0), 6),
            'clustering_coefficient': round(cc.get(node, 0), 4),
            'in_degree': in_deg.get(node, 0),
            'out_degree': out_deg.get(node, 0),
            'weighted_in_degree': round(w_in_deg.get(node, 0), 1),
            'weighted_out_degree': round(w_out_deg.get(node, 0), 1),
            'avg_out_delay': node_data.get('avg_out_delay', 0),
            'median_out_delay': node_data.get('median_out_delay', 0),
            'pct_delayed_out': node_data.get('pct_delayed_out', 0),
            'avg_in_delay': node_data.get('avg_in_delay', 0),
            'pct_delayed_in': node_data.get('pct_delayed_in', 0),
        }
    
    df = pd.DataFrame(metrics.values())
    print(f"  Computed metrics for {len(df)} facilities")
    return df


def compute_bottleneck_score(node_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a composite bottleneck score for each hub.
    
    Score = weighted combination of:
    - Normalized betweenness centrality (30%) — structural importance
    - Normalized pct_delayed_out (40%) — operational unreliability
    - Normalized total_volume (30%) — impact scale
    
    Higher score = more critical bottleneck that needs attention.
    """
    df = node_metrics.copy()
    
    def normalize(series):
        """Min-max normalize to [0, 1]."""
        s_min, s_max = series.min(), series.max()
        if s_max == s_min:
            return pd.Series(0.5, index=series.index)
        return (series - s_min) / (s_max - s_min)
    
    df['norm_betweenness'] = normalize(df['betweenness_centrality'])
    df['norm_delay'] = normalize(df['pct_delayed_out'])
    df['norm_volume'] = normalize(df['total_volume'])
    
    df['bottleneck_score'] = (
        0.30 * df['norm_betweenness'] +
        0.40 * df['norm_delay'] +
        0.30 * df['norm_volume']
    )
    
    df = df.sort_values('bottleneck_score', ascending=False)
    df['bottleneck_rank'] = range(1, len(df) + 1)
    
    print(f"\n── Top 10 Bottleneck Hubs ──")
    top10 = df.head(10)[['bottleneck_rank', 'facility_name', 'state',
                          'bottleneck_score', 'betweenness_centrality',
                          'pct_delayed_out', 'total_volume']]
    print(top10.to_string(index=False))
    
    return df


# ─── Corridor-Level Analysis ─────────────────────────────────────────────────

def identify_chronic_delay_corridors(G: nx.DiGraph, threshold: float = 1.2) -> pd.DataFrame:
    """
    Identify corridors where actual time consistently exceeds OSRM by >20%.
    
    Returns corridors ranked by SLA breach contribution.
    """
    records = []
    for u, v, data in G.edges(data=True):
        records.append({
            'source': u,
            'destination': v,
            'source_name': G.nodes[u].get('facility_name', u),
            'dest_name': G.nodes[v].get('facility_name', v),
            'source_state': G.nodes[u].get('state', 'Unknown'),
            'dest_state': G.nodes[v].get('state', 'Unknown'),
            'trip_count': data.get('trip_count', 0),
            'median_delay_ratio': data.get('median_delay_ratio', 1.0),
            'mean_delay_ratio': data.get('mean_delay_ratio', 1.0),
            'pct_delayed': data.get('pct_delayed', 0),
            'sla_breach_count': data.get('sla_breach_count', 0),
            'sla_breach_contribution': data.get('sla_breach_contribution', 0),
            'median_actual_time': data.get('median_actual_time', 0),
            'median_osrm_time': data.get('median_osrm_time', 0),
            'median_distance': data.get('median_distance', 0),
        })
    
    corridors = pd.DataFrame(records)
    
    # Flag chronic delay corridors
    corridors['is_chronic_delay'] = corridors['median_delay_ratio'] > threshold
    
    # Sort by SLA breach contribution (volume × delay rate)
    corridors['breach_impact'] = corridors['trip_count'] * corridors['pct_delayed']
    corridors = corridors.sort_values('breach_impact', ascending=False)
    
    chronic = corridors[corridors['is_chronic_delay']]
    print(f"\n── Chronic Delay Corridors (delay ratio > {threshold}) ──")
    print(f"  {len(chronic)} of {len(corridors)} corridors are chronically delayed "
          f"({len(chronic)/len(corridors)*100:.1f}%)")
    print(f"  These account for {chronic['sla_breach_count'].sum():.0f} SLA breaches")
    
    return corridors


# ─── SLA Breach Contribution ─────────────────────────────────────────────────

def compute_hub_sla_contribution(
    node_metrics: pd.DataFrame, 
    corridor_df: pd.DataFrame
) -> pd.DataFrame:
    """
    For each hub, estimate its contribution to total SLA breaches.
    
    A hub contributes to breaches on both its incoming and outgoing corridors.
    We sum breach_impact for all corridors touching the hub.
    """
    total_breach_impact = corridor_df['breach_impact'].sum()
    
    hub_breach = {}
    for _, row in node_metrics.iterrows():
        fac = row['facility_code']
        
        # Outgoing corridors
        out_impact = corridor_df[corridor_df['source'] == fac]['breach_impact'].sum()
        # Incoming corridors
        in_impact = corridor_df[corridor_df['destination'] == fac]['breach_impact'].sum()
        
        hub_breach[fac] = {
            'outgoing_breach_impact': out_impact,
            'incoming_breach_impact': in_impact,
            'total_breach_impact': out_impact + in_impact,
            'sla_contribution_pct': (
                (out_impact + in_impact) / total_breach_impact * 100
                if total_breach_impact > 0 else 0
            ),
        }
    
    breach_df = pd.DataFrame(hub_breach).T
    breach_df.index.name = 'facility_code'
    breach_df = breach_df.reset_index()
    
    # Merge with node metrics
    result = node_metrics.merge(breach_df, on='facility_code', how='left')
    result = result.sort_values('sla_contribution_pct', ascending=False)
    
    print(f"\n── Top 10 Hubs by SLA Breach Contribution ──")
    top10 = result.head(10)[['facility_name', 'state', 'sla_contribution_pct',
                              'total_volume', 'bottleneck_score']]
    print(top10.to_string(index=False))
    
    return result


def run_bottleneck_analysis(G: nx.DiGraph) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full bottleneck analysis pipeline."""
    print("=" * 60)
    print("BOTTLENECK & CORRIDOR AUDIT")
    print("=" * 60)
    
    # Node metrics
    node_metrics = compute_node_centrality(G)
    node_metrics = compute_bottleneck_score(node_metrics)
    
    # Corridor analysis
    corridor_df = identify_chronic_delay_corridors(G)
    
    # Hub SLA contribution
    node_metrics = compute_hub_sla_contribution(node_metrics, corridor_df)
    
    # Save results
    node_metrics.to_csv('outputs/hub_metrics.csv', index=False)
    corridor_df.to_csv('outputs/corridor_metrics.csv', index=False)
    print(f"\n  Results saved to outputs/hub_metrics.csv and outputs/corridor_metrics.csv")
    
    return node_metrics, corridor_df


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_pipeline import run_pipeline
    from graph_builder import build_graph
    
    df = run_pipeline()
    G = build_graph(df)
    node_metrics, corridor_df = run_bottleneck_analysis(G)
