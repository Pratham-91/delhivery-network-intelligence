"""
graph_builder.py — Construct a directed weighted graph of Delhivery's logistics network.

Nodes = facilities (source_center / destination_center)
Edges = corridors between facilities, weighted by delay metrics

Edge weights are stratified by:
  - route_type (FTL, Carting)
  - time_bin (night, morning, afternoon, evening)
"""

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, Any
import json


def build_node_attributes(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Compute node-level attributes from trip data.
    
    For each facility, we compute:
    - facility_name: human-readable name
    - state: geographic state
    - total_volume: total trips through (as source or destination)
    - avg_outgoing_delay: mean delay ratio for outgoing trips
    - avg_incoming_delay: mean delay ratio for incoming trips
    - pct_delayed_outgoing: fraction of outgoing trips that breach SLA
    """
    node_attrs = {}
    
    # ── Source-side stats ──
    src_stats = df.groupby('source_center').agg(
        source_name=('source_name', 'first'),
        source_state=('source_state', 'first'),
        out_volume=('trip_uuid', 'count'),
        avg_out_delay=('delay_ratio', 'mean'),
        median_out_delay=('delay_ratio', 'median'),
        pct_delayed_out=('is_delayed', 'mean'),
    )
    
    # ── Destination-side stats ──
    dst_stats = df.groupby('destination_center').agg(
        dest_name=('destination_name', 'first'),
        dest_state=('dest_state', 'first'),
        in_volume=('trip_uuid', 'count'),
        avg_in_delay=('delay_ratio', 'mean'),
        median_in_delay=('delay_ratio', 'median'),
        pct_delayed_in=('is_delayed', 'mean'),
    )
    
    # ── Merge into node attributes ──
    all_facilities = set(df['source_center'].unique()) | set(df['destination_center'].unique())
    
    for fac in all_facilities:
        attrs = {'facility_code': fac}
        
        if fac in src_stats.index:
            s = src_stats.loc[fac]
            attrs['facility_name'] = s['source_name']
            attrs['state'] = s['source_state']
            attrs['out_volume'] = int(s['out_volume'])
            attrs['avg_out_delay'] = round(float(s['avg_out_delay']), 3)
            attrs['median_out_delay'] = round(float(s['median_out_delay']), 3)
            attrs['pct_delayed_out'] = round(float(s['pct_delayed_out']), 3)
        else:
            attrs['out_volume'] = 0
            attrs['avg_out_delay'] = 0.0
            attrs['median_out_delay'] = 0.0
            attrs['pct_delayed_out'] = 0.0
        
        if fac in dst_stats.index:
            d = dst_stats.loc[fac]
            if 'facility_name' not in attrs or attrs.get('facility_name') is None:
                attrs['facility_name'] = d['dest_name']
                attrs['state'] = d['dest_state']
            attrs['in_volume'] = int(d['in_volume'])
            attrs['avg_in_delay'] = round(float(d['avg_in_delay']), 3)
            attrs['median_in_delay'] = round(float(d['median_in_delay']), 3)
            attrs['pct_delayed_in'] = round(float(d['pct_delayed_in']), 3)
        else:
            attrs['in_volume'] = 0
            attrs['avg_in_delay'] = 0.0
            attrs['median_in_delay'] = 0.0
            attrs['pct_delayed_in'] = 0.0
        
        attrs['total_volume'] = attrs['out_volume'] + attrs['in_volume']
        node_attrs[fac] = attrs
    
    return node_attrs


def build_edge_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute edge-level (corridor) attributes.
    
    For each (source, destination) corridor:
    - Aggregate delay metrics overall and stratified by route_type × time_bin
    - Compute trip count, median delay, SLA breach rate
    """
    # ── Overall corridor stats ──
    corridor_stats = df.groupby(['source_center', 'destination_center']).agg(
        trip_count=('trip_uuid', 'count'),
        median_delay_ratio=('delay_ratio', 'median'),
        mean_delay_ratio=('delay_ratio', 'mean'),
        std_delay_ratio=('delay_ratio', 'std'),
        pct_delayed=('is_delayed', 'mean'),
        median_actual_time=('segment_actual_time', 'median'),
        median_osrm_time=('segment_osrm_time', 'median'),
        median_distance=('segment_osrm_distance', 'median'),
        sla_breach_count=('is_delayed', 'sum'),
    ).reset_index()
    
    corridor_stats['std_delay_ratio'] = corridor_stats['std_delay_ratio'].fillna(0)
    
    # ── SLA breach contribution: proportion of all breaches attributable to this corridor ──
    total_breaches = corridor_stats['sla_breach_count'].sum()
    corridor_stats['sla_breach_contribution'] = (
        corridor_stats['sla_breach_count'] / total_breaches if total_breaches > 0 else 0
    )
    
    # ── Stratified stats by route_type ──
    route_stats = df.groupby(
        ['source_center', 'destination_center', 'route_type']
    ).agg(
        rt_trip_count=('trip_uuid', 'count'),
        rt_median_delay=('delay_ratio', 'median'),
        rt_pct_delayed=('is_delayed', 'mean'),
    ).reset_index()
    
    # Pivot route_type stats into columns
    for rt in df['route_type'].unique():
        rt_data = route_stats[route_stats['route_type'] == rt].copy()
        rt_clean = str(rt).lower().replace(' ', '_')
        rt_data = rt_data.rename(columns={
            'rt_trip_count': f'{rt_clean}_trip_count',
            'rt_median_delay': f'{rt_clean}_median_delay',
            'rt_pct_delayed': f'{rt_clean}_pct_delayed',
        })
        corridor_stats = corridor_stats.merge(
            rt_data[['source_center', 'destination_center',
                     f'{rt_clean}_trip_count', f'{rt_clean}_median_delay',
                     f'{rt_clean}_pct_delayed']],
            on=['source_center', 'destination_center'],
            how='left'
        )
    
    # ── Stratified stats by time_bin ──
    time_stats = df.groupby(
        ['source_center', 'destination_center', 'time_bin']
    ).agg(
        tb_median_delay=('delay_ratio', 'median'),
        tb_trip_count=('trip_uuid', 'count'),
    ).reset_index()
    
    for tb in df['time_bin'].unique():
        tb_data = time_stats[time_stats['time_bin'] == tb].copy()
        tb_clean = str(tb).lower()
        tb_data = tb_data.rename(columns={
            'tb_median_delay': f'{tb_clean}_median_delay',
            'tb_trip_count': f'{tb_clean}_trip_count',
        })
        corridor_stats = corridor_stats.merge(
            tb_data[['source_center', 'destination_center',
                     f'{tb_clean}_median_delay', f'{tb_clean}_trip_count']],
            on=['source_center', 'destination_center'],
            how='left'
        )
    
    corridor_stats = corridor_stats.fillna(0)
    return corridor_stats


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Build the directed weighted logistics graph.
    
    Returns a NetworkX DiGraph where:
    - Each node is a facility with rich attributes
    - Each edge is a corridor with delay metrics and stratified weights
    """
    print("Building logistics graph...")
    
    G = nx.DiGraph()
    
    # ── Add nodes ──
    node_attrs = build_node_attributes(df)
    for fac_code, attrs in node_attrs.items():
        G.add_node(fac_code, **attrs)
    print(f"  Nodes: {G.number_of_nodes()} facilities")
    
    # ── Add edges ──
    edge_df = build_edge_attributes(df)
    for _, row in edge_df.iterrows():
        src = row['source_center']
        dst = row['destination_center']
        edge_attrs = row.to_dict()
        # Remove source/dest from attrs (they're the edge endpoints)
        edge_attrs.pop('source_center')
        edge_attrs.pop('destination_center')
        # Convert numpy types to native Python types for serialization
        edge_attrs = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) 
                      for k, v in edge_attrs.items()}
        G.add_edge(src, dst, **edge_attrs)
    
    print(f"  Edges: {G.number_of_edges()} corridors")
    
    # ── Graph summary ──
    if G.number_of_nodes() > 0:
        # Connected components (treating as undirected for connectivity)
        undirected = G.to_undirected()
        components = list(nx.connected_components(undirected))
        print(f"  Connected components: {len(components)}")
        print(f"  Largest component: {len(max(components, key=len))} nodes")
        
        # Density
        density = nx.density(G)
        print(f"  Graph density: {density:.4f}")
        
        # Average degree
        avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
        print(f"  Average degree: {avg_degree:.1f}")
    
    return G


def save_graph(G: nx.DiGraph, filepath: str = "outputs/logistics_graph.graphml"):
    """Save graph to GraphML format for reuse."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, filepath)
    print(f"  Graph saved to {filepath}")


def get_corridor_summary(G: nx.DiGraph) -> pd.DataFrame:
    """Extract edge data as a DataFrame for analysis."""
    records = []
    for u, v, data in G.edges(data=True):
        record = {'source': u, 'destination': v}
        record.update(data)
        records.append(record)
    return pd.DataFrame(records)


def get_facility_summary(G: nx.DiGraph) -> pd.DataFrame:
    """Extract node data as a DataFrame for analysis."""
    records = []
    for node, data in G.nodes(data=True):
        record = {'facility_code': node}
        record.update(data)
        records.append(record)
    return pd.DataFrame(records)


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_pipeline import run_pipeline
    
    df = run_pipeline()
    G = build_graph(df)
    save_graph(G)
    
    # Print top corridors by delay
    corridors = get_corridor_summary(G)
    print("\n── Top 10 Corridors by Delay Ratio ──")
    top_delayed = corridors.nlargest(10, 'median_delay_ratio')
    print(top_delayed[['source', 'destination', 'trip_count', 
                       'median_delay_ratio', 'pct_delayed']].to_string())
    
    print("\n── Top 10 Corridors by Volume ──")
    top_volume = corridors.nlargest(10, 'trip_count')
    print(top_volume[['source', 'destination', 'trip_count',
                      'median_delay_ratio', 'pct_delayed']].to_string())
