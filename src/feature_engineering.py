"""
feature_engineering.py — Prepare features for ETA prediction models.

Two feature sets:
1. Baseline: trip-level features only (distance, time, route type, etc.)
2. Graph-enhanced: baseline + graph-derived features (centrality, historical corridor stats, node2vec embeddings)
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Optional


def build_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build baseline feature matrix using only trip-level information.
    
    Features:
    - osrm_time, osrm_distance, actual_distance_to_destination
    - route_type (encoded)
    - hour_of_day, day_of_week, is_weekend
    - is_cutoff, cutoff_factor
    - trip_segment_count
    - segment_osrm_time, segment_osrm_distance
    """
    features = pd.DataFrame(index=df.index)
    
    # Numeric features
    features['segment_osrm_time'] = df['segment_osrm_time']
    features['segment_osrm_distance'] = df['segment_osrm_distance']
    features['osrm_time'] = df['osrm_time']
    features['osrm_distance'] = df['osrm_distance']
    features['actual_distance_to_destination'] = df['actual_distance_to_destination']
    features['cutoff_factor'] = df['cutoff_factor']
    features['trip_segment_count'] = df['trip_segment_count']
    features['start_scan_to_end_scan'] = df['start_scan_to_end_scan']
    
    # Temporal features
    features['hour_of_day'] = df['hour_of_day']
    features['day_of_week'] = df['day_of_week']
    features['is_weekend'] = df['is_weekend']
    
    # Categorical features (encoded)
    features['is_cutoff'] = df['is_cutoff'].astype(int)
    features['is_ftl'] = (df['route_type'] == 'FTL').astype(int)
    
    # Time bin encoding
    time_bin_map = {'night': 0, 'morning': 1, 'afternoon': 2, 'evening': 3}
    features['time_bin_encoded'] = df['time_bin'].map(time_bin_map).fillna(0).astype(int)
    
    # Interaction features
    features['osrm_speed'] = np.where(
        features['segment_osrm_time'] > 0,
        features['segment_osrm_distance'] / features['segment_osrm_time'],
        0
    )
    features['distance_per_segment'] = np.where(
        features['trip_segment_count'] > 0,
        features['actual_distance_to_destination'] / features['trip_segment_count'],
        0
    )
    
    return features


def build_graph_features(
    df: pd.DataFrame, 
    G: nx.DiGraph,
    node_metrics: pd.DataFrame,
    corridor_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build graph-derived features to enhance ETA predictions.
    
    For each trip segment, adds:
    - Source node metrics: betweenness, pagerank, degree, clustering, delay history
    - Destination node metrics: same
    - Corridor metrics: historical delay ratio, SLA breach rate, volume
    """
    features = pd.DataFrame(index=df.index)
    
    # ── Create lookup dictionaries from node_metrics ──
    node_lookup = node_metrics.set_index('facility_code')
    
    node_feature_cols = [
        'betweenness_centrality', 'pagerank', 'clustering_coefficient',
        'in_degree', 'out_degree', 'weighted_in_degree', 'weighted_out_degree',
        'avg_out_delay', 'median_out_delay', 'pct_delayed_out',
        'avg_in_delay', 'pct_delayed_in', 'total_volume',
    ]
    
    # Check which columns exist
    available_cols = [c for c in node_feature_cols if c in node_lookup.columns]
    
    # ── Source node features ──
    for col in available_cols:
        src_map = node_lookup[col].to_dict()
        features[f'src_{col}'] = df['source_center'].map(src_map).fillna(0)
    
    # ── Destination node features ──
    for col in available_cols:
        dst_map = node_lookup[col].to_dict()
        features[f'dst_{col}'] = df['destination_center'].map(dst_map).fillna(0)
    
    # ── Corridor features ──
    corridor_lookup = corridor_df.set_index(['source', 'destination'])
    
    corridor_feature_cols = [
        'median_delay_ratio', 'mean_delay_ratio', 'pct_delayed',
        'trip_count', 'median_actual_time', 'median_osrm_time',
    ]
    available_corridor = [c for c in corridor_feature_cols if c in corridor_lookup.columns]
    
    # Create (source, dest) tuple for lookup
    corridor_keys = list(zip(df['source_center'], df['destination_center']))
    
    for col in available_corridor:
        col_map = corridor_lookup[col].to_dict()
        features[f'corridor_{col}'] = [col_map.get(k, 0) for k in corridor_keys]
    
    # ── Derived graph features ──
    # Centrality difference (asymmetry between source and destination)
    if 'src_betweenness_centrality' in features.columns:
        features['centrality_diff'] = (
            features['src_betweenness_centrality'] - features['dst_betweenness_centrality']
        )
    
    # Volume asymmetry
    if 'src_total_volume' in features.columns:
        features['volume_ratio'] = np.where(
            features['dst_total_volume'] > 0,
            features['src_total_volume'] / features['dst_total_volume'],
            1.0
        )
    
    # Source reliability score (inverse of delay rate)
    if 'src_pct_delayed_out' in features.columns:
        features['src_reliability'] = 1.0 - features['src_pct_delayed_out']
    
    return features


def build_node2vec_features(
    G: nx.DiGraph, 
    df: pd.DataFrame,
    dimensions: int = 64,
    walk_length: int = 20,
    num_walks: int = 80,
    p: float = 1.0,
    q: float = 1.0,
) -> pd.DataFrame:
    """
    Generate Node2Vec embeddings for source and destination facilities.
    
    Uses random walks on the graph to learn node representations that
    capture structural similarity and graph position.
    """
    print("Generating Node2Vec embeddings...")
    
    try:
        from node2vec import Node2Vec
        
        # Node2Vec works on undirected graphs; convert but keep weights
        G_undirected = G.to_undirected()
        
        # Remove edge attributes that might cause issues, keep only 'weight'
        for u, v, data in G_undirected.edges(data=True):
            # Use trip_count as weight (higher traffic = more connected)
            weight = data.get('trip_count', 1)
            G_undirected[u][v]['weight'] = max(float(weight), 0.1)
        
        # Fit Node2Vec
        node2vec = Node2Vec(
            G_undirected, 
            dimensions=dimensions,
            walk_length=walk_length, 
            num_walks=num_walks,
            p=p, q=q, 
            workers=1,
            quiet=True,
        )
        
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        
        # Extract embeddings using VECTORIZED lookup (not iterrows)
        # Build embedding lookup dict: node -> vector
        embedding_dict = {}
        for node in model.wv.index_to_key:
            embedding_dict[node] = model.wv[node]
        
        default_vec = np.zeros(dimensions)
        
        # Vectorized: map source/dest to embeddings
        src_embeddings = df['source_center'].map(
            lambda x: embedding_dict.get(x, default_vec)
        )
        dst_embeddings = df['destination_center'].map(
            lambda x: embedding_dict.get(x, default_vec)
        )
        
        # Convert to DataFrame columns
        src_matrix = np.vstack(src_embeddings.values)
        dst_matrix = np.vstack(dst_embeddings.values)
        
        features = pd.DataFrame(index=df.index)
        for i in range(dimensions):
            features[f'src_n2v_{i}'] = src_matrix[:, i]
            features[f'dst_n2v_{i}'] = dst_matrix[:, i]
        
        print(f"  Generated {dimensions}-dim embeddings for {len(model.wv)} nodes")
        return features
        
    except ImportError:
        print("  Warning: node2vec not installed. Skipping embeddings.")
        return pd.DataFrame(index=df.index)
    except Exception as e:
        print(f"  Warning: Node2Vec failed ({e}). Skipping embeddings.")
        return pd.DataFrame(index=df.index)


def prepare_model_data(
    df: pd.DataFrame,
    G: nx.DiGraph,
    node_metrics: pd.DataFrame,
    corridor_df: pd.DataFrame,
    include_node2vec: bool = True,
    n2v_dimensions: int = 32,
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Prepare complete feature matrices and targets for model training.
    
    Returns dict with keys:
    - 'baseline': (X_baseline, y)
    - 'graph_enhanced': (X_graph, y)
    - 'graph_n2v': (X_graph_n2v, y)  [if node2vec available]
    
    For both train and test splits.
    """
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    target = df['segment_actual_time'].copy()
    
    # ── Baseline features ──
    baseline = build_baseline_features(df)
    print(f"  Baseline features: {baseline.shape[1]} columns")
    
    # ── Graph features ──
    graph_feats = build_graph_features(df, G, node_metrics, corridor_df)
    graph_enhanced = pd.concat([baseline, graph_feats], axis=1)
    print(f"  Graph-enhanced features: {graph_enhanced.shape[1]} columns")
    
    # ── Node2Vec features ──
    n2v_df = pd.DataFrame(index=df.index)
    if include_node2vec:
        n2v_df = build_node2vec_features(G, df, dimensions=n2v_dimensions)
        if not n2v_df.empty:
            graph_n2v = pd.concat([graph_enhanced, n2v_df], axis=1)
            print(f"  Graph+Node2Vec features: {graph_n2v.shape[1]} columns")
        else:
            graph_n2v = graph_enhanced.copy()
    else:
        graph_n2v = graph_enhanced.copy()
    
    # ── Clean up: fill NaN, handle infinities ──
    for feat_df in [baseline, graph_enhanced, graph_n2v]:
        feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        feat_df.fillna(0, inplace=True)
    
    # ── Split by train/test ──
    train_mask = df['data'] == 'training'
    test_mask = df['data'] == 'test'
    
    result = {
        'train': {
            'baseline': (baseline[train_mask], target[train_mask]),
            'graph_enhanced': (graph_enhanced[train_mask], target[train_mask]),
            'graph_n2v': (graph_n2v[train_mask], target[train_mask]),
        },
        'test': {
            'baseline': (baseline[test_mask], target[test_mask]),
            'graph_enhanced': (graph_enhanced[test_mask], target[test_mask]),
            'graph_n2v': (graph_n2v[test_mask], target[test_mask]),
        },
        'feature_names': {
            'baseline': list(baseline.columns),
            'graph_enhanced': list(graph_enhanced.columns),
            'graph_n2v': list(graph_n2v.columns),
        }
    }
    
    print(f"\n  Train samples: {train_mask.sum():,}")
    print(f"  Test samples: {test_mask.sum():,}")
    
    return result
