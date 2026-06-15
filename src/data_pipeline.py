"""
data_pipeline.py — Data loading, cleaning, and feature extraction for Delhivery logistics data.

This module handles:
1. Loading raw CSV data with proper type parsing
2. Cleaning invalid/missing records
3. Extracting temporal features (hour, day, time-of-day bin)
4. Extracting geographic features (state from facility names)
5. Computing derived delay metrics
6. Deduplicating trip segments
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple


# ─── Constants ────────────────────────────────────────────────────────────────

TIME_BINS = {
    'night':     (0, 6),
    'morning':   (6, 12),
    'afternoon': (12, 18),
    'evening':   (18, 24),
}

SLA_BREACH_THRESHOLD = 1.2  # delay_ratio > 1.2 means >20% overrun


# ─── Core Pipeline ───────────────────────────────────────────────────────────

def load_raw_data(filepath: str = "delivery_data.csv") -> pd.DataFrame:
    """Load the raw delivery CSV with proper dtypes."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(
        filepath,
        parse_dates=[
            'trip_creation_time', 'od_start_time', 'od_end_time', 'cutoff_timestamp'
        ],
        dtype={
            'data': 'category',
            'route_type': 'category',
            'is_cutoff': 'bool',
            'source_center': 'str',
            'destination_center': 'str',
            'source_name': 'str',
            'destination_name': 'str',
        }
    )
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by removing invalid records.
    
    Rules:
    - Drop rows where osrm_time <= 0 (can't compute meaningful delay ratio)
    - Drop rows where actual_time <= 0
    - Drop rows where segment_osrm_time <= 0
    - Drop rows with missing source/destination centers
    - Cap extreme outliers: delay_ratio > 20 likely data errors
    """
    initial_count = len(df)
    
    # Remove invalid timing records
    df = df[df['osrm_time'] > 0].copy()
    df = df[df['actual_time'] > 0].copy()
    df = df[df['segment_osrm_time'] > 0].copy()
    df = df[df['segment_actual_time'] > 0].copy()
    
    # Remove missing facility codes
    df = df.dropna(subset=['source_center', 'destination_center'])
    
    # Remove self-loops (source == destination)
    df = df[df['source_center'] != df['destination_center']].copy()
    
    removed = initial_count - len(df)
    print(f"  Cleaned: removed {removed:,} invalid rows ({removed/initial_count*100:.1f}%)")
    print(f"  Remaining: {len(df):,} rows")
    return df


def extract_state(name: str) -> str:
    """Extract state from facility name like 'Anand_VUNagar_DC (Gujarat)' → 'Gujarat'."""
    if pd.isna(name):
        return 'Unknown'
    try:
        # State is in parentheses at the end
        start = name.rfind('(')
        end = name.rfind(')')
        if start != -1 and end != -1 and end > start:
            return name[start + 1:end].strip()
    except (ValueError, IndexError):
        pass
    return 'Unknown'


def get_time_bin(hour: int) -> str:
    """Map hour (0-23) to time-of-day bin."""
    for bin_name, (start, end) in TIME_BINS.items():
        if start <= hour < end:
            return bin_name
    return 'night'  # fallback


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to the dataset.
    
    Features added:
    - hour_of_day, day_of_week, is_weekend: from od_start_time
    - time_bin: categorical time-of-day bucket
    - source_state, dest_state: extracted from facility names
    - delay_ratio: segment_actual_time / segment_osrm_time
    - is_delayed: delay_ratio > SLA_BREACH_THRESHOLD
    - trip_segment_count: number of segments per trip
    - distance_ratio: actual_distance / osrm_distance
    """
    print("  Extracting features...")
    
    # ── Temporal features ──
    df['hour_of_day'] = df['od_start_time'].dt.hour
    df['day_of_week'] = df['od_start_time'].dt.dayofweek  # 0=Mon, 6=Sun
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['time_bin'] = df['hour_of_day'].apply(get_time_bin).astype('category')
    
    # ── Geographic features ──
    df['source_state'] = df['source_name'].apply(extract_state)
    df['dest_state'] = df['destination_name'].apply(extract_state)
    
    # ── Delay metrics ──
    df['delay_ratio'] = df['segment_actual_time'] / df['segment_osrm_time']
    df['is_delayed'] = (df['delay_ratio'] > SLA_BREACH_THRESHOLD).astype(int)
    
    # Cap extreme outliers (delay_ratio > 20x is likely data error)
    df.loc[df['delay_ratio'] > 20, 'delay_ratio'] = 20.0
    
    # ── Trip-level features ──
    segment_counts = df.groupby('trip_uuid').size().rename('trip_segment_count')
    df = df.merge(segment_counts, on='trip_uuid', how='left')
    
    # ── Distance ratio ──
    df['distance_ratio'] = np.where(
        df['osrm_distance'] > 0,
        df['actual_distance_to_destination'] / df['osrm_distance'],
        1.0
    )
    
    print(f"  Features added. Final shape: {df.shape}")
    return df


def deduplicate_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate trip segments.
    
    Multiple rows can exist for the same segment if cutoff_factor differs.
    We keep the row with the highest cutoff_factor (most complete observation).
    """
    initial_count = len(df)
    
    df = df.sort_values('cutoff_factor', ascending=False)
    df = df.drop_duplicates(
        subset=['trip_uuid', 'source_center', 'destination_center'],
        keep='first'
    )
    
    removed = initial_count - len(df)
    print(f"  Deduplicated: removed {removed:,} duplicate segments")
    print(f"  Remaining: {len(df):,} unique trip-segments")
    return df


def get_train_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data using the built-in 'data' column (training/test)."""
    train = df[df['data'] == 'training'].copy()
    test = df[df['data'] == 'test'].copy()
    print(f"  Train: {len(train):,} rows | Test: {len(test):,} rows")
    return train, test


def run_pipeline(filepath: str = "delivery_data.csv") -> pd.DataFrame:
    """Run the full data pipeline end-to-end."""
    print("=" * 60)
    print("DELHIVERY DATA PIPELINE")
    print("=" * 60)
    
    df = load_raw_data(filepath)
    df = clean_data(df)
    df = deduplicate_segments(df)
    df = add_features(df)
    
    # Print summary statistics
    print("\n── Summary ──")
    print(f"  Unique facilities: {df['source_center'].nunique() + df['destination_center'].nunique()}")
    print(f"  Unique source facilities: {df['source_center'].nunique()}")
    print(f"  Unique destination facilities: {df['destination_center'].nunique()}")
    print(f"  Unique trips: {df['trip_uuid'].nunique()}")
    print(f"  Unique corridors: {df.groupby(['source_center', 'destination_center']).ngroups}")
    print(f"  Route types: {df['route_type'].value_counts().to_dict()}")
    print(f"  Delay ratio: mean={df['delay_ratio'].mean():.2f}, "
          f"median={df['delay_ratio'].median():.2f}")
    print(f"  SLA breach rate: {df['is_delayed'].mean()*100:.1f}%")
    print(f"  States covered: {df['source_state'].nunique()}")
    print("=" * 60)
    
    return df


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = run_pipeline()
    print("\nSample data:")
    print(df[['source_center', 'destination_center', 'route_type', 
              'segment_actual_time', 'segment_osrm_time', 'delay_ratio',
              'is_delayed', 'time_bin', 'source_state']].head(10))
