"""
visualizations.py — All graph and chart visualizations for the Delhivery network analysis.

Generates:
1. Network graph with bottleneck hubs and delay corridors highlighted
2. Heatmap: corridors × time-of-day delay ratios
3. Bar charts: top hubs by centrality and SLA breach contribution
4. Scatter plots: structural importance vs operational delay
5. Distribution plots: delay ratios by route type
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path

# ── Style config ──
sns.set_theme(style='whitegrid', font_scale=1.1)
COLORS = {
    'primary': '#1a73e8',
    'danger': '#e53935',
    'warning': '#fb8c00',
    'success': '#43a047',
    'dark': '#1e1e2f',
    'light': '#f5f5f5',
}
OUTPUT_DIR = Path('outputs/graphs')


def save_fig(fig, name: str, dpi: int = 150):
    """Save figure to outputs directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / f"{name}.png"
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {filepath}")


# ─── 1. Network Graph Visualization ──────────────────────────────────────────

def plot_network_graph(
    G: nx.DiGraph,
    node_metrics: pd.DataFrame,
    corridor_df: pd.DataFrame,
    top_n_hubs: int = 10,
    top_n_corridors: int = 15,
):
    """
    Visualize the logistics network with bottleneck hubs and delay corridors.
    
    - Node size ∝ total volume
    - Node color ∝ bottleneck score (green → red)
    - Edge width ∝ trip count
    - Edge color ∝ delay ratio
    - Top hubs are labeled
    """
    print("Generating network graph visualization...")
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    
    # Layout
    pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)
    
    # ── Node sizes and colors ──
    score_map = dict(zip(node_metrics['facility_code'], node_metrics['bottleneck_score']))
    volume_map = dict(zip(node_metrics['facility_code'], node_metrics['total_volume']))
    
    node_sizes = []
    node_colors = []
    for node in G.nodes():
        vol = volume_map.get(node, 1)
        node_sizes.append(max(20, min(vol * 0.5, 500)))
        node_colors.append(score_map.get(node, 0))
    
    # ── Edge widths and colors ──
    edge_widths = []
    edge_colors = []
    for u, v, data in G.edges(data=True):
        tc = data.get('trip_count', 1)
        edge_widths.append(max(0.3, min(tc * 0.05, 3.0)))
        delay = data.get('median_delay_ratio', 1.0)
        edge_colors.append(delay)
    
    # Draw edges
    edge_cmap = plt.cm.RdYlGn_r  # Red = high delay, Green = low
    if edge_colors:
        edges = nx.draw_networkx_edges(
            G, pos, ax=ax, width=edge_widths, alpha=0.3,
            edge_color=edge_colors, edge_cmap=edge_cmap,
            edge_vmin=0.8, edge_vmax=3.0, arrows=True, arrowsize=8,
            connectionstyle="arc3,rad=0.1"
        )
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=node_sizes,
        node_color=node_colors, cmap=plt.cm.RdYlGn_r,
        vmin=0, vmax=1, alpha=0.85, edgecolors='white', linewidths=0.5
    )
    
    # Label top bottleneck hubs
    top_hubs = node_metrics.nsmallest(top_n_hubs, 'bottleneck_rank')
    labels = {}
    for _, row in top_hubs.iterrows():
        fac = row['facility_code']
        name = str(row.get('facility_name', fac))
        # Shorten name
        short = name.split('_')[0] if '_' in name else name[:15]
        labels[fac] = f"#{int(row['bottleneck_rank'])} {short}"
    
    nx.draw_networkx_labels(
        G, pos, labels, ax=ax, font_size=7, font_weight='bold',
        font_color='black',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7)
    )
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label('Bottleneck Score', fontsize=12)
    
    ax.set_title("Delhivery Logistics Network\nNode color = Bottleneck Score | Edge color = Delay Ratio",
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_facecolor('#fafafa')
    ax.axis('off')
    
    save_fig(fig, 'network_graph')


# ─── 2. Top Bottleneck Hubs Bar Chart ────────────────────────────────────────

def plot_bottleneck_hubs(node_metrics: pd.DataFrame, top_n: int = 20):
    """Bar chart of top bottleneck hubs by composite score."""
    print("Generating bottleneck hubs chart...")
    
    top = node_metrics.nsmallest(top_n, 'bottleneck_rank').copy()
    top['label'] = top.apply(
        lambda r: f"{str(r['facility_name']).split('_')[0]}\n({r['state']})", axis=1
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    
    # Bottleneck Score
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(top)))
    axes[0].barh(top['label'], top['bottleneck_score'], color=colors)
    axes[0].set_xlabel('Bottleneck Score')
    axes[0].set_title('Composite Bottleneck Score', fontweight='bold')
    axes[0].invert_yaxis()
    
    # Betweenness Centrality
    axes[1].barh(top['label'], top['betweenness_centrality'], 
                 color=COLORS['primary'], alpha=0.8)
    axes[1].set_xlabel('Betweenness Centrality')
    axes[1].set_title('Betweenness Centrality', fontweight='bold')
    axes[1].invert_yaxis()
    
    # SLA Breach Contribution
    if 'sla_contribution_pct' in top.columns:
        axes[2].barh(top['label'], top['sla_contribution_pct'],
                     color=COLORS['danger'], alpha=0.8)
        axes[2].set_xlabel('SLA Breach Contribution (%)')
        axes[2].set_title('SLA Breach Contribution', fontweight='bold')
    else:
        axes[2].barh(top['label'], top['pct_delayed_out'] * 100,
                     color=COLORS['danger'], alpha=0.8)
        axes[2].set_xlabel('% Outgoing Trips Delayed')
        axes[2].set_title('Outgoing Delay Rate', fontweight='bold')
    axes[2].invert_yaxis()
    
    fig.suptitle(f'Top {top_n} Bottleneck Hubs', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, 'bottleneck_hubs')


# ─── 3. Betweenness vs Delay Scatter ─────────────────────────────────────────

def plot_centrality_vs_delay(node_metrics: pd.DataFrame):
    """Scatter: betweenness centrality vs delay rate, sized by volume."""
    print("Generating centrality vs delay scatter...")
    
    df = node_metrics[node_metrics['total_volume'] > 5].copy()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    scatter = ax.scatter(
        df['betweenness_centrality'],
        df['pct_delayed_out'] * 100,
        s=np.clip(df['total_volume'] * 0.3, 10, 500),
        c=df['bottleneck_score'],
        cmap='RdYlGn_r', alpha=0.65,
        edgecolors='grey', linewidths=0.5
    )
    
    # Label the danger zone (high centrality AND high delay)
    danger = df[
        (df['betweenness_centrality'] > df['betweenness_centrality'].quantile(0.9)) &
        (df['pct_delayed_out'] > df['pct_delayed_out'].quantile(0.7))
    ]
    for _, row in danger.iterrows():
        name = str(row['facility_name']).split('_')[0]
        ax.annotate(name, (row['betweenness_centrality'], row['pct_delayed_out'] * 100),
                    fontsize=7, fontweight='bold', alpha=0.8,
                    xytext=(5, 5), textcoords='offset points')
    
    # Quadrant lines
    bc_med = df['betweenness_centrality'].median()
    delay_med = df['pct_delayed_out'].median() * 100
    ax.axvline(bc_med, color='grey', linestyle='--', alpha=0.4)
    ax.axhline(delay_med, color='grey', linestyle='--', alpha=0.4)
    
    # Quadrant labels
    ax.text(0.98, 0.98, "High Centrality\nHigh Delay\n⚠ CRITICAL", 
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            color=COLORS['danger'], fontweight='bold', alpha=0.7)
    ax.text(0.02, 0.02, "Low Centrality\nLow Delay\n✓ Healthy",
            transform=ax.transAxes, ha='left', va='bottom', fontsize=9,
            color=COLORS['success'], fontweight='bold', alpha=0.7)
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label('Bottleneck Score')
    
    ax.set_xlabel('Betweenness Centrality', fontsize=12)
    ax.set_ylabel('% Outgoing Trips Delayed (SLA Breach)', fontsize=12)
    ax.set_title('Hub Structural Importance vs Operational Delay\n'
                 'Size = Trip Volume | Color = Bottleneck Score',
                 fontsize=14, fontweight='bold')
    
    save_fig(fig, 'centrality_vs_delay')


# ─── 4. Delay Distribution by Route Type ─────────────────────────────────────

def plot_delay_distribution(df: pd.DataFrame):
    """Distribution of delay ratios by route type."""
    print("Generating delay distribution plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram
    for rt in df['route_type'].unique():
        subset = df[df['route_type'] == rt]['delay_ratio'].clip(upper=5)
        axes[0].hist(subset, bins=50, alpha=0.6, label=str(rt), density=True)
    
    axes[0].axvline(1.0, color='green', linestyle='--', alpha=0.7, label='On-time (1.0)')
    axes[0].axvline(1.2, color='red', linestyle='--', alpha=0.7, label='SLA Breach (1.2)')
    axes[0].set_xlabel('Delay Ratio (Actual / OSRM)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Delay Ratio Distribution by Route Type', fontweight='bold')
    axes[0].legend()
    
    # Box plot by time bin
    order = ['night', 'morning', 'afternoon', 'evening']
    valid_bins = [b for b in order if b in df['time_bin'].unique()]
    sns.boxplot(data=df[df['delay_ratio'] < 5], x='time_bin', y='delay_ratio',
                hue='route_type', ax=axes[1], order=valid_bins,
                showfliers=False)
    axes[1].axhline(1.2, color='red', linestyle='--', alpha=0.5, label='SLA Breach')
    axes[1].set_xlabel('Time of Day')
    axes[1].set_ylabel('Delay Ratio')
    axes[1].set_title('Delay Ratio by Time of Day & Route Type', fontweight='bold')
    
    plt.tight_layout()
    save_fig(fig, 'delay_distribution')


# ─── 5. Corridor Delay Heatmap ───────────────────────────────────────────────

def plot_corridor_heatmap(df: pd.DataFrame, top_n: int = 20):
    """Heatmap of delay ratios across time-of-day bins for top corridors."""
    print("Generating corridor delay heatmap...")
    
    # Get top corridors by volume
    corridor_volume = df.groupby(['source_center', 'destination_center']).size()
    top_corridors = corridor_volume.nlargest(top_n).index
    
    # Compute median delay by corridor × time_bin
    subset = df[df.set_index(['source_center', 'destination_center']).index.isin(top_corridors)]
    
    pivot = subset.pivot_table(
        values='delay_ratio',
        index=['source_center', 'destination_center'],
        columns='time_bin',
        aggfunc='median'
    )
    
    # Reorder columns
    col_order = [c for c in ['night', 'morning', 'afternoon', 'evening'] if c in pivot.columns]
    pivot = pivot[col_order]
    
    # Create readable labels
    name_map = dict(zip(df['source_center'], df['source_name']))
    name_map.update(dict(zip(df['destination_center'], df['destination_name'])))
    
    labels = []
    for src, dst in pivot.index:
        src_short = str(name_map.get(src, src)).split('_')[0][:12]
        dst_short = str(name_map.get(dst, dst)).split('_')[0][:12]
        labels.append(f"{src_short} → {dst_short}")
    
    pivot.index = labels
    
    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.5)))
    
    sns.heatmap(
        pivot, annot=True, fmt='.2f', cmap='RdYlGn_r',
        center=1.2, vmin=0.8, vmax=3.0,
        linewidths=0.5, ax=ax,
        cbar_kws={'label': 'Median Delay Ratio'}
    )
    
    ax.set_title(f'Top {top_n} Corridors: Delay Ratio by Time of Day\n'
                 '(Red = Severe Delay, Green = On-Time)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Time of Day Bin')
    ax.set_ylabel('Corridor (Source → Destination)')
    
    plt.tight_layout()
    save_fig(fig, 'corridor_heatmap')


# ─── 6. SLA Breach Corridor Ranking ──────────────────────────────────────────

def plot_sla_breach_corridors(corridor_df: pd.DataFrame, top_n: int = 15):
    """Bar chart of corridors ranked by SLA breach contribution."""
    print("Generating SLA breach corridor ranking...")
    
    top = corridor_df.nlargest(top_n, 'breach_impact').copy()
    top['label'] = top.apply(
        lambda r: f"{str(r['source_name']).split('_')[0][:10]} → {str(r['dest_name']).split('_')[0][:10]}",
        axis=1
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Breach impact
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top)))
    axes[0].barh(top['label'], top['breach_impact'], color=colors)
    axes[0].set_xlabel('Breach Impact (Volume × Delay Rate)')
    axes[0].set_title('Top Corridors by SLA Breach Impact', fontweight='bold')
    axes[0].invert_yaxis()
    
    # Delay ratio with trip count as annotation
    axes[1].barh(top['label'], top['median_delay_ratio'], color=COLORS['warning'], alpha=0.8)
    for i, (_, row) in enumerate(top.iterrows()):
        axes[1].text(row['median_delay_ratio'] + 0.02, i,
                     f"n={int(row['trip_count'])}", va='center', fontsize=8)
    axes[1].axvline(1.2, color='red', linestyle='--', alpha=0.5, label='SLA Threshold')
    axes[1].set_xlabel('Median Delay Ratio')
    axes[1].set_title('Median Delay Ratio (n = trip count)', fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].legend()
    
    fig.suptitle(f'Top {top_n} SLA Breach Corridors', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, 'sla_breach_corridors')


# ─── 7. State-Level Summary ──────────────────────────────────────────────────

def plot_state_summary(df: pd.DataFrame):
    """State-level delay analysis."""
    print("Generating state-level summary...")
    
    state_stats = df.groupby('source_state').agg(
        trip_count=('trip_uuid', 'count'),
        mean_delay=('delay_ratio', 'mean'),
        pct_delayed=('is_delayed', 'mean'),
    ).reset_index()
    
    state_stats = state_stats[state_stats['trip_count'] >= 20]
    state_stats = state_stats.sort_values('pct_delayed', ascending=False).head(20)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.RdYlGn_r(state_stats['pct_delayed'] / state_stats['pct_delayed'].max())
    bars = ax.barh(state_stats['source_state'], state_stats['pct_delayed'] * 100, color=colors)
    
    for bar, count in zip(bars, state_stats['trip_count']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'n={count:,}', va='center', fontsize=8)
    
    ax.axvline(20, color='red', linestyle='--', alpha=0.5, label='20% threshold')
    ax.set_xlabel('% Trips Delayed (SLA Breach Rate)')
    ax.set_title('State-Level SLA Breach Rate', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.legend()
    
    plt.tight_layout()
    save_fig(fig, 'state_summary')


# ─── Master Visualization Runner ─────────────────────────────────────────────

def generate_all_visualizations(
    G: nx.DiGraph,
    df: pd.DataFrame,
    node_metrics: pd.DataFrame,
    corridor_df: pd.DataFrame,
):
    """Generate all visualizations."""
    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_network_graph(G, node_metrics, corridor_df)
    plot_bottleneck_hubs(node_metrics)
    plot_centrality_vs_delay(node_metrics)
    plot_delay_distribution(df)
    plot_corridor_heatmap(df)
    plot_sla_breach_corridors(corridor_df)
    plot_state_summary(df)
    
    print("\n✓ All visualizations generated in outputs/graphs/")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_pipeline import run_pipeline
    from graph_builder import build_graph
    from bottleneck_analysis import run_bottleneck_analysis
    
    df = run_pipeline()
    G = build_graph(df)
    node_metrics, corridor_df = run_bottleneck_analysis(G)
    generate_all_visualizations(G, df, node_metrics, corridor_df)
