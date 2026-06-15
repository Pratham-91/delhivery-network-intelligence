"""
dashboard.py — Streamlit interactive dashboard for Delhivery Network Intelligence.

Launch: streamlit run deliverables/dashboard.py

Features:
1. Network overview with key metrics
2. Hub risk scorecard (sortable, filterable)
3. Corridor delay heatmap
4. ETA model comparison
5. FTL vs Carting recommendation lookup
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json

# ── Page Config ──
st.set_page_config(
    page_title="Delhivery Network Intelligence",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1a73e8 0%, #e53935 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
    }
    .stDataFrame {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Data Loading ──
# Resolve project root relative to this script's location
PROJECT_ROOT = Path(__file__).resolve().parent.parent

@st.cache_data
def load_data():
    """Load all output data files."""
    data = {}
    output_dir = PROJECT_ROOT / 'outputs'
    
    try:
        data['hub_metrics'] = pd.read_csv(output_dir / 'hub_metrics.csv')
    except FileNotFoundError:
        st.error("Hub metrics not found. Run `python main.py` first.")
        return None
    
    try:
        data['corridor_metrics'] = pd.read_csv(output_dir / 'corridor_metrics.csv')
    except FileNotFoundError:
        data['corridor_metrics'] = pd.DataFrame()
    
    try:
        data['model_comparison'] = pd.read_csv(output_dir / 'model_comparison.csv')
    except FileNotFoundError:
        data['model_comparison'] = pd.DataFrame()
    
    try:
        data['ftl_comparison'] = pd.read_csv(output_dir / 'ftl_vs_carting_comparison.csv')
    except FileNotFoundError:
        data['ftl_comparison'] = pd.DataFrame()
    
    try:
        data['decision_matrix'] = pd.read_csv(output_dir / 'route_type_decision_matrix.csv')
    except FileNotFoundError:
        data['decision_matrix'] = pd.DataFrame()
    
    try:
        with open(output_dir / 'memo_data.json', 'r') as f:
            data['memo_data'] = json.load(f)
    except FileNotFoundError:
        data['memo_data'] = {}
    
    return data


data = load_data()

if data is None:
    st.stop()


# ── Sidebar ──
st.sidebar.markdown("## 🚛 Navigation")
page = st.sidebar.radio(
    "Select View",
    ["📊 Network Overview", "🏭 Hub Risk Scorecard", "🛤️ Corridor Analysis",
     "🤖 ETA Model Performance", "🚚 FTL vs Carting", "📋 Strategy Insights"]
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Network Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Network Overview":
    st.markdown('<h1 class="main-header">Delhivery Network Intelligence</h1>', 
                unsafe_allow_html=True)
    st.markdown("### Real-time Logistics Network Health Dashboard")
    
    hub_df = data['hub_metrics']
    corridor_df = data['corridor_metrics']
    memo = data.get('memo_data', {})
    stats = memo.get('overall_stats', {})
    
    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Facilities", f"{len(hub_df):,}")
    with col2:
        st.metric("Total Corridors", f"{len(corridor_df):,}")
    with col3:
        breach_rate = stats.get('sla_breach_rate', 
                                hub_df['pct_delayed_out'].mean() * 100 if 'pct_delayed_out' in hub_df.columns else 0)
        st.metric("SLA Breach Rate", f"{breach_rate:.1f}%")
    with col4:
        avg_delay = stats.get('avg_delay_ratio', 
                              hub_df['avg_out_delay'].mean() if 'avg_out_delay' in hub_df.columns else 0)
        st.metric("Avg Delay Ratio", f"{avg_delay:.2f}x")
    with col5:
        states = stats.get('states_covered', hub_df['state'].nunique() if 'state' in hub_df.columns else 0)
        st.metric("States Covered", f"{states}")
    
    st.markdown("---")
    
    # Network Graph Image
    col1, col2 = st.columns([2, 1])
    with col1:
        graph_path = PROJECT_ROOT / 'outputs/graphs/network_graph.png'
        if graph_path.exists():
            st.image(str(graph_path), caption="Logistics Network Graph", use_container_width=True)
        else:
            st.info("Network graph image not generated yet. Run the full pipeline.")
    
    with col2:
        st.markdown("### 🔥 Top 5 Bottleneck Hubs")
        if 'bottleneck_rank' in hub_df.columns:
            top5 = hub_df.nsmallest(5, 'bottleneck_rank')[
                ['facility_name', 'state', 'bottleneck_score', 'sla_contribution_pct']
            ].copy()
            top5.columns = ['Facility', 'State', 'Risk Score', 'SLA Impact %']
            top5['Risk Score'] = top5['Risk Score'].round(3)
            top5['SLA Impact %'] = top5['SLA Impact %'].round(2)
            st.dataframe(top5, hide_index=True, use_container_width=True)
    
    # State-level summary
    st.markdown("### 📍 State-Level Delay Analysis")
    if 'state' in hub_df.columns:
        state_agg = hub_df.groupby('state').agg(
            facilities=('facility_code', 'count') if 'facility_code' in hub_df.columns else ('facility_name', 'count'),
            avg_delay=('avg_out_delay', 'mean'),
            avg_breach_rate=('pct_delayed_out', 'mean'),
        ).reset_index()
        state_agg = state_agg[state_agg['facilities'] >= 3].sort_values('avg_breach_rate', ascending=False)
        
        fig = px.bar(
            state_agg.head(15), x='state', y='avg_breach_rate',
            color='avg_delay', color_continuous_scale='RdYlGn_r',
            title='SLA Breach Rate by State',
            labels={'avg_breach_rate': 'Avg Breach Rate', 'state': 'State', 'avg_delay': 'Avg Delay'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Hub Risk Scorecard
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏭 Hub Risk Scorecard":
    st.markdown("## 🏭 Hub Risk Scorecard")
    st.markdown("Identify and prioritize facility upgrades based on structural risk and SLA impact.")
    
    hub_df = data['hub_metrics']
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        if 'state' in hub_df.columns:
            states = ['All'] + sorted(hub_df['state'].dropna().unique().tolist())
            selected_state = st.selectbox("Filter by State", states)
    with col2:
        min_volume = st.slider("Min Trip Volume", 0, int(hub_df['total_volume'].max()), 0)
    with col3:
        top_n = st.slider("Show Top N Hubs", 10, 100, 30)
    
    filtered = hub_df.copy()
    if 'state' in hub_df.columns and selected_state != 'All':
        filtered = filtered[filtered['state'] == selected_state]
    filtered = filtered[filtered['total_volume'] >= min_volume]
    
    if 'bottleneck_rank' in filtered.columns:
        filtered = filtered.nsmallest(top_n, 'bottleneck_rank')
    
    # Display columns
    display_cols = ['facility_name', 'state', 'bottleneck_rank', 'bottleneck_score',
                    'betweenness_centrality', 'pct_delayed_out', 'total_volume',
                    'in_degree', 'out_degree', 'pagerank']
    display_cols = [c for c in display_cols if c in filtered.columns]
    
    st.dataframe(
        filtered[display_cols].style.background_gradient(
            subset=['bottleneck_score'] if 'bottleneck_score' in display_cols else [],
            cmap='RdYlGn_r'
        ),
        hide_index=True, use_container_width=True, height=500
    )
    
    # Scatter plot
    st.markdown("### Structural Importance vs Operational Delay")
    img_path = PROJECT_ROOT / 'outputs/graphs/centrality_vs_delay.png'
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Corridor Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🛤️ Corridor Analysis":
    st.markdown("## 🛤️ Corridor Delay Analysis")
    
    corridor_df = data['corridor_metrics']
    
    if corridor_df.empty:
        st.warning("No corridor data available. Run the pipeline first.")
    else:
        # Top delayed corridors
        st.markdown("### Top Delayed Corridors by Impact")
        
        display_cols = ['source_name', 'dest_name', 'trip_count', 'median_delay_ratio',
                        'pct_delayed', 'breach_impact', 'source_state', 'dest_state']
        display_cols = [c for c in display_cols if c in corridor_df.columns]
        
        sort_col = 'breach_impact' if 'breach_impact' in corridor_df.columns else 'median_delay_ratio'
        top = corridor_df.nlargest(30, sort_col)
        
        st.dataframe(
            top[display_cols].style.background_gradient(
                subset=['median_delay_ratio'] if 'median_delay_ratio' in display_cols else [],
                cmap='RdYlGn_r'
            ),
            hide_index=True, use_container_width=True
        )
        
        # Heatmap
        st.markdown("### Corridor Delay Heatmap")
        img_path = PROJECT_ROOT / 'outputs/graphs/corridor_heatmap.png'
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        
        # SLA breach corridors
        st.markdown("### SLA Breach Corridor Ranking")
        img_path = PROJECT_ROOT / 'outputs/graphs/sla_breach_corridors.png'
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: ETA Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ETA Model Performance":
    st.markdown("## 🤖 ETA Prediction Model Comparison")
    st.markdown("Baseline (trip features only) vs Graph-Enhanced (+ centrality + corridor history)")
    
    model_df = data['model_comparison']
    
    if model_df.empty:
        st.warning("No model results available. Run the pipeline first.")
    else:
        # Metrics comparison
        col1, col2, col3 = st.columns(3)
        
        for i, (_, row) in enumerate(model_df.iterrows()):
            with [col1, col2, col3][i % 3]:
                st.markdown(f"### {row['model']}")
                st.metric("MAE", f"{row['MAE']:.2f} min")
                st.metric("15%-Accuracy", f"{row['Accuracy_15pct']:.1f}%")
                st.metric("RMSE", f"{row['RMSE']:.2f} min")
        
        st.markdown("---")
        
        # Graph advantage
        if len(model_df) >= 2:
            baseline = model_df[model_df['model'] == 'baseline'].iloc[0]
            graph = model_df[model_df['model'] == 'graph_enhanced'].iloc[0]
            
            mae_imp = baseline['MAE'] - graph['MAE']
            acc_imp = graph['Accuracy_15pct'] - baseline['Accuracy_15pct']
            
            st.markdown("### 📈 Graph Advantage")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("MAE Reduction", f"{mae_imp:.2f} min",
                          delta=f"{mae_imp/baseline['MAE']*100:.1f}% improvement")
            with col2:
                st.metric("15%-Accuracy Gain", f"+{acc_imp:.1f} pp",
                          delta=f"{acc_imp:.1f} percentage points")
        
        # Comparison charts
        st.markdown("### Model Comparison Charts")
        img_path = PROJECT_ROOT / 'outputs/graphs/model_comparison.png'
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        
        st.markdown("### Actual vs Predicted")
        img_path = PROJECT_ROOT / 'outputs/graphs/actual_vs_predicted.png'
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        
        st.markdown("### Feature Importance")
        img_path = PROJECT_ROOT / 'outputs/graphs/feature_importance.png'
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: FTL vs Carting
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚚 FTL vs Carting":
    st.markdown("## 🚚 FTL vs Carting Decision Framework")
    
    ftl_df = data['ftl_comparison']
    decision_df = data['decision_matrix']
    
    if ftl_df.empty:
        st.warning("No FTL comparison data available.")
    else:
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            ftl_better = (ftl_df['better_route'] == 'FTL').mean() * 100
            st.metric("Corridors Where FTL Wins", f"{ftl_better:.1f}%")
        with col2:
            avg_savings = ftl_df['time_saved_by_ftl'].mean()
            st.metric("Avg Time Saved by FTL", f"{avg_savings:.1f} min")
        with col3:
            st.metric("Dual-Type Corridors", f"{len(ftl_df):,}")
        
        # Decision matrix
        if not decision_df.empty:
            st.markdown("### Decision Matrix by Distance")
            st.dataframe(decision_df, hide_index=True, use_container_width=True)
        
        # Corridor lookup
        st.markdown("### 🔍 Corridor Recommendation Lookup")
        if 'source_center' in ftl_df.columns:
            sources = sorted(ftl_df['source_center'].unique())
            selected_src = st.selectbox("Source Facility", sources)
            
            subset = ftl_df[ftl_df['source_center'] == selected_src]
            if not subset.empty:
                display_cols = ['destination_center', 'better_route', 'delay_diff',
                                'time_saved_by_ftl', 'ftl_delay', 'carting_delay',
                                'ftl_trips', 'carting_trips']
                display_cols = [c for c in display_cols if c in subset.columns]
                st.dataframe(subset[display_cols], hide_index=True, use_container_width=True)
        
        # FTL vs Carting plots
        st.markdown("### Analysis Charts")
        img_path = PROJECT_ROOT / 'outputs/graphs/ftl_vs_carting.png'
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: Strategy Insights
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Strategy Insights":
    st.markdown("## 📋 Strategy Insights for Network Operations")
    
    memo = data.get('memo_data', {})
    hub_df = data['hub_metrics']
    
    if not memo:
        st.warning("Run the full pipeline to generate strategy insights.")
    else:
        # Top 5 Bottleneck Hubs
        st.markdown("### 🔥 Top 5 Bottleneck Hubs — Priority Upgrade Targets")
        if 'top_5_hubs' in memo:
            for i, hub in enumerate(memo['top_5_hubs'], 1):
                with st.expander(f"#{i}: {hub.get('facility_name', 'Unknown')} — {hub.get('state', '')}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Bottleneck Score", f"{hub.get('bottleneck_score', 0):.3f}")
                    with col2:
                        st.metric("SLA Breach Contribution", 
                                  f"{hub.get('sla_contribution_pct', 0):.2f}%")
                    with col3:
                        st.metric("Trip Volume", f"{hub.get('total_volume', 0):,}")
                    
                    st.markdown(f"**Betweenness Centrality**: {hub.get('betweenness_centrality', 0):.4f}")
                    st.markdown(f"**Outgoing Delay Rate**: {hub.get('pct_delayed_out', 0)*100:.1f}%")
        
        # Revenue Impact
        st.markdown("### 💰 Revenue Impact Estimation")
        stats = memo.get('overall_stats', {})
        total_segments = stats.get('total_segments', 0)
        breach_rate = stats.get('sla_breach_rate', 0) / 100
        total_breaches = int(total_segments * breach_rate)
        
        col1, col2 = st.columns(2)
        with col1:
            cost_per_breach = st.slider("Estimated Cost per SLA Breach (₹)", 50, 500, 150)
        with col2:
            hub_improvement = st.slider("Expected Improvement from Hub Upgrade (%)", 10, 50, 25)
        
        revenue_at_risk = total_breaches * cost_per_breach
        recoverable = revenue_at_risk * (hub_improvement / 100)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total SLA Breaches", f"{total_breaches:,}")
        with col2:
            st.metric("Revenue at Risk", f"₹{revenue_at_risk:,.0f}")
        with col3:
            st.metric("Recoverable Revenue", f"₹{recoverable:,.0f}",
                      delta=f"{hub_improvement}% from top hub upgrades")


# ── Footer ──
st.sidebar.markdown("---")
st.sidebar.markdown("Built for Delhivery Network Operations")
st.sidebar.markdown("*Graph-Based Intelligence System*")
