# Delhivery Graph-Based Network Intelligence

> Optimizing Delivery ETAs with Graph-Based Network Intelligence

A comprehensive data science solution that models Delhivery's logistics network as a directed graph to produce smarter ETA predictions, identify bottleneck hubs, and generate actionable operational recommendations.

## Project Overview

Delhivery's OSRM routing engine underestimates actual delivery times on a significant fraction of routes. This project builds a **graph-based intelligence system** that:

1. **Models the logistics network as a graph** — facilities as nodes, corridors as edges
2. **Identifies bottleneck hubs** — using centrality metrics and SLA breach analysis
3. **Produces graph-enhanced ETA predictions** — outperforming the baseline by incorporating network structure
4. **Recommends FTL vs Carting decisions** — with quantified time-cost trade-offs
5. **Delivers a strategy memo** — actionable recommendations for the Head of Network Operations

## Project Structure

```
Summer_project/
├── delivery_data.csv              # Raw trip segment data (~55 MB, 144K rows)
├── requirements.txt               # Python dependencies
├── main.py                        # Master pipeline orchestration
├── README.md                      # This file
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py          # Data loading, cleaning, feature extraction
│   ├── graph_builder.py          # NetworkX directed graph construction
│   ├── bottleneck_analysis.py    # Centrality metrics, bottleneck scoring
│   ├── feature_engineering.py    # Baseline + graph-derived + Node2Vec features
│   ├── models.py                 # XGBoost ETA models + benchmarking
│   ├── ftl_vs_carting.py         # Route-type decision framework
│   └── visualizations.py         # Network graph + chart visualizations
├── deliverables/
│   ├── strategy_memo.md          # Operations strategy memo (1-2 pages)
│   └── dashboard.py              # Streamlit interactive dashboard
├── outputs/
│   ├── graphs/                   # All generated visualization PNGs
│   ├── models/                   # Saved XGBoost model artifacts
│   ├── hub_metrics.csv           # Hub-level centrality + bottleneck scores
│   ├── corridor_metrics.csv      # Corridor-level delay analysis
│   ├── model_comparison.csv      # Baseline vs graph-enhanced metrics
│   └── memo_data.json            # Data for strategy memo
└── notebooks/
    └── (analysis notebooks)
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline
```bash
python main.py
```

This runs all 6 phases sequentially:
1. Data Pipeline (load, clean, feature extraction)
2. Graph Construction (NetworkX directed graph)
3. Bottleneck & Corridor Audit (centrality, SLA analysis)
4. Visualizations (7 publication-quality charts)
5. ETA Prediction Models (baseline vs graph-enhanced benchmarking)
6. FTL vs Carting Framework (route-type optimization)

### 3. Launch Dashboard
```bash
streamlit run deliverables/dashboard.py
```

## Methodology

### Graph Construction
- **Nodes**: Unique facility codes with attributes (state, volume, delay history)
- **Edges**: Corridors weighted by median delay ratio, stratified by route type and time-of-day
- **Edge weights**: `median_delay_ratio = median(actual_time / osrm_time)` per corridor

### Bottleneck Identification
- **Betweenness centrality**: Fraction of shortest paths through each hub
- **Composite bottleneck score**: `0.3 × betweenness + 0.4 × delay_rate + 0.3 × volume`
- **SLA breach contribution**: Hub's share of total network breaches

### ETA Prediction Models
| Model | Features |
|-------|----------|
| **Baseline** | OSRM time/distance, route type, time-of-day, cutoff factors |
| **Graph-Enhanced** | Baseline + node centrality + corridor history + reliability scores |
| **Graph+Node2Vec** | Graph-Enhanced + 32-dim Node2Vec embeddings |

### FTL vs Carting Framework
- Corridor-level comparison of delay ratios by route type
- Gradient Boosted Classifier for optimal route-type prediction
- Decision matrix segmented by distance category and time of day

## Key Metrics
- **MAE**: Mean Absolute Error (minutes)
- **15%-Accuracy**: % of predictions within 15% of actual time
- **SLA Breach Rate**: % of trips where `actual_time > 1.2 × OSRM_time`
- **Bottleneck Score**: Composite risk metric [0, 1]

## Technologies
- **Python 3.12** — Core language
- **Pandas / NumPy** — Data processing
- **NetworkX** — Graph construction and analysis
- **XGBoost** — ETA prediction models
- **Node2Vec** — Graph representation learning
- **Matplotlib / Seaborn** — Static visualizations
- **Plotly** — Interactive charts
- **Streamlit** — Interactive dashboard
- **scikit-learn** — Classification and evaluation

## Authors
Data Science Team — Summer Project 2026
