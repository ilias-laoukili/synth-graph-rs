# synth-graph-rs

Rust library for generating synthetic graphs (SBM, DC-SBM, cSBM) with Python bindings via PyO3. Built for GNN research — it generates both graph structure and node features together.

Two output modes:
- **Direct numpy arrays** in PyG format (`edge_index [2, E]`, feature matrix `[N, F]`)
- **JSON** following the team contract (metadata / nodes / edges), with a `json_to_pyg()` helper

## Prerequisites

```bash
# Rust toolchain (>= 1.70)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Python (>= 3.9) with pip
python3 --version   # should be 3.9+

# maturin — the PyO3 build tool
pip3 install maturin

# Python dependencies
pip3 install numpy
```

## 1. Build the Rust extension (development mode)

```bash
# In the project root (synth-graph-rs/)
PYO3_PYTHON=$(which python3) maturin develop

# release build (faster)
PYO3_PYTHON=$(which python3) maturin develop --release
```

This compiles the Rust code and installs `synth_graph_rs` into your current Python environment.

### If you use a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install maturin numpy
maturin develop --release     # PYO3_PYTHON not needed inside venv
```

## 2. Use the library from Python

### Direct-to-PyG API

```python
import numpy as np
import synth_graph_rs as sgr

# SBM
edge_index, labels, h_eff = sgr.generate_sbm(
    n_nodes=1000,
    n_communities=5,
    p_in=0.15,
    p_out=0.01,
    # theta_exponent=2.5,   # uncomment for DC-SBM (power-law degrees, γ > 2)
    seed=42,
)
print(edge_index.shape)  # (2, E)  — bidirectional edge list
print(labels.shape)      # (1000,) — community per node
print(f"Effective homophily: {h_eff:.3f}")

# cSBM with node features
edge_index, x, labels, h_eff = sgr.generate_csbm(
    n_nodes=1000,
    n_communities=5,
    homophily=0.8,      # homophily in [0, 1]
    avg_degree=10.0,    # average degree
    feat_dim=16,        # feature dimensionality
    mu=1.0,             # signal-to-noise ratio
    # feature_dist="gaussian",   # "gaussian" | "uniform" | "laplacian"
    # class_weights=None,        # list of n_communities weights summing to 1.0
    # theta_exponent=2.5,        # DC-SBM power-law degrees (γ > 2)
    # p_triangle=0.1,            # triangle-closing probability per edge
    # feat_noise_ratio=0.1,      # fraction of nodes with wrong-class features
    # ensure_connected=True,     # stitch disconnected components
    seed=42,
)
print(edge_index.shape)  # (2, E)
print(x.shape)           # (1000, 16)
print(labels.shape)      # (1000,)
```

### JSON output API (roadmap contract)

```python
import synth_graph_rs as sgr

json_str = sgr.generate_csbm_json(
    n_nodes=500, n_communities=3, homophily=0.8, avg_degree=8.0, feat_dim=10, mu=1.0, seed=42,
)
# or write directly to disk
sgr.generate_csbm_json_to_file(
    path="graph.json", n_nodes=500, n_communities=3, homophily=0.8, avg_degree=8.0, feat_dim=10, mu=1.0, seed=42,
)

# JSON → PyG arrays
edge_index, labels, features = sgr.json_to_pyg(json_str)
print(edge_index.shape)  # (2, 2E) — bidirectional
print(features.shape)    # (500, 10)
```

### Config-driven API (for Wilfried's TUI)

```python
import json, synth_graph_rs as sgr

config = json.dumps({
    "model_type": "csbm",          # "sbm_classique" | "sbm" | "csbm"
    "seed": 42,
    "parameters": {
        "n_nodes": 1000,
        "n_communities": 5,
        "homophily": 0.8,
        "avg_degree": 10.0,
        "features_dim": 16,
        "mu": 1.0,
    }
})
json_output = sgr.generate_from_config(config)
```

### JSON output format

```json
{
  "metadata": { "n_nodes": 100, "n_edges": 450, "homophily": 0.85 },
  "nodes": [
    { "id": 0, "community": 1, "features": [0.5, 3.4] }
  ],
  "edges": [
    { "source": 0, "target": 1 }
  ]
}
```

- **edges** are canonical (each pair once, undirected). `json_to_pyg()` doubles them.
- **features** is present only for cSBM; omitted for SBM.

## 3. Building a distributable wheel

```bash
maturin build --release
# → target/wheels/synth_graph_rs-0.1.0-*.whl
pip install target/wheels/synth_graph_rs-0.1.0-*.whl
```

## Project structure

```
synth-graph-rs/
├── Cargo.toml          ← Rust manifest & dependencies
├── pyproject.toml      ← maturin / Python packaging config
├── src/
│   └── lib.rs          ← Rust implementation (PyO3 bindings)
└── README.md           ← this file
```

## API reference

### `generate_sbm`

```
generate_sbm(n_nodes, n_communities, p_in, p_out,
             theta_exponent=None, seed=None)
→ (edge_index: i64[2,E], labels: i64[N], effective_h: float)
```

| Parameter | Type | Description |
|---|---|---|
| `n_nodes` | `int` | Number of nodes |
| `n_communities` | `int` | Number of blocks |
| `p_in` | `float` | Intra-block edge probability |
| `p_out` | `float` | Inter-block edge probability |
| `theta_exponent` | `float?` | Pareto exponent γ > 2 for DC-SBM degree correction; `None` = standard SBM |
| `seed` | `int?` | RNG seed for reproducibility |

### `generate_csbm`

```
generate_csbm(n_nodes, n_communities, homophily, avg_degree, feat_dim, mu,
              feature_dist="gaussian", class_weights=None,
              theta_exponent=None, p_triangle=0.0,
              feat_noise_ratio=0.0, ensure_connected=False,
              p_in_override=None, p_out_override=None,
              b_matrix=None, seed=None)
→ (edge_index: i64[2,E], x: f32[N,F], labels: i64[N], effective_h: float)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_nodes` | `int` | — | Number of nodes |
| `n_communities` | `int` | — | Number of classes / blocks |
| `homophily` | `float` | — | Homophily ratio in [0, 1] (1 = perfectly homophilic) |
| `avg_degree` | `float` | — | Target average degree |
| `feat_dim` | `int` | — | Feature dimensionality |
| `mu` | `float` | — | Signal-to-noise ratio for node features (≥ 0) |
| `feature_dist` | `str` | `"gaussian"` | Noise distribution: `"gaussian"`, `"uniform"`, or `"laplacian"` |
| `class_weights` | `list[float]?` | `None` | Per-class node fractions (length `n_communities`, sums to 1.0); `None` = balanced |
| `theta_exponent` | `float?` | `None` | Pareto exponent γ > 2 for DC-SBM; `None` = standard SBM |
| `p_triangle` | `float` | `0.0` | Per-edge triangle-closing probability in [0, 1] |
| `feat_noise_ratio` | `float` | `0.0` | Fraction of nodes with wrong-class features in [0, 1] |
| `ensure_connected` | `bool` | `False` | Stitch disconnected components with bridge edges |
| `p_in_override` | `float?` | `None` | Override computed intra-block probability |
| `p_out_override` | `float?` | `None` | Override computed inter-block probability |
| `b_matrix` | `list[list[float]]?` | `None` | Full `n_communities×n_communities` block probability matrix (overrides homophily/avg_degree) |
| `seed` | `int?` | `None` | RNG seed for reproducibility |

### `generate_sbm_json` / `generate_csbm_json`

Same parameters as above, return a JSON string.

### `generate_sbm_json_to_file` / `generate_csbm_json_to_file`

Same parameters plus `path: str`. Write JSON to the given file path.

### `json_to_pyg`

```
json_to_pyg(json_str: str)
→ (edge_index: i64[2,2E], labels: i64[N], features: f32[N,F] | None)
```

### `generate_from_config`

```
generate_from_config(config_json: str) → str
```

Takes the TUI config JSON (`model_type`, `seed`, `parameters`) and returns the graph JSON.

## Troubleshooting

| Error | Fix |
|---|---|
| `python interpreter not found` | Set `PYO3_PYTHON=$(which python3)` before maturin/cargo |
| `maturin: command not found` | `pip install maturin` |
| `module not found: synth_graph_rs` | Run `maturin develop` first |
