"""
PyTorch Geometric convenience wrappers for synth-graph-rs.

These functions call the compiled Rust extension and wrap the returned numpy
arrays into ``torch_geometric.data.Data`` objects.  The conversion is
zero-copy where possible: ``torch.from_numpy`` shares memory with the numpy
buffer when the array is contiguous and the dtypes match.

Requires:
    pip install torch torch_geometric

Quick start
-----------
>>> from synth_graph_rs.pyg import sbm_to_pyg, csbm_to_pyg
>>> data = sbm_to_pyg(n_nodes=500, n_communities=5, p_in=0.3, p_out=0.02)
>>> data
Data(edge_index=[2, ...], y=[500], num_nodes=500, effective_h=...)
"""

from __future__ import annotations

import json
from typing import Optional, Sequence

import torch
from torch_geometric.data import Data

import synth_graph_rs as _rs


def sbm_to_pyg(
    n_nodes: int,
    n_communities: int,
    p_in: float,
    p_out: float,
    theta_exponent: Optional[float] = None,
    seed: Optional[int] = None,
) -> Data:
    """Generate an SBM or DC-SBM graph and return a PyG ``Data`` object.

    Parameters
    ----------
    n_nodes:
        Total number of nodes.
    n_communities:
        Number of communities (blocks).
    p_in:
        Intra-community edge probability.
    p_out:
        Inter-community edge probability.
    theta_exponent:
        If set, activates DC-SBM: node degrees follow a Pareto distribution
        with shape parameter ``theta_exponent`` (must be > 2).
    seed:
        Optional RNG seed for reproducibility.

    Returns
    -------
    Data
        ``edge_index`` [2, 2E] (bidirectional), ``y`` [N] community labels,
        ``num_nodes``, ``effective_h`` (measured homophily ratio).
    """
    edge_index_np, labels_np, effective_h = _rs.generate_sbm(
        n_nodes, n_communities, p_in, p_out, theta_exponent, seed
    )
    return Data(
        edge_index=torch.from_numpy(edge_index_np.copy()),
        y=torch.from_numpy(labels_np.copy()),
        num_nodes=n_nodes,
        effective_h=float(effective_h),
    )


def csbm_to_pyg(
    n_nodes: int,
    n_communities: int,
    homophily: float,
    avg_degree: float,
    feat_dim: int,
    mu: float,
    feature_dist: str = "gaussian",
    class_weights: Optional[Sequence[float]] = None,
    theta_exponent: Optional[float] = None,
    p_triangle: float = 0.0,
    feat_noise_ratio: float = 0.0,
    ensure_connected: bool = False,
    p_in_override: Optional[float] = None,
    p_out_override: Optional[float] = None,
    b_matrix: Optional[Sequence[float]] = None,
    seed: Optional[int] = None,
) -> Data:
    """Generate a cSBM graph with node features and return a PyG ``Data`` object.

    Parameters
    ----------
    n_nodes:
        Total number of nodes.
    n_communities:
        Number of communities.
    homophily:
        Target homophily ratio in [0, 1].  Used to derive ``p_in``/``p_out``
        unless ``p_in_override`` and ``p_out_override`` are both provided.
    avg_degree:
        Target average node degree.
    feat_dim:
        Dimension of node feature vectors.
    mu:
        Signal-to-noise ratio: centroid scale relative to noise variance.
        ``mu=0`` produces pure noise features; higher values separate classes.
    feature_dist:
        Noise distribution for node features: ``"gaussian"``, ``"uniform"``,
        or ``"laplacian"``.
    class_weights:
        Optional list of length ``n_communities`` that sums to 1.0.  Controls
        the fraction of nodes in each community.  Equal split if ``None``.
    theta_exponent:
        Activates DC-SBM degree correction (Pareto shape, must be > 2).
    p_triangle:
        Probability of closing each existing edge into a triangle.  Increases
        the clustering coefficient.
    feat_noise_ratio:
        Fraction of nodes that receive features from a wrong community.
    ensure_connected:
        If ``True``, bridge-edge stitching guarantees a connected graph.
    p_in_override / p_out_override:
        Manually set ``p_in`` and ``p_out`` instead of deriving them from
        ``homophily`` and ``avg_degree``.  Both must be provided together.
    b_matrix:
        Full ``n_communities × n_communities`` connection-probability matrix
        (row-major, symmetric).  Overrides the homophily-derived block matrix.
    seed:
        Optional RNG seed for reproducibility.

    Returns
    -------
    Data
        ``x`` [N, F] node features, ``edge_index`` [2, 2E] (bidirectional),
        ``y`` [N] labels, ``num_nodes``, ``effective_h``.
    """
    edge_index_np, x_np, labels_np, effective_h = _rs.generate_csbm(
        n_nodes,
        n_communities,
        homophily,
        avg_degree,
        feat_dim,
        mu,
        feature_dist,
        list(class_weights) if class_weights is not None else None,
        theta_exponent,
        p_triangle,
        feat_noise_ratio,
        ensure_connected,
        p_in_override,
        p_out_override,
        list(b_matrix) if b_matrix is not None else None,
        seed,
    )
    return Data(
        x=torch.from_numpy(x_np.copy()),
        edge_index=torch.from_numpy(edge_index_np.copy()),
        y=torch.from_numpy(labels_np.copy()),
        num_nodes=n_nodes,
        effective_h=float(effective_h),
    )


def json_to_pyg_data(json_str: str) -> Data:
    """Parse a roadmap-contract JSON string and return a PyG ``Data`` object.

    This is the integration point between the Rust CLI (which writes a JSON
    file) and a Python training pipeline that needs a ``Data`` object.

    Parameters
    ----------
    json_str:
        JSON string in the roadmap contract format:
        ``{ metadata, nodes: [{id, community, features?}], edges: [{source, target}] }``.

    Returns
    -------
    Data
        ``edge_index`` [2, 2E], ``y`` [N] labels, ``num_nodes``.
        ``x`` [N, F] is present only when the JSON contains node features
        (cSBM graphs).
    """
    edge_index_np, labels_np, x_np = _rs.json_to_pyg(json_str)
    data = Data(
        edge_index=torch.from_numpy(edge_index_np.copy()),
        y=torch.from_numpy(labels_np.copy()),
        num_nodes=int(labels_np.shape[0]),
    )
    if x_np is not None:
        data.x = torch.from_numpy(x_np.copy())
    return data


def json_file_to_pyg_data(path: str) -> Data:
    """Load a graph JSON file produced by the CLI and return a PyG ``Data`` object.

    Parameters
    ----------
    path:
        Path to a JSON file written by ``synth-graph-cli`` or
        ``generate_sbm_json_to_file`` / ``generate_csbm_json_to_file``.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return json_to_pyg_data(fh.read())
