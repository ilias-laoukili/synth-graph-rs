//! High-performance synthetic graph generation for GNN research (SBM / cSBM).
//! Python bindings via PyO3 + maturin.
//!
//! Block-pair sampling could be parallelised with `rayon`, but sequential
//! execution guarantees identical output for the same seed across platforms.

use std::collections::HashSet;
use std::fs;

use ndarray;
use numpy::{PyArray1, PyArray2, ToPyArray as _};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
use rand_distr::{Distribution, Exp, Geometric, Normal, Pareto, Uniform};
use serde::{Deserialize, Serialize};

// Upper-triangle index decode (used by geometric skip sampling)
//
// Maps a linear index flat_idx (0-based) in the upper triangle of an n×n
// matrix to the pair (row, col) with 0 ≤ row < col < matrix_size.
//
// Row `row` contains (matrix_size-1-row) entries. Cumulative count before row:
//   cumul(row) = row*(2*matrix_size-1-row)/2
//
// We binary-search for the largest row s.t. cumul(row) ≤ flat_idx.

/// Maps a flat upper-triangle index to `(row, col)` with `row < col < matrix_size`.
fn decode_upper_tri(flat_idx: u64, matrix_size: usize) -> (usize, usize) {
    debug_assert!(matrix_size >= 2);
    let size64 = matrix_size as u64;
    let cumul = |row: u64| row * (2 * size64 - 1 - row) / 2;

    // Binary search: largest row in [0, matrix_size-2] s.t. cumul(row) <= flat_idx
    let mut lo = 0u64;
    let mut hi = size64 - 2; // last valid row (has exactly 1 entry)
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if cumul(mid) <= flat_idx {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let row = lo as usize;
    let col = row + 1 + (flat_idx - cumul(lo)) as usize;
    (row, col)
}

// Geometric skip sampling helpers
//
// For a Bernoulli(p) process over `total` positions the gap between consecutive
// successes is Geometric(p) — number of failures before the next success.
// This turns O(N²) trial-by-trial work into O(E) where E is the output size.
//
// For DC-SBM we use p_max = p_base * max_θ_a * max_θ_b as the proposal and
// accept each candidate with probability p_actual/p_max (rejection sampling).

/// Samples intra-block edges using geometric skip sampling.
fn sample_intra_block(
    nodes: &[usize],
    theta: &[f64],
    p_base: f64,
    p_max: f64, // clipped to [0, 1]
    use_dc: bool,
    rng: &mut StdRng,
    edges: &mut Vec<(i64, i64)>,
) {
    let block_size = nodes.len();
    if block_size < 2 || p_max <= 0.0 {
        return;
    }
    let total = (block_size as u64) * (block_size as u64 - 1) / 2;
    let p_proposal = p_max.min(1.0);
    let geom = Geometric::new(p_proposal).expect("p_proposal in (0,1]");
    let uniform = Uniform::new(0.0f64, 1.0);

    let mut pair_idx: u64 = geom.sample(rng);
    while pair_idx < total {
        let (row, col) = decode_upper_tri(pair_idx, block_size);
        let node_u = nodes[row];
        let node_v = nodes[col];

        let accept = if use_dc {
            let p_actual = (p_base * theta[node_u] * theta[node_v]).min(1.0);
            p_actual >= p_proposal || uniform.sample(rng) < p_actual / p_proposal
        } else {
            true // p_proposal == p_base, always accept
        };

        if accept {
            edges.push((node_u as i64, node_v as i64));
        }

        let gap: u64 = geom.sample(rng);
        pair_idx = match pair_idx.checked_add(gap + 1) {
            Some(next) => next,
            None => break,
        };
    }
}

/// Samples inter-block edges between two distinct blocks using geometric skip sampling.
fn sample_inter_block(
    nodes_a: &[usize],
    nodes_b: &[usize],
    theta: &[f64],
    p_base: f64,
    p_max: f64,
    use_dc: bool,
    rng: &mut StdRng,
    edges: &mut Vec<(i64, i64)>,
) {
    let count_a = nodes_a.len() as u64;
    let count_b = nodes_b.len() as u64;
    let total = count_a * count_b;
    if total == 0 || p_max <= 0.0 {
        return;
    }
    let p_proposal = p_max.min(1.0);
    let geom = Geometric::new(p_proposal).expect("p_proposal in (0,1]");
    let uniform = Uniform::new(0.0f64, 1.0);

    let mut pair_idx: u64 = geom.sample(rng);
    while pair_idx < total {
        let row = (pair_idx / count_b) as usize;
        let col = (pair_idx % count_b) as usize;
        let node_u = nodes_a[row];
        let node_v = nodes_b[col];

        let accept = if use_dc {
            let p_actual = (p_base * theta[node_u] * theta[node_v]).min(1.0);
            p_actual >= p_proposal || uniform.sample(rng) < p_actual / p_proposal
        } else {
            true
        };

        if accept {
            edges.push((node_u as i64, node_v as i64));
        }

        let gap: u64 = geom.sample(rng);
        pair_idx = match pair_idx.checked_add(gap + 1) {
            Some(next) => next,
            None => break,
        };
    }
}

/// Expands canonical undirected edges to a bidirectional flat `[2, 2E]` array for PyG.
///
/// Each canonical edge `(src, dst)` produces `src→dst` and `dst→src`.
/// Row 0 holds all sources, row 1 all targets (column-major within the row).
fn build_bidirectional_edge_flat(canonical: &[(i64, i64)]) -> (Vec<i64>, usize) {
    let n_canonical = canonical.len();
    let total_directed = 2 * n_canonical;
    let mut flat = vec![0i64; 2 * total_directed];
    for (edge_idx, &(src, dst)) in canonical.iter().enumerate() {
        flat[edge_idx] = src;
        flat[n_canonical + edge_idx] = dst;
        flat[total_directed + edge_idx] = dst;
        flat[total_directed + n_canonical + edge_idx] = src;
    }
    (flat, total_directed)
}

// JSON output structs — roadmap contract: metadata / nodes / edges
//
// These four types are `pub` so that `synth-graph-cli` and the visualisation
// module can deserialise a graph JSON directly without a Python round-trip.

/// Graph-level statistics serialised in the JSON metadata block.
#[derive(Serialize, Deserialize, Clone)]
pub struct Metadata {
    pub n_nodes: usize,
    pub n_edges: usize,
    pub homophily: f64,
}

/// Node entry in the JSON output.
#[derive(Serialize, Deserialize, Clone)]
pub struct NodeJson {
    pub id: usize,
    pub community: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub features: Option<Vec<f32>>,
}

/// Canonical undirected edge in the JSON output (source < target).
#[derive(Serialize, Deserialize, Clone)]
pub struct EdgeJson {
    pub source: usize,
    pub target: usize,
}

/// Root structure of the JSON output format.
///
/// Matches the roadmap contract:
/// `{ metadata, nodes: [{ id, community, features? }], edges: [{ source, target }] }`
#[derive(Serialize, Deserialize, Clone)]
pub struct GraphOutput {
    pub metadata: Metadata,
    pub nodes: Vec<NodeJson>,
    pub edges: Vec<EdgeJson>,
}

// Core result types (shared by PyG and JSON output paths)

/// Output of SBM / DC-SBM generation.
#[derive(Debug)]
struct SbmCoreResult {
    canonical_edges: Vec<(i64, i64)>,
    labels: Vec<usize>,
    effective_h: f64,
    n_nodes: usize,
}

/// Output of cSBM generation.
struct CsbmCoreResult {
    canonical_edges: Vec<(i64, i64)>,
    labels: Vec<usize>,
    /// Row-major node features: `x_flat[node_idx*feat_dim..(node_idx+1)*feat_dim]` is node `node_idx`.
    x_flat: Vec<f32>,
    effective_h: f64,
    n_nodes: usize,
    feat_dim: usize,
}

// SBM core

/// Generates an SBM or DC-SBM graph; returns a pure-Rust result with no Python types.
fn generate_sbm_core(
    n_nodes: usize,
    n_communities: usize,
    p_in: f64,
    p_out: f64,
    theta_exponent: Option<f64>,
    seed: Option<u64>,
) -> Result<SbmCoreResult, String> {
    if n_communities == 0 || n_nodes == 0 {
        return Err("n_nodes and n_communities must be > 0".into());
    }
    if n_communities > n_nodes {
        return Err(format!(
            "n_communities ({n_communities}) must be ≤ n_nodes ({n_nodes})"
        ));
    }
    if !(0.0..=1.0).contains(&p_in) || !(0.0..=1.0).contains(&p_out) {
        return Err("p_in and p_out must be in [0, 1]".into());
    }
    if let Some(gamma) = theta_exponent {
        if !gamma.is_finite() {
            return Err("theta_exponent must be finite".into());
        }
        if gamma <= 2.0 {
            return Err("theta_exponent (γ) must be > 2 (finite-mean Pareto requirement)".into());
        }
    }

    let mut rng: StdRng = match seed {
        Some(seed_value) => SeedableRng::seed_from_u64(seed_value),
        None => SeedableRng::from_entropy(),
    };

    let labels: Vec<usize> =
        (0..n_nodes).map(|node_idx| node_idx * n_communities / n_nodes).collect();
    let mut block_nodes: Vec<Vec<usize>> = vec![Vec::new(); n_communities];
    for (node_idx, &community) in labels.iter().enumerate() {
        block_nodes[community].push(node_idx);
    }

    let use_dc = theta_exponent.is_some();
    let theta: Vec<f64> = match theta_exponent {
        None => vec![1.0f64; n_nodes],
        Some(gamma) => {
            let pareto = Pareto::new(1.0, gamma - 1.0)
                .map_err(|err| format!("Pareto dist error: {err}"))?;
            let raw: Vec<f64> =
                (0..n_nodes).map(|_| pareto.sample(&mut rng)).collect();
            for (community_idx, community_nodes) in block_nodes.iter().enumerate() {
                if community_nodes.is_empty() {
                    return Err(format!(
                        "block {community_idx} has no nodes; cannot normalise DC-SBM theta \
                         (ensure n_communities ≤ n_nodes)"
                    ));
                }
            }
            let mut block_sum = vec![0.0f64; n_communities];
            for (node_idx, &community) in labels.iter().enumerate() {
                block_sum[community] += raw[node_idx];
            }
            let block_mean: Vec<f64> = (0..n_communities)
                .map(|community_idx| {
                    block_sum[community_idx] / block_nodes[community_idx].len() as f64
                })
                .collect();
            raw.iter()
                .enumerate()
                .map(|(node_idx, &raw_theta)| raw_theta / block_mean[labels[node_idx]])
                .collect()
        }
    };

    let block_max_theta: Vec<f64> = if use_dc {
        (0..n_communities)
            .map(|community_idx| {
                block_nodes[community_idx]
                    .iter()
                    .map(|&node_idx| theta[node_idx])
                    .fold(0.0f64, f64::max)
            })
            .collect()
    } else {
        vec![1.0f64; n_communities]
    };

    let mut edges: Vec<(i64, i64)> = Vec::new();
    for block_a in 0..n_communities {
        for block_b in block_a..n_communities {
            let p_base = if block_a == block_b { p_in } else { p_out };
            if p_base == 0.0 {
                continue;
            }
            let p_max = if use_dc {
                (p_base * block_max_theta[block_a] * block_max_theta[block_b]).min(1.0)
            } else {
                p_base
            };
            if p_max <= 0.0 {
                continue;
            }
            if block_a == block_b {
                sample_intra_block(
                    &block_nodes[block_a], &theta, p_base, p_max, use_dc, &mut rng, &mut edges,
                );
            } else {
                sample_inter_block(
                    &block_nodes[block_a], &block_nodes[block_b], &theta, p_base, p_max,
                    use_dc, &mut rng, &mut edges,
                );
            }
        }
    }

    let homo = edges
        .iter()
        .filter(|&&(src, dst)| labels[src as usize] == labels[dst as usize])
        .count();
    let effective_h =
        if edges.is_empty() { 0.0f64 } else { homo as f64 / edges.len() as f64 };

    Ok(SbmCoreResult { canonical_edges: edges, labels, effective_h, n_nodes })
}

/// SBM/DC-SBM: returns `(edge_index [2,2E], labels [N], effective_h)`.
#[pyfunction]
#[pyo3(signature = (n_nodes, n_communities, p_in, p_out, theta_exponent=None, seed=None))]
fn generate_sbm<'py>(
    py: Python<'py>,
    n_nodes: usize,
    n_communities: usize,
    p_in: f64,
    p_out: f64,
    theta_exponent: Option<f64>,
    seed: Option<u64>,
) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray1<i64>>, f64)> {
    let result = py.allow_threads(|| {
        generate_sbm_core(n_nodes, n_communities, p_in, p_out, theta_exponent, seed)
    }).map_err(PyValueError::new_err)?;

    let (flat, total_directed) = build_bidirectional_edge_flat(&result.canonical_edges);
    let edge_arr = ndarray::Array2::from_shape_vec((2, total_directed), flat)
        .map_err(|err| PyValueError::new_err(format!("shape error: {err}")))?;
    let label_arr =
        ndarray::Array1::from(result.labels.iter().map(|&lbl| lbl as i64).collect::<Vec<_>>());

    Ok((edge_arr.to_pyarray_bound(py), label_arr.to_pyarray_bound(py), result.effective_h))
}

/// Node feature noise distribution for cSBM generation.
enum FeatDist {
    /// Zero-mean unit-variance Gaussian.
    Gaussian,
    /// Uniform on `[-√3, √3]` (unit variance).
    Uniform,
    /// Laplacian with unit variance (scale = 1/√2).
    Laplacian,
}

// cSBM core

/// Generates a cSBM graph with node features; returns a pure-Rust result with no Python types.
#[allow(clippy::too_many_arguments)]
fn generate_csbm_core(
    n_nodes: usize,
    n_communities: usize,
    homophily: f64,
    avg_degree: f64,
    feat_dim: usize,
    mu: f64,
    feature_dist: &str,
    class_weights: Option<&[f64]>,
    theta_exponent: Option<f64>,
    p_triangle: f64,
    feat_noise_ratio: f64,
    ensure_connected: bool,
    p_in_override: Option<f64>,
    p_out_override: Option<f64>,
    b_matrix: Option<&[f64]>,
    seed: Option<u64>,
) -> Result<CsbmCoreResult, String> {
    if n_nodes == 0 || n_communities == 0 {
        return Err("n_nodes and n_communities must be > 0".into());
    }
    if n_communities > n_nodes {
        return Err("n_communities must be ≤ n_nodes".into());
    }
    if !(0.0..=1.0).contains(&homophily) {
        return Err("homophily must be in [0, 1]".into());
    }
    if !avg_degree.is_finite() || avg_degree < 0.0 {
        return Err("avg_degree must be finite and ≥ 0".into());
    }
    if feat_dim == 0 {
        return Err("feat_dim must be > 0".into());
    }
    if !mu.is_finite() || mu < 0.0 {
        return Err("mu must be finite and ≥ 0".into());
    }
    let dist_enum = match feature_dist {
        "gaussian" => FeatDist::Gaussian,
        "uniform" => FeatDist::Uniform,
        "laplacian" => FeatDist::Laplacian,
        other => {
            return Err(format!(
                "feature_dist must be 'gaussian', 'uniform', or 'laplacian'; got '{other}'"
            ))
        }
    };
    if let Some(gamma) = theta_exponent {
        if !gamma.is_finite() {
            return Err("theta_exponent must be finite".into());
        }
        if gamma <= 2.0 {
            return Err("theta_exponent (γ) must be > 2 (finite-mean Pareto requirement)".into());
        }
    }
    if !(0.0..=1.0).contains(&p_triangle) {
        return Err("p_triangle must be in [0, 1]".into());
    }
    if !(0.0..=1.0).contains(&feat_noise_ratio) {
        return Err("feat_noise_ratio must be in [0, 1]".into());
    }

    match (p_in_override, p_out_override) {
        (Some(p_in), Some(p_out)) => {
            if !(0.0..=1.0).contains(&p_in) || !(0.0..=1.0).contains(&p_out) {
                return Err("p_in_override and p_out_override must be in [0, 1]".into());
            }
        }
        (None, None) => {}
        _ => {
            return Err(
                "p_in_override and p_out_override must both be set or both be None".into(),
            )
        }
    }

    if let Some(bm) = b_matrix {
        if bm.len() != n_communities * n_communities {
            return Err(format!(
                "b_matrix must have length n_communities²={}, got {}",
                n_communities * n_communities,
                bm.len()
            ));
        }
        for &prob in bm {
            if !(0.0..=1.0).contains(&prob) {
                return Err("b_matrix values must be in [0, 1]".into());
            }
        }
        for row in 0..n_communities {
            for col in 0..n_communities {
                if (bm[row * n_communities + col] - bm[col * n_communities + row]).abs() > 1e-9 {
                    return Err(format!(
                        "b_matrix must be symmetric: B[{row},{col}]={} != B[{col},{row}]={}",
                        bm[row * n_communities + col],
                        bm[col * n_communities + row]
                    ));
                }
            }
        }
    }

    // block sizes
    let block_sizes: Vec<usize> = if let Some(weights) = class_weights {
        if weights.len() != n_communities {
            return Err(format!(
                "class_weights length ({}) must equal n_communities ({})",
                weights.len(),
                n_communities
            ));
        }
        if weights.iter().any(|&weight| !weight.is_finite() || weight < 0.0) {
            return Err("class_weights values must all be finite and ≥ 0".into());
        }
        let total_weight: f64 = weights.iter().sum();
        if (total_weight - 1.0).abs() > 1e-6 {
            return Err(format!("class_weights must sum to 1.0, got {total_weight:.6}"));
        }
        let exact: Vec<f64> = weights.iter().map(|&weight| weight * n_nodes as f64).collect();
        let mut sizes: Vec<usize> = exact.iter().map(|&exact_size| exact_size.floor() as usize).collect();
        let remainder: usize = n_nodes - sizes.iter().sum::<usize>();
        let mut fracs: Vec<(usize, f64)> = exact
            .iter()
            .enumerate()
            .map(|(frac_idx, &exact_size)| (frac_idx, exact_size - exact_size.floor()))
            .collect();
        fracs.sort_unstable_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1));
        for remainder_idx in 0..remainder {
            sizes[fracs[remainder_idx].0] += 1;
        }
        if sizes.iter().any(|&sz| sz == 0) {
            return Err(
                "some blocks have zero nodes; increase n_nodes or adjust class_weights".into(),
            );
        }
        sizes
    } else {
        let base_size = n_nodes / n_communities;
        let remainder = n_nodes % n_communities;
        (0..n_communities)
            .map(|community_idx| base_size + usize::from(community_idx < remainder))
            .collect()
    };

    let mut labels = vec![0usize; n_nodes];
    {
        let mut offset = 0usize;
        for (community_idx, &community_size) in block_sizes.iter().enumerate() {
            for node_idx in offset..offset + community_size {
                labels[node_idx] = community_idx;
            }
            offset += community_size;
        }
    }

    let mut block_nodes: Vec<Vec<usize>> = vec![Vec::new(); n_communities];
    for (node_idx, &community) in labels.iter().enumerate() {
        block_nodes[community].push(node_idx);
    }

    let (p_in, p_out) = if let (Some(p_in), Some(p_out)) = (p_in_override, p_out_override) {
        (p_in, p_out)
    } else {
        // Generalised formula supporting unequal block sizes.
        // p_in  = homophily·avg_degree·n / Σ_k s_k(s_k−1)
        // p_out = (1−homophily)·avg_degree·n / (n² − Σ_k s_k²)
        let sum_sk_sk1: f64 = block_sizes
            .iter()
            .map(|&block_size| (block_size as f64) * (block_size as f64 - 1.0))
            .sum();
        let sum_sk2: f64 = block_sizes
            .iter()
            .map(|&block_size| (block_size as f64) * (block_size as f64))
            .sum();
        let n_float = n_nodes as f64;
        let cross_pairs = n_float * n_float - sum_sk2; // = 2 · Σ_{a<b} s_a·s_b
        let p_in = if sum_sk_sk1 > 0.0 {
            (homophily * avg_degree * n_float / sum_sk_sk1).min(1.0)
        } else {
            0.0
        };
        let p_out = if cross_pairs > 0.0 {
            ((1.0 - homophily) * avg_degree * n_float / cross_pairs).min(1.0)
        } else {
            0.0
        };
        (p_in, p_out)
    };

    let edge_probs: Vec<f64> = if let Some(bm) = b_matrix {
        bm.to_vec()
    } else {
        let mut ep = vec![0.0f64; n_communities * n_communities];
        for row in 0..n_communities {
            for col in 0..n_communities {
                ep[row * n_communities + col] = if row == col { p_in } else { p_out };
            }
        }
        ep
    };

    let mut rng: StdRng = match seed {
        Some(seed_value) => SeedableRng::seed_from_u64(seed_value),
        None => SeedableRng::from_entropy(),
    };

    let use_dc = theta_exponent.is_some();
    let theta: Vec<f64> = match theta_exponent {
        None => vec![1.0f64; n_nodes],
        Some(gamma) => {
            let pareto = Pareto::new(1.0, gamma - 1.0)
                .map_err(|err| format!("Pareto dist error: {err}"))?;
            let raw: Vec<f64> = (0..n_nodes).map(|_| pareto.sample(&mut rng)).collect();
            let mut block_sum = vec![0.0f64; n_communities];
            for (node_idx, &community) in labels.iter().enumerate() {
                block_sum[community] += raw[node_idx];
            }
            let block_mean: Vec<f64> = (0..n_communities)
                .map(|community_idx| {
                    block_sum[community_idx] / block_nodes[community_idx].len() as f64
                })
                .collect();
            raw.iter()
                .enumerate()
                .map(|(node_idx, &raw_theta)| raw_theta / block_mean[labels[node_idx]])
                .collect()
        }
    };

    let block_max_theta: Vec<f64> = if use_dc {
        (0..n_communities)
            .map(|community_idx| {
                block_nodes[community_idx]
                    .iter()
                    .map(|&node_idx| theta[node_idx])
                    .fold(0.0f64, f64::max)
            })
            .collect()
    } else {
        vec![1.0f64; n_communities]
    };

    let need_adj = p_triangle > 0.0 || ensure_connected;

    let mut edges: Vec<(i64, i64)> = Vec::new();
    for block_a in 0..n_communities {
        for block_b in block_a..n_communities {
            let p_base = edge_probs[block_a * n_communities + block_b];
            if p_base == 0.0 {
                continue;
            }
            let p_max = if use_dc {
                (p_base * block_max_theta[block_a] * block_max_theta[block_b]).min(1.0)
            } else {
                p_base
            };
            if p_max <= 0.0 {
                continue;
            }
            if block_a == block_b {
                sample_intra_block(
                    &block_nodes[block_a], &theta, p_base, p_max, use_dc, &mut rng, &mut edges,
                );
            } else {
                sample_inter_block(
                    &block_nodes[block_a], &block_nodes[block_b], &theta, p_base, p_max,
                    use_dc, &mut rng, &mut edges,
                );
            }
        }
    }

    // adjacency list (needed for triangle closing / connectivity stitching)
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); if need_adj { n_nodes } else { 0 }];
    let mut edge_set: HashSet<(usize, usize)> = HashSet::new();
    if need_adj {
        for &(src, dst) in &edges {
            let src_idx = src as usize;
            let dst_idx = dst as usize;
            let key = (src_idx.min(dst_idx), src_idx.max(dst_idx));
            if edge_set.insert(key) {
                adj[src_idx].push(dst_idx);
                adj[dst_idx].push(src_idx);
            }
        }
    }

    // triangle closing (randomly pick which endpoint is the hub)
    if p_triangle > 0.0 {
        let snapshot: Vec<(usize, usize)> = edge_set.iter().cloned().collect();
        for (endpoint_a, endpoint_b) in snapshot {
            if rng.gen::<f64>() < p_triangle {
                let (hub, other) = if rng.gen::<bool>() {
                    (endpoint_a, endpoint_b)
                } else {
                    (endpoint_b, endpoint_a)
                };
                let hub_neighbors: Vec<usize> =
                    adj[hub].iter().cloned().filter(|&w| w != other).collect();
                if hub_neighbors.is_empty() {
                    continue;
                }
                let witness = hub_neighbors[rng.gen_range(0..hub_neighbors.len())];
                let key = (other.min(witness), other.max(witness));
                if edge_set.insert(key) {
                    edges.push((key.0 as i64, key.1 as i64));
                    adj[other].push(witness);
                    adj[witness].push(other);
                }
            }
        }
    }

    // connectivity stitching
    if ensure_connected && n_nodes > 0 {
        let mut visited = vec![false; n_nodes];
        let mut main_component: Vec<usize> = Vec::new();
        visited[0] = true;
        main_component.push(0);
        let mut head = 0;
        while head < main_component.len() {
            let current = main_component[head];
            head += 1;
            for &neighbor in &adj[current] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    main_component.push(neighbor);
                }
            }
        }
        loop {
            let unvisited_start = match visited.iter().position(|&was_visited| !was_visited) {
                Some(pos) => pos,
                None => break,
            };
            let mut component: Vec<usize> = vec![unvisited_start];
            visited[unvisited_start] = true;
            let mut head2 = 0;
            while head2 < component.len() {
                let current = component[head2];
                head2 += 1;
                for &neighbor in &adj[current] {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        component.push(neighbor);
                    }
                }
            }
            let bridge_src = component[0];
            let bridge_dst = main_component[rng.gen_range(0..main_component.len())];
            let key = (bridge_src.min(bridge_dst), bridge_src.max(bridge_dst));
            if edge_set.insert(key) {
                edges.push((key.0 as i64, key.1 as i64));
                adj[bridge_src].push(bridge_dst);
                adj[bridge_dst].push(bridge_src);
            }
            main_component.extend_from_slice(&component);
        }
    }

    let homo = edges
        .iter()
        .filter(|&&(src, dst)| labels[src as usize] == labels[dst as usize])
        .count();
    let effective_h =
        if edges.is_empty() { 0.0f64 } else { homo as f64 / edges.len() as f64 };

    let centroid_std = (1.0_f64 / feat_dim as f64).sqrt();
    let centroid_dist = Normal::new(0.0_f64, centroid_std).expect("centroid_std > 0");
    let centroids: Vec<f64> =
        (0..n_communities * feat_dim).map(|_| centroid_dist.sample(&mut rng)).collect();

    // Optionally reassign a node to a wrong class for label noise.
    let noisy_class = |true_class: usize, rng: &mut StdRng| -> usize {
        if n_communities > 1 && feat_noise_ratio > 0.0 && rng.gen::<f64>() < feat_noise_ratio {
            let offset = rng.gen_range(1..n_communities);
            (true_class + offset) % n_communities
        } else {
            true_class
        }
    };

    let mut x_flat: Vec<f32> = Vec::with_capacity(n_nodes * feat_dim);
    match dist_enum {
        FeatDist::Gaussian => {
            let noise = Normal::new(0.0_f64, 1.0_f64).expect("valid");
            for node_idx in 0..n_nodes {
                let class_idx = noisy_class(labels[node_idx], &mut rng);
                for feat_idx in 0..feat_dim {
                    x_flat.push(
                        (mu * centroids[class_idx * feat_dim + feat_idx]
                            + noise.sample(&mut rng)) as f32,
                    );
                }
            }
        }
        FeatDist::Uniform => {
            let sqrt3 = 3.0_f64.sqrt();
            let noise = Uniform::new(-sqrt3, sqrt3);
            for node_idx in 0..n_nodes {
                let class_idx = noisy_class(labels[node_idx], &mut rng);
                for feat_idx in 0..feat_dim {
                    x_flat.push(
                        (mu * centroids[class_idx * feat_dim + feat_idx]
                            + noise.sample(&mut rng)) as f32,
                    );
                }
            }
        }
        FeatDist::Laplacian => {
            let scale = 1.0_f64 / 2.0_f64.sqrt();
            let exp_dist = Exp::new(1.0_f64).expect("rate > 0");
            let sign_dist = Uniform::new(0.0_f64, 1.0_f64);
            for node_idx in 0..n_nodes {
                let class_idx = noisy_class(labels[node_idx], &mut rng);
                for feat_idx in 0..feat_dim {
                    let exp_sample = exp_dist.sample(&mut rng);
                    let sign = if sign_dist.sample(&mut rng) < 0.5 { 1.0_f64 } else { -1.0_f64 };
                    x_flat.push(
                        (mu * centroids[class_idx * feat_dim + feat_idx]
                            + scale * exp_sample * sign) as f32,
                    );
                }
            }
        }
    }

    Ok(CsbmCoreResult {
        canonical_edges: edges,
        labels,
        x_flat,
        effective_h,
        n_nodes,
        feat_dim,
    })
}

/// cSBM: returns `(edge_index [2,2E], x [N,F], labels [N], effective_h)`.
#[pyfunction]
#[pyo3(signature = (n_nodes, n_communities, homophily, avg_degree, feat_dim, mu, feature_dist="gaussian", class_weights=None, theta_exponent=None, p_triangle=0.0, feat_noise_ratio=0.0, ensure_connected=false, p_in_override=None, p_out_override=None, b_matrix=None, seed=None))]
fn generate_csbm<'py>(
    py: Python<'py>,
    n_nodes: usize,
    n_communities: usize,
    homophily: f64,
    avg_degree: f64,
    feat_dim: usize,
    mu: f64,
    feature_dist: &str,
    class_weights: Option<Vec<f64>>,
    theta_exponent: Option<f64>,
    p_triangle: f64,
    feat_noise_ratio: f64,
    ensure_connected: bool,
    p_in_override: Option<f64>,
    p_out_override: Option<f64>,
    b_matrix: Option<Vec<f64>>,
    seed: Option<u64>,
) -> PyResult<(
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<i64>>,
    f64,
)> {
    let fd = feature_dist.to_owned();
    let cw = class_weights.clone();
    let bm = b_matrix.clone();
    let result = py.allow_threads(move || {
        generate_csbm_core(
            n_nodes, n_communities, homophily, avg_degree, feat_dim, mu, &fd,
            cw.as_deref(), theta_exponent, p_triangle, feat_noise_ratio,
            ensure_connected, p_in_override, p_out_override, bm.as_deref(), seed,
        )
    }).map_err(PyValueError::new_err)?;

    let (flat, total_directed) = build_bidirectional_edge_flat(&result.canonical_edges);
    let edge_arr = ndarray::Array2::from_shape_vec((2, total_directed), flat)
        .map_err(|err| PyValueError::new_err(format!("edge_index shape error: {err}")))?;

    let x_arr = ndarray::Array2::from_shape_vec((result.n_nodes, result.feat_dim), result.x_flat)
        .map_err(|err| PyValueError::new_err(format!("x shape error: {err}")))?;

    let label_arr =
        ndarray::Array1::from(result.labels.iter().map(|&lbl| lbl as i64).collect::<Vec<_>>());

    Ok((
        edge_arr.to_pyarray_bound(py),
        x_arr.to_pyarray_bound(py),
        label_arr.to_pyarray_bound(py),
        result.effective_h,
    ))
}

// Helpers: core result → GraphOutput

/// Converts an SBM result into the JSON output structure.
fn sbm_result_to_graph_output(result: &SbmCoreResult) -> GraphOutput {
    GraphOutput {
        metadata: Metadata {
            n_nodes: result.n_nodes,
            n_edges: result.canonical_edges.len(),
            homophily: result.effective_h,
        },
        nodes: (0..result.n_nodes)
            .map(|node_idx| NodeJson {
                id: node_idx,
                community: result.labels[node_idx],
                features: None,
            })
            .collect(),
        edges: result
            .canonical_edges
            .iter()
            .map(|&(src, dst)| EdgeJson { source: src as usize, target: dst as usize })
            .collect(),
    }
}

/// Converts a cSBM result into the JSON output structure.
fn csbm_result_to_graph_output(result: &CsbmCoreResult) -> GraphOutput {
    GraphOutput {
        metadata: Metadata {
            n_nodes: result.n_nodes,
            n_edges: result.canonical_edges.len(),
            homophily: result.effective_h,
        },
        nodes: (0..result.n_nodes)
            .map(|node_idx| {
                let feats: Vec<f32> =
                    result.x_flat[node_idx * result.feat_dim..(node_idx + 1) * result.feat_dim]
                        .to_vec();
                NodeJson {
                    id: node_idx,
                    community: result.labels[node_idx],
                    features: Some(feats),
                }
            })
            .collect(),
        edges: result
            .canonical_edges
            .iter()
            .map(|&(src, dst)| EdgeJson { source: src as usize, target: dst as usize })
            .collect(),
    }
}

// JSON output functions

/// SBM JSON output.
#[pyfunction]
#[pyo3(signature = (n_nodes, n_communities, p_in, p_out, theta_exponent=None, seed=None))]
fn generate_sbm_json(
    py: Python<'_>,
    n_nodes: usize,
    n_communities: usize,
    p_in: f64,
    p_out: f64,
    theta_exponent: Option<f64>,
    seed: Option<u64>,
) -> PyResult<String> {
    let result = py.allow_threads(|| {
        generate_sbm_core(n_nodes, n_communities, p_in, p_out, theta_exponent, seed)
    }).map_err(PyValueError::new_err)?;
    let graph = sbm_result_to_graph_output(&result);
    serde_json::to_string_pretty(&graph)
        .map_err(|err| PyValueError::new_err(format!("JSON serialization error: {err}")))
}

/// SBM JSON to file.
#[pyfunction]
#[pyo3(signature = (path, n_nodes, n_communities, p_in, p_out, theta_exponent=None, seed=None))]
fn generate_sbm_json_to_file(
    py: Python<'_>,
    path: &str,
    n_nodes: usize,
    n_communities: usize,
    p_in: f64,
    p_out: f64,
    theta_exponent: Option<f64>,
    seed: Option<u64>,
) -> PyResult<()> {
    let result = py.allow_threads(|| {
        generate_sbm_core(n_nodes, n_communities, p_in, p_out, theta_exponent, seed)
    }).map_err(PyValueError::new_err)?;
    let graph = sbm_result_to_graph_output(&result);
    let json = serde_json::to_string_pretty(&graph)
        .map_err(|err| PyValueError::new_err(format!("JSON serialization error: {err}")))?;
    fs::write(path, json)
        .map_err(|err| PyValueError::new_err(format!("File write error: {err}")))
}

/// cSBM JSON output.
#[pyfunction]
#[pyo3(signature = (n_nodes, n_communities, homophily, avg_degree, feat_dim, mu, feature_dist="gaussian", class_weights=None, theta_exponent=None, p_triangle=0.0, feat_noise_ratio=0.0, ensure_connected=false, p_in_override=None, p_out_override=None, b_matrix=None, seed=None))]
fn generate_csbm_json(
    py: Python<'_>,
    n_nodes: usize,
    n_communities: usize,
    homophily: f64,
    avg_degree: f64,
    feat_dim: usize,
    mu: f64,
    feature_dist: &str,
    class_weights: Option<Vec<f64>>,
    theta_exponent: Option<f64>,
    p_triangle: f64,
    feat_noise_ratio: f64,
    ensure_connected: bool,
    p_in_override: Option<f64>,
    p_out_override: Option<f64>,
    b_matrix: Option<Vec<f64>>,
    seed: Option<u64>,
) -> PyResult<String> {
    let fd = feature_dist.to_owned();
    let cw = class_weights.clone();
    let bm = b_matrix.clone();
    let result = py.allow_threads(move || {
        generate_csbm_core(
            n_nodes, n_communities, homophily, avg_degree, feat_dim, mu, &fd,
            cw.as_deref(), theta_exponent, p_triangle, feat_noise_ratio,
            ensure_connected, p_in_override, p_out_override, bm.as_deref(), seed,
        )
    }).map_err(PyValueError::new_err)?;
    let graph = csbm_result_to_graph_output(&result);
    serde_json::to_string_pretty(&graph)
        .map_err(|err| PyValueError::new_err(format!("JSON serialization error: {err}")))
}

/// cSBM JSON to file.
#[pyfunction]
#[pyo3(signature = (path, n_nodes, n_communities, homophily, avg_degree, feat_dim, mu, feature_dist="gaussian", class_weights=None, theta_exponent=None, p_triangle=0.0, feat_noise_ratio=0.0, ensure_connected=false, p_in_override=None, p_out_override=None, b_matrix=None, seed=None))]
fn generate_csbm_json_to_file(
    py: Python<'_>,
    path: &str,
    n_nodes: usize,
    n_communities: usize,
    homophily: f64,
    avg_degree: f64,
    feat_dim: usize,
    mu: f64,
    feature_dist: &str,
    class_weights: Option<Vec<f64>>,
    theta_exponent: Option<f64>,
    p_triangle: f64,
    feat_noise_ratio: f64,
    ensure_connected: bool,
    p_in_override: Option<f64>,
    p_out_override: Option<f64>,
    b_matrix: Option<Vec<f64>>,
    seed: Option<u64>,
) -> PyResult<()> {
    let fd = feature_dist.to_owned();
    let cw = class_weights.clone();
    let bm = b_matrix.clone();
    let result = py.allow_threads(move || {
        generate_csbm_core(
            n_nodes, n_communities, homophily, avg_degree, feat_dim, mu, &fd,
            cw.as_deref(), theta_exponent, p_triangle, feat_noise_ratio,
            ensure_connected, p_in_override, p_out_override, bm.as_deref(), seed,
        )
    }).map_err(PyValueError::new_err)?;
    let graph = csbm_result_to_graph_output(&result);
    let json = serde_json::to_string_pretty(&graph)
        .map_err(|err| PyValueError::new_err(format!("JSON serialization error: {err}")))?;
    fs::write(path, json)
        .map_err(|err| PyValueError::new_err(format!("File write error: {err}")))
}

/// Collects node features into a flat buffer.
/// Returns `Some((feat_dim, flat))` if all nodes have features, `None` if none do,
/// or an error if the feature vectors are inconsistent.
fn collect_node_features(nodes: &[NodeJson]) -> Result<Option<(usize, Vec<f32>)>, String> {
    let has_any = nodes.iter().any(|node| node.features.is_some());
    if !has_any {
        return Ok(None);
    }
    // At least one node has features — all must.
    let feat_dim = match nodes.iter().find(|node| node.features.is_some()) {
        Some(node) => match node.features.as_ref() {
            Some(feats) => feats.len(),
            None => return Err("internal: feature predicate inconsistency".into()),
        },
        None => {
            return Err("internal: no node with features found despite has_any=true".into())
        }
    };
    let mut flat: Vec<f32> = Vec::with_capacity(nodes.len() * feat_dim);
    for node in nodes {
        match &node.features {
            Some(feats) => {
                if feats.len() != feat_dim {
                    return Err(format!(
                        "node {} has {} features, expected {feat_dim}",
                        node.id,
                        feats.len()
                    ));
                }
                flat.extend_from_slice(feats);
            }
            None => {
                return Err(format!(
                    "node {} is missing features — all nodes must have features or none",
                    node.id
                ));
            }
        }
    }
    Ok(Some((feat_dim, flat)))
}

/// Parses a JSON graph string into PyG numpy arrays.
#[pyfunction]
#[pyo3(signature = (json_str))]
fn json_to_pyg<'py>(
    py: Python<'py>,
    json_str: &str,
) -> PyResult<(
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray1<i64>>,
    Option<Bound<'py, PyArray2<f32>>>,
)> {
    let graph: GraphOutput = serde_json::from_str(json_str)
        .map_err(|err| PyValueError::new_err(format!("JSON parse error: {err}")))?;

    let n_nodes = graph.nodes.len();

    let labels: Vec<i64> = graph.nodes.iter().map(|node| node.community as i64).collect();
    let label_arr = ndarray::Array1::from(labels);

    let canonical: Vec<(i64, i64)> = graph
        .edges
        .iter()
        .map(|edge| (edge.source as i64, edge.target as i64))
        .collect();
    let (flat, total_directed) = build_bidirectional_edge_flat(&canonical);
    let edge_arr = ndarray::Array2::from_shape_vec((2, total_directed), flat)
        .map_err(|err| PyValueError::new_err(format!("edge_index shape error: {err}")))?;

    let x_arr = match collect_node_features(&graph.nodes)
        .map_err(PyValueError::new_err)?
    {
        Some((feat_dim, x_flat)) => {
            let arr = ndarray::Array2::from_shape_vec((n_nodes, feat_dim), x_flat)
                .map_err(|err| PyValueError::new_err(format!("x shape error: {err}")))?;
            Some(arr.to_pyarray_bound(py))
        }
        None => None,
    };

    Ok((edge_arr.to_pyarray_bound(py), label_arr.to_pyarray_bound(py), x_arr))
}

// Config-driven API
//
// Two entry points with identical logic but different type signatures:
//
//   generate_from_config        — #[pyfunction], called from Python
//   generate_from_config_native — pub fn, called from the Rust CLI binary
//
// Both parse the same JSON contract defined in the roadmap and return a
// pretty-printed JSON string on success or an error message on failure.

/// Config-driven generation request parsed from the TUI JSON input.
#[derive(Deserialize)]
struct GenerationConfig {
    model_type: String,
    seed: Option<u64>,
    parameters: ConfigParameters,
}

/// Parameters block within a `GenerationConfig`.
#[derive(Deserialize)]
struct ConfigParameters {
    n_nodes: usize,
    n_communities: usize,
    homophily: f64,
    avg_degree: f64,
    #[serde(default)]
    features_dim: Option<usize>,
    #[serde(default)]
    mu: Option<f64>,
    #[serde(default)]
    theta_exponent: Option<f64>,
    #[serde(default)]
    feat_noise_ratio: Option<f64>,
}

/// Config-driven dispatch. `model_type`: `"sbm_classique"`, `"sbm"`, or `"csbm"`.
#[pyfunction]
#[pyo3(signature = (config_json))]
fn generate_from_config(py: Python<'_>, config_json: &str) -> PyResult<String> {
    let config: GenerationConfig = serde_json::from_str(config_json)
        .map_err(|err| PyValueError::new_err(format!("Config parse error: {err}")))?;

    let params = &config.parameters;
    match config.model_type.as_str() {
        "sbm_classique" | "sbm" => {
            let block_size = params.n_nodes as f64 / params.n_communities as f64;
            let p_in = if block_size > 1.0 {
                (params.homophily * params.avg_degree / (block_size - 1.0)).min(1.0)
            } else {
                0.0
            };
            let p_out = if params.n_communities > 1 {
                ((1.0 - params.homophily) * params.avg_degree
                    / ((params.n_communities - 1) as f64 * block_size))
                    .min(1.0)
            } else {
                0.0
            };
            let theta = params.theta_exponent;
            let seed = config.seed;
            let n_nodes = params.n_nodes;
            let n_communities = params.n_communities;
            let result = py
                .allow_threads(move || {
                    generate_sbm_core(n_nodes, n_communities, p_in, p_out, theta, seed)
                })
                .map_err(PyValueError::new_err)?;
            let graph = sbm_result_to_graph_output(&result);
            serde_json::to_string_pretty(&graph)
                .map_err(|err| PyValueError::new_err(format!("JSON serialization error: {err}")))
        }
        "csbm" => {
            let n_nodes = params.n_nodes;
            let n_communities = params.n_communities;
            let homophily = params.homophily;
            let avg_degree = params.avg_degree;
            let feat_dim = params.features_dim.unwrap_or(16);
            let mu = params.mu.unwrap_or(1.0);
            let theta = params.theta_exponent;
            let noise_ratio = params.feat_noise_ratio.unwrap_or(0.0);
            let seed = config.seed;
            let result = py
                .allow_threads(move || {
                    generate_csbm_core(
                        n_nodes, n_communities, homophily, avg_degree, feat_dim, mu,
                        "gaussian", None, theta, 0.0, noise_ratio, false, None, None, None,
                        seed,
                    )
                })
                .map_err(PyValueError::new_err)?;
            let graph = csbm_result_to_graph_output(&result);
            serde_json::to_string_pretty(&graph)
                .map_err(|err| PyValueError::new_err(format!("JSON serialization error: {err}")))
        }
        other => Err(PyValueError::new_err(format!(
            "Unknown model_type: '{other}'. Use 'sbm_classique', 'sbm', or 'csbm'"
        ))),
    }
}

/// Pure-Rust entry point for the CLI binary (`synth-graph-cli`).
///
/// Accepts the same JSON contract as [`generate_from_config`] and returns the
/// graph JSON string on success, or a plain error message on failure.
///
/// No PyO3 types are involved — this function is safe to call from any Rust
/// binary without a Python runtime.
///
/// # JSON input format
/// ```json
/// {
///   "model_type": "sbm_classique" | "sbm" | "csbm",
///   "seed": 42,
///   "parameters": {
///     "n_nodes": 500, "n_communities": 5,
///     "homophily": 0.8, "avg_degree": 10.0,
///     "features_dim": 16, "mu": 1.0,
///     "theta_exponent": null, "feat_noise_ratio": 0.0
///   }
/// }
/// ```
pub fn generate_from_config_native(config_json: &str) -> Result<String, String> {
    let config: GenerationConfig = serde_json::from_str(config_json)
        .map_err(|err| format!("Config parse error: {err}"))?;

    let params = &config.parameters;
    match config.model_type.as_str() {
        "sbm_classique" | "sbm" => {
            let block_size = params.n_nodes as f64 / params.n_communities as f64;
            let p_in = if block_size > 1.0 {
                (params.homophily * params.avg_degree / (block_size - 1.0)).min(1.0)
            } else {
                0.0
            };
            let p_out = if params.n_communities > 1 {
                ((1.0 - params.homophily) * params.avg_degree
                    / ((params.n_communities - 1) as f64 * block_size))
                    .min(1.0)
            } else {
                0.0
            };
            let result = generate_sbm_core(
                params.n_nodes,
                params.n_communities,
                p_in,
                p_out,
                params.theta_exponent,
                config.seed,
            )?;
            let graph = sbm_result_to_graph_output(&result);
            serde_json::to_string_pretty(&graph)
                .map_err(|err| format!("JSON serialization error: {err}"))
        }
        "csbm" => {
            let feat_dim = params.features_dim.unwrap_or(16);
            let mu = params.mu.unwrap_or(1.0);
            let noise_ratio = params.feat_noise_ratio.unwrap_or(0.0);
            let result = generate_csbm_core(
                params.n_nodes,
                params.n_communities,
                params.homophily,
                params.avg_degree,
                feat_dim,
                mu,
                "gaussian",
                None,
                params.theta_exponent,
                0.0,
                noise_ratio,
                false,
                None,
                None,
                None,
                config.seed,
            )?;
            let graph = csbm_result_to_graph_output(&result);
            serde_json::to_string_pretty(&graph)
                .map_err(|err| format!("JSON serialization error: {err}"))
        }
        other => Err(format!(
            "Unknown model_type: '{other}'. Use 'sbm_classique', 'sbm', or 'csbm'"
        )),
    }
}

#[pymodule]
fn synth_graph_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_sbm, m)?)?;
    m.add_function(wrap_pyfunction!(generate_csbm, m)?)?;
    m.add_function(wrap_pyfunction!(generate_sbm_json, m)?)?;
    m.add_function(wrap_pyfunction!(generate_sbm_json_to_file, m)?)?;
    m.add_function(wrap_pyfunction!(generate_csbm_json, m)?)?;
    m.add_function(wrap_pyfunction!(generate_csbm_json_to_file, m)?)?;
    m.add_function(wrap_pyfunction!(json_to_pyg, m)?)?;
    m.add_function(wrap_pyfunction!(generate_from_config, m)?)?;
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sbm_rejects_more_communities_than_nodes() {
        let err = generate_sbm_core(5, 10, 0.5, 0.1, None, Some(42)).unwrap_err();
        assert!(err.contains("n_communities"), "expected 'n_communities' in error: {err}");
    }

    #[test]
    fn sbm_allows_equal_communities_and_nodes() {
        generate_sbm_core(5, 5, 0.5, 0.1, None, Some(42))
            .expect("n_communities == n_nodes should be valid");
    }

    #[test]
    fn sbm_dc_rejects_more_communities_than_nodes() {
        let err = generate_sbm_core(5, 10, 0.5, 0.1, Some(3.0), Some(42)).unwrap_err();
        assert!(!err.is_empty());
    }

    #[test]
    fn sbm_dc_empty_block_guard_fires() {
        assert!(generate_sbm_core(2, 3, 0.5, 0.1, Some(3.0), Some(1)).is_err());
    }

    #[test]
    fn sbm_dc_equal_communities_and_nodes() {
        generate_sbm_core(4, 4, 0.6, 0.1, Some(3.0), Some(7))
            .expect("DC-SBM with n_communities == n_nodes should be valid");
    }

    fn make_node(id: usize, community: usize, features: Option<Vec<f32>>) -> NodeJson {
        NodeJson { id, community, features }
    }

    #[test]
    fn features_first_present_second_absent_errors() {
        let nodes = vec![
            make_node(0, 0, Some(vec![1.0, 2.0])),
            make_node(1, 1, None),
        ];
        assert!(collect_node_features(&nodes).is_err());
    }

    #[test]
    fn features_first_absent_second_present_errors() {
        let nodes = vec![
            make_node(0, 0, None),
            make_node(1, 1, Some(vec![1.0, 2.0])),
        ];
        assert!(collect_node_features(&nodes).is_err());
    }

    #[test]
    fn features_inconsistent_dim_errors() {
        let nodes = vec![
            make_node(0, 0, Some(vec![1.0, 2.0])),
            make_node(1, 1, Some(vec![3.0])),
        ];
        assert!(collect_node_features(&nodes).is_err());
    }

    #[test]
    fn features_all_absent_returns_none() {
        let nodes = vec![make_node(0, 0, None), make_node(1, 1, None)];
        assert_eq!(collect_node_features(&nodes).unwrap(), None);
    }

    #[test]
    fn features_empty_graph_returns_none() {
        assert_eq!(collect_node_features(&[]).unwrap(), None);
    }

    #[test]
    fn features_consistent_returns_flat_buffer() {
        let nodes = vec![
            make_node(0, 0, Some(vec![1.0, 2.0])),
            make_node(1, 1, Some(vec![3.0, 4.0])),
        ];
        assert_eq!(collect_node_features(&nodes).unwrap(), Some((2, vec![1.0, 2.0, 3.0, 4.0])));
    }

    #[test]
    fn decode_upper_tri_n2_k0() {
        assert_eq!(decode_upper_tri(0, 2), (0, 1));
    }

    #[test]
    fn decode_upper_tri_n4_all_entries() {
        let expected = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        for (flat_idx, &(expected_row, expected_col)) in expected.iter().enumerate() {
            assert_eq!(
                decode_upper_tri(flat_idx as u64, 4),
                (expected_row, expected_col),
                "flat_idx={flat_idx} should map to ({expected_row},{expected_col})"
            );
        }
    }

    #[test]
    fn decode_upper_tri_last_entry_n3() {
        assert_eq!(decode_upper_tri(2, 3), (1, 2));
    }

    #[test]
    fn bidirectional_flat_empty() {
        let (flat, total_directed) = build_bidirectional_edge_flat(&[]);
        assert_eq!(total_directed, 0);
        assert_eq!(flat.len(), 0);
    }

    #[test]
    fn bidirectional_flat_single_edge() {
        let (flat, total_directed) = build_bidirectional_edge_flat(&[(2, 5)]);
        assert_eq!(total_directed, 2);
        // Row-major 2×2: [src_fwd, src_bwd | dst_fwd, dst_bwd] = [2, 5, 5, 2]
        assert_eq!(flat, vec![2i64, 5, 5, 2]);
    }

    #[test]
    fn bidirectional_flat_two_edges() {
        let canonical = vec![(0i64, 1i64), (2i64, 3i64)];
        let (flat, total_directed) = build_bidirectional_edge_flat(&canonical);
        assert_eq!(total_directed, 4);
        assert_eq!(flat.len(), 2 * total_directed);
        // Row 0 (sources): fwd then rev = [0, 2, 1, 3]
        // Row 1 (targets): fwd then rev = [1, 3, 0, 2]
        assert_eq!(&flat[..4], &[0, 2, 1, 3]);
        assert_eq!(&flat[4..], &[1, 3, 0, 2]);
    }

    #[test]
    fn effective_h_in_unit_interval() {
        let result = generate_csbm_core(
            100, 4, 0.7, 5.0, 8, 1.0, "gaussian", None, None, 0.0, 0.0, false, None, None,
            None, Some(42),
        )
        .unwrap();
        assert!(
            result.effective_h >= 0.0 && result.effective_h <= 1.0,
            "effective_h={} not in [0,1]",
            result.effective_h
        );
    }

    #[test]
    fn fully_intra_graph_has_effective_h_one() {
        // p_out=0 → all edges are intra-block → effective_h must be 1.0
        let result = generate_sbm_core(20, 4, 0.5, 0.0, None, Some(5)).unwrap();
        if !result.canonical_edges.is_empty() {
            assert_eq!(result.effective_h, 1.0);
        }
    }

    #[test]
    fn empty_edge_graph_has_effective_h_zero() {
        // p_in=p_out=0 → no edges → effective_h defined as 0.0
        let result = generate_sbm_core(10, 2, 0.0, 0.0, None, Some(0)).unwrap();
        assert!(result.canonical_edges.is_empty());
        assert_eq!(result.effective_h, 0.0);
    }

    #[test]
    fn p_triangle_zero_is_deterministic() {
        let r1 = generate_csbm_core(
            20, 2, 0.8, 5.0, 4, 1.0, "gaussian", None, None, 0.0, 0.0, false, None, None,
            None, Some(99),
        )
        .unwrap();
        let r2 = generate_csbm_core(
            20, 2, 0.8, 5.0, 4, 1.0, "gaussian", None, None, 0.0, 0.0, false, None, None,
            None, Some(99),
        )
        .unwrap();
        assert_eq!(r1.canonical_edges.len(), r2.canonical_edges.len());
    }

    #[test]
    fn p_triangle_one_never_removes_edges() {
        let base = generate_csbm_core(
            30, 2, 0.5, 4.0, 4, 1.0, "gaussian", None, None, 0.0, 0.0, false, None, None,
            None, Some(7),
        )
        .unwrap();
        let with_triangles = generate_csbm_core(
            30, 2, 0.5, 4.0, 4, 1.0, "gaussian", None, None, 1.0, 0.0, false, None, None,
            None, Some(7),
        )
        .unwrap();
        assert!(with_triangles.canonical_edges.len() >= base.canonical_edges.len());
    }

    #[test]
    fn ensure_connected_produces_one_component() {
        let result = generate_csbm_core(
            50, 5, 0.5, 1.0, 4, 1.0, "gaussian", None, None, 0.0, 0.0, true, None, None,
            None, Some(11),
        )
        .unwrap();
        let n_nodes = result.n_nodes;
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
        for &(src, dst) in &result.canonical_edges {
            adj[src as usize].push(dst as usize);
            adj[dst as usize].push(src as usize);
        }
        let mut visited = vec![false; n_nodes];
        visited[0] = true;
        let mut queue = vec![0usize];
        let mut head = 0;
        while head < queue.len() {
            for &neighbor in &adj[queue[head]] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push(neighbor);
                }
            }
            head += 1;
        }
        assert!(visited.iter().all(|&was_visited| was_visited), "graph must be connected");
    }

    #[test]
    fn ensure_connected_false_does_not_panic() {
        generate_csbm_core(
            100, 10, 0.5, 0.001, 4, 1.0, "gaussian", None, None, 0.0, 0.0, false, None,
            None, None, Some(1),
        )
        .expect("must not panic even if disconnected");
    }

    #[test]
    fn gaussian_features_shape_and_finite() {
        let result = generate_csbm_core(
            10, 2, 0.7, 3.0, 8, 1.0, "gaussian", None, None, 0.0, 0.0, false, None, None,
            None, Some(1),
        )
        .unwrap();
        assert_eq!(result.x_flat.len(), result.n_nodes * result.feat_dim);
        assert!(result.x_flat.iter().all(|feat| feat.is_finite()));
    }

    #[test]
    fn uniform_features_shape_and_finite() {
        let result = generate_csbm_core(
            10, 2, 0.7, 3.0, 4, 1.0, "uniform", None, None, 0.0, 0.0, false, None, None,
            None, Some(2),
        )
        .unwrap();
        assert_eq!(result.x_flat.len(), result.n_nodes * result.feat_dim);
        assert!(result.x_flat.iter().all(|feat| feat.is_finite()));
    }

    #[test]
    fn laplacian_features_shape_and_finite() {
        let result = generate_csbm_core(
            10, 2, 0.7, 3.0, 4, 1.0, "laplacian", None, None, 0.0, 0.0, false, None, None,
            None, Some(3),
        )
        .unwrap();
        assert_eq!(result.x_flat.len(), result.n_nodes * result.feat_dim);
        assert!(result.x_flat.iter().all(|feat| feat.is_finite()));
    }

    #[test]
    fn sbm_json_round_trip_preserves_structure() {
        let result = generate_sbm_core(20, 3, 0.5, 0.1, None, Some(77)).unwrap();
        let graph_out = sbm_result_to_graph_output(&result);
        let json = serde_json::to_string(&graph_out).unwrap();
        let parsed: GraphOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.nodes.len(), result.n_nodes);
        assert_eq!(parsed.edges.len(), result.canonical_edges.len());
        let orig: std::collections::HashSet<(usize, usize)> = result
            .canonical_edges
            .iter()
            .map(|&(src, dst)| (src as usize, dst as usize))
            .collect();
        let parsed_set: std::collections::HashSet<(usize, usize)> =
            parsed.edges.iter().map(|edge| (edge.source, edge.target)).collect();
        assert_eq!(orig, parsed_set);
    }

    #[test]
    fn csbm_json_round_trip_preserves_features() {
        let result = generate_csbm_core(
            8, 2, 0.7, 3.0, 4, 1.0, "gaussian", None, None, 0.0, 0.0, false, None, None,
            None, Some(5),
        )
        .unwrap();
        let graph_out = csbm_result_to_graph_output(&result);
        let json = serde_json::to_string(&graph_out).unwrap();
        let parsed: GraphOutput = serde_json::from_str(&json).unwrap();
        for (node_idx, node) in parsed.nodes.iter().enumerate() {
            let feats = node.features.as_ref().expect("cSBM nodes must have features");
            assert_eq!(feats.len(), result.feat_dim);
            for (feat_idx, &orig) in
                result.x_flat[node_idx * result.feat_dim..(node_idx + 1) * result.feat_dim]
                    .iter()
                    .enumerate()
            {
                assert!(
                    (feats[feat_idx] - orig).abs() < 1e-6,
                    "feature [{node_idx},{feat_idx}]: got {}, expected {orig}",
                    feats[feat_idx]
                );
            }
        }
    }

    #[test]
    fn csbm_rejects_nan_d() {
        assert!(generate_csbm_core(
            10, 2, 0.5, f64::NAN, 4, 1.0, "gaussian", None, None, 0.0, 0.0, false, None,
            None, None, Some(1)
        )
        .is_err());
    }

    #[test]
    fn csbm_rejects_inf_d() {
        assert!(generate_csbm_core(
            10, 2, 0.5, f64::INFINITY, 4, 1.0, "gaussian", None, None, 0.0, 0.0, false,
            None, None, None, Some(1)
        )
        .is_err());
    }

    #[test]
    fn csbm_rejects_nan_mu() {
        assert!(generate_csbm_core(
            10, 2, 0.5, 3.0, 4, f64::NAN, "gaussian", None, None, 0.0, 0.0, false, None,
            None, None, Some(1)
        )
        .is_err());
    }

    #[test]
    fn csbm_rejects_nan_class_weight() {
        assert!(generate_csbm_core(
            10, 2, 0.5, 3.0, 4, 1.0, "gaussian",
            Some(&[f64::NAN, 1.0]),
            None, 0.0, 0.0, false, None, None, None, Some(1)
        )
        .is_err());
    }

    #[test]
    fn csbm_rejects_inf_class_weight() {
        assert!(generate_csbm_core(
            10, 2, 0.5, 3.0, 4, 1.0, "gaussian",
            Some(&[f64::INFINITY, 0.0]),
            None, 0.0, 0.0, false, None, None, None, Some(1)
        )
        .is_err());
    }

    #[test]
    fn sbm_rejects_nan_theta_exponent() {
        assert!(generate_sbm_core(10, 2, 0.5, 0.1, Some(f64::NAN), Some(1)).is_err());
    }

    #[test]
    fn csbm_rejects_nan_theta_exponent() {
        assert!(generate_csbm_core(
            10, 2, 0.5, 3.0, 4, 1.0, "gaussian", None, Some(f64::NAN), 0.0, 0.0, false,
            None, None, None, Some(1)
        )
        .is_err());
    }
}
