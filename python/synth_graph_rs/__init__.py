# Re-export everything from the compiled Rust extension so users only need to
# `import synth_graph_rs` — they don't have to know about the `.so` internals.
#
# Low-level numpy API (raw arrays):
#   generate_sbm(n_nodes, n_communities, p_in, p_out, ...)
#     → (edge_index [2,2E], labels [N], effective_h)
#
#   generate_csbm(n_nodes, n_communities, homophily, avg_degree, feat_dim, mu, ...)
#     → (edge_index [2,2E], x [N,F], labels [N], effective_h)
#
#   generate_sbm_json / generate_csbm_json
#     → JSON string matching the roadmap contract
#
#   json_to_pyg(json_str)
#     → (edge_index [2,2E], labels [N], x [N,F] | None)
#
#   generate_from_config(config_json)
#     → JSON string  (config-driven entry point, mirrors the CLI contract)
#
# High-level PyG API (torch_geometric.data.Data objects):
#   from synth_graph_rs.pyg import sbm_to_pyg, csbm_to_pyg, json_to_pyg_data

from synth_graph_rs._synth_graph_rs import (  # type: ignore[import]
    generate_sbm,
    generate_csbm,
    generate_sbm_json,
    generate_sbm_json_to_file,
    generate_csbm_json,
    generate_csbm_json_to_file,
    json_to_pyg,
    generate_from_config,
)

__all__ = [
    "generate_sbm",
    "generate_csbm",
    "generate_sbm_json",
    "generate_sbm_json_to_file",
    "generate_csbm_json",
    "generate_csbm_json_to_file",
    "json_to_pyg",
    "generate_from_config",
]
