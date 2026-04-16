//! Synth Graph CLI — Main entry point and orchestration.
//!
//! This binary provides an interactive terminal interface for generating
//! and visualizing synthetic graphs (SBM, DC-SBM, cSBM).
//!
//! # Workflow
//! 1. **Gather configuration** via interactive prompts (config module)
//! 2. **Serialize to JSON** for the native generator
//! 3. **Generate graph** using Rust native engine (synth_graph_rs)
//! 4. **Save output** to `graph_output.json`
//! 5. **Deserialize result** back to Rust structures
//! 6. **Visualize** in terminal using Ratatui (visualise module)
//!
//! # Error Handling
//! Errors at any step are caught and logged to stderr. The application
//! continues gracefully with informative messages.
//!
//! # Output
//! - `graph_output.json`: Generated graph in JSON format with metadata,
//!   nodes (with community assignments and optional features), and edges.

mod config;
mod visualise;

use crate::config::prompt_config;
use synth_graph_rs::generate_from_config_native;
use crossterm::style::Stylize;

/// Entry point for the Synth Graph CLI application.
///
/// # Execution Steps
/// 1. Call `prompt_config()` to launch interactive configuration UI
/// 2. On success:
///    - Serialize config to JSON
///    - Print generated configuration to stdout
///    - Call native generator via `generate_from_config_native()`
///    - Write result to `graph_output.json`
///    - Launch visualization with chosen render mode
/// 3. On error at any step:
///    - Print error message to stderr
///    - Exit gracefully
///
/// # Dependencies
/// - `config`: Interactive parameter collection
/// - `visualise`: Terminal rendering
/// - `synth_graph_rs`: Native graph generation engine
fn main() {
    match prompt_config() {
        Ok(cfg) => {
            println!("\n{}\n", "Configuration générée avec succès :".bold().green());
            
            let json = cfg.to_json();
            println!("{}", json);
            println!("\n{}", "=> Transmission au moteur natif...".italic().dim());
            
            match generate_from_config_native(&json) {
                Ok(graph_json) => {
                    let path = "graph_output.json";
                    std::fs::write(path, &graph_json)
                        .expect("Impossible d'écrire le fichier de sortie");
                    println!("{} {}", "Graphe généré avec succès →".bold().green(), path);
                    
                    println!("\n{}", "=> Lancement du visualisateur Ratatui...".bold().yellow());
                    
                    // Désérialisation du JSON pour Ratatui
                    let graph: synth_graph_rs::GraphOutput = serde_json::from_str(&graph_json)
                        .expect("Erreur lors de la lecture du graphe généré");
                    
                    if let Err(e) = visualise::render(&graph, cfg.render_mode()) {
                        eprintln!("Erreur lors de l'exécution de Ratatui : {}", e);
                    }
                }
                Err(e) => eprintln!("{} {}", "Erreur de génération :".bold().red(), e),
            }
        }
        Err(e) => eprintln!("{} {}", "Erreur de saisie :".bold().red(), e),
    }
}

