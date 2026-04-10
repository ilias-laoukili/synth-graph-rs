mod config;
mod visualise;

use crate::config::prompt_config;
use synth_graph_rs::generate_from_config_native;
use crossterm::style::Stylize;

/// Point d'entrée de l'application cliente d'interface (CLI)
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

