mod config;

use crate::config::prompt_config;
use crossterm::style::Stylize;

/// Point d'entrée de l'application cliente d'interface (CLI)
fn main() {
    match prompt_config() {
        Ok(configuration) => {
            println!("\n{}\n", "Configuration générée avec succès :".bold().green());
            println!("{:#?}", configuration);
            
            println!("\n{}", "=> Étape suivante : Transmission à la lib d'Ilias...".italic().dim());
        }
        Err(erreur) => {
            eprintln!("\n{} {}", "Erreur de saisie :".bold().red(), erreur);
        }
    }
}

