use crossterm::style::Stylize;
use inquire::{CustomType, Select, validator::Validation};
use serde::Serialize;

/// Represents the mathematical model chosen for graph generation.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelType {
    /// Classic Stochastic Block Model
    #[serde(rename = "sbm_classique")]
    SBM,
    /// Degree-Corrected Stochastic Block Model
    #[serde(rename = "sbm")]
    DCSBM,
    /// Contextual Stochastic Block Model
    #[serde(rename = "csbm")]
    CSBM,
}

impl std::fmt::Display for ModelType {
    /// Displays models in the selection menu
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::SBM => write!(formatter, "SBM Classic (Stochastic Block Model)"),
            ModelType::DCSBM => write!(formatter, "DC-SBM (Degree-Corrected SBM - Pareto Law)"),
            ModelType::CSBM => write!(formatter, "cSBM (Contextual SBM - With features)"),
        }
    }
}

/// Graph generation parameters.
/// Fields are kept private to enforce encapsulation.
#[derive(Debug, Serialize)]
pub struct Parameters {
    n_nodes: usize,
    n_communities: usize,
    homophily: f64,
    avg_degree: f64,
    features_dim: Option<usize>,
    mu: Option<f64>,
    theta_exponent: Option<f64>,
    feat_noise_ratio: Option<f64>,
}

/// Visual rendering strategy chosen by the user.
#[derive(Debug, Clone)]
pub enum RenderMode {
    /// Static circular layout
    Basic,
    /// Physics-based algorithm (node repulsion, edge attraction)
    ForceDirected,
    /// Aggregated view for very large graphs
    MacroStats,
}

impl std::fmt::Display for RenderMode {
    /// Displays rendering modes in the selection menu
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderMode::Basic => write!(formatter, "Basic (Fixed circular layout)"),
            RenderMode::ForceDirected => {
                write!(formatter, "Force-Directed (Organic, Fruchterman-Reingold)")
            }
            RenderMode::MacroStats => write!(formatter, "Macro (For >100k nodes, Heatmap)"),
        }
    }
}

/// Complete configuration combining the model, its parameters and the render mode.
#[derive(Debug, Serialize)]
pub struct Config {
    model_type: ModelType,
    #[serde(skip)]
    render_mode: RenderMode,
    seed: Option<u64>,
    parameters: Parameters,
}

impl Config {
    /// Convertit la configuration en JSON pour le moteur de génération.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap()
    }

    /// Récupère le mode de rendu choisi
    pub fn render_mode(&self) -> &RenderMode {
        &self.render_mode
    }
}

/// Launches the interactive terminal interface to build the configuration.
pub fn prompt_config() -> Result<Config, inquire::InquireError> {
    println!("\n{}\n", "=== Graph Generation Interface ===".bold().cyan());

    let model_options = vec![ModelType::SBM, ModelType::DCSBM, ModelType::CSBM];
    let model_type = Select::new(
        "Which mathematical model do you want to generate?",
        model_options,
    )
    .prompt()?;

    let render_options = vec![
        RenderMode::Basic,
        RenderMode::ForceDirected,
        RenderMode::MacroStats,
    ];
    let render_mode =
        Select::new("Which visual rendering mode do you want?", render_options).prompt()?;

    let n_nodes = CustomType::<usize>::new("Total number of vertices (n_nodes):")
        .with_default(100)
        .with_validator(validate_pos_usize("Number of nodes must be positive"))
        .prompt()?;

    let n_communities = CustomType::<usize>::new("Number of blocks/communities (k):")
        .with_default(3)
        .with_validator(validate_pos_usize("Number of communities must be positive"))
        .prompt()?;

    let homophily =
        CustomType::<f64>::new("Homophily (Internal vs external link probability [0.0 - 1.0]):")
            .with_default(0.8)
            .with_validator(validate_range_f64(
                0.0,
                1.0,
                "Homophily must be between 0.0 and 1.0",
            ))
            .prompt()?;

    let avg_degree = CustomType::<f64>::new("Expected average degree:")
        .with_default(5.0)
        .with_validator(validate_min_f64(
            0.0,
            "Average degree must be strictly positive",
            false,
        ))
        .prompt()?;

    let mut features_dim = None;
    let mut mu = None;
    let mut theta_exponent = None;
    let mut feat_noise_ratio = None;

    match model_type {
        ModelType::CSBM => {
            println!("\n  {} cSBM-specific parameters", "::".bold().yellow());
            features_dim = Some(
                CustomType::<usize>::new("Feature vector size (dimension):")
                    .with_default(8)
                    .with_validator(validate_pos_usize("Dimension must be positive"))
                    .prompt()?,
            );

            mu = Some(
                CustomType::<f64>::new("Average centroid separation (mu):")
                    .with_default(1.0)
                    .with_validator(validate_min_f64(0.0, "Mu must be non-negative", true))
                    .prompt()?,
            );

            feat_noise_ratio = Some(
                CustomType::<f64>::new("Noise / anomaly ratio (0.0=perfect, 1.0=chaos):")
                    .with_default(0.0)
                    .with_validator(validate_range_f64(
                        0.0,
                        1.0,
                        "Noise ratio must be between 0.0 and 1.0",
                    ))
                    .prompt()?,
            );
        }
        ModelType::DCSBM => {
            println!("\n  {} DC-SBM-specific parameters", "::".bold().blue());
            theta_exponent = Some(
                CustomType::<f64>::new("Pareto law for degrees (theta exponent):")
                    .with_default(2.5)
                    .with_validator(validate_min_f64(
                        1.0,
                        "Exponent must be greater than 1.0",
                        false,
                    ))
                    .prompt()?,
            );
        }
        ModelType::SBM => {}
    }

    Ok(Config {
        model_type,
        render_mode,
        seed: None,
        parameters: Parameters {
            n_nodes,
            n_communities,
            homophily,
            avg_degree,
            features_dim,
            mu,
            theta_exponent,
            feat_noise_ratio,
        },
    })
}

// --- Validation Helpers ---

fn validate_pos_usize(
    error_msg: &'static str,
) -> impl inquire::validator::CustomTypeValidator<usize> {
    move |val: &usize| {
        if *val > 0 {
            Ok(Validation::Valid)
        } else {
            Ok(Validation::Invalid(error_msg.into()))
        }
    }
}

fn validate_range_f64(
    min: f64,
    max: f64,
    error_msg: &'static str,
) -> impl inquire::validator::CustomTypeValidator<f64> {
    move |val: &f64| {
        if (*val >= min) && (*val <= max) {
            Ok(Validation::Valid)
        } else {
            Ok(Validation::Invalid(error_msg.into()))
        }
    }
}

fn validate_min_f64(
    min: f64,
    error_msg: &'static str,
    inclusive: bool,
) -> impl inquire::validator::CustomTypeValidator<f64> {
    move |val: &f64| {
        let valid = if inclusive { *val >= min } else { *val > min };
        if valid {
            Ok(Validation::Valid)
        } else {
            Ok(Validation::Invalid(error_msg.into()))
        }
    }
}
