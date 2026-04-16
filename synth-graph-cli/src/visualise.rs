//! Terminal visualization module using Ratatui.
//!
//! This module provides three rendering strategies for graph visualization:
//! - `render_circular()` — Fixed circular layout (fast, O(n))
//! - `render_force_directed()` — Physics-based layout (in development)
//! - `render_stats()` — Aggregated statistics for large graphs (fast, O(n))
//!
//! The main entry point `render()` handles:
//! - Terminal state management (raw mode, alternate screen)
//! - Event loop for keyboard input
//! - Dispatching to appropriate renderer based on `RenderMode`

use crate::config::RenderMode;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, canvas::{Canvas, Line as CanvasLine}},
    Terminal,
};
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use std::io::{self, stdout};
use std::collections::HashMap;
use std::f64::consts::PI;
use synth_graph_rs::GraphOutput;

/// Main rendering entry point for the graph visualization.
///
/// # Responsibilities
/// 1. Enable raw terminal mode for event handling
/// 2. Switch to alternate screen buffer
/// 3. Enter event loop that:
///    - Redraws the graph on each iteration
///    - Listens for keyboard input
///    - Exits on 'q' press
/// 4. Restore terminal state
///
/// # Arguments
/// - `graph`: The graph to visualize (contains nodes, edges, metadata)
/// - `mode`: The rendering strategy (Basic, Force-Directed, MacroStats)
///
/// # Returns
/// - `Ok(())` on successful completion
/// - `Err(io::Error)` on terminal I/O failure
pub fn render(graph: &GraphOutput, mode: &RenderMode) -> io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    loop {
        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(1)
                .constraints([Constraint::Length(3), Constraint::Min(10)].as_ref())
                .split(f.area());

            let title_block = Block::default()
                .title(" Synth Graph RS - Visualisateur ")
                .borders(Borders::ALL)
                .style(Style::default().fg(Color::Cyan));
            let title_text = Paragraph::new(format!("Mode de rendu: {}", mode))
                .block(title_block);
            f.render_widget(title_text, chunks[0]);

            let main_block = Block::default()
                .title(" Rendu du graphe (Appuyez sur 'q' pour quitter) ")
                .borders(Borders::ALL);

            match mode {
                RenderMode::Basic => render_circular(f, chunks[1], graph, main_block),
                RenderMode::ForceDirected => render_force_directed(f, chunks[1], graph, main_block),
                RenderMode::MacroStats => render_stats(f, chunks[1], graph, main_block),
            }
        })?;

        if let Event::Key(key) = event::read()? {
            if let KeyCode::Char('q') = key.code {
                break;
            }
        }
    }

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen
    )?;
    terminal.show_cursor()?;

    Ok(())
}

/// Renders the graph using a fixed circular layout.
///
/// # Algorithm
/// 1. Arrange nodes evenly on a circle (trigonometric distribution)
/// 2. Store node positions in a HashMap for edge drawing
/// 3. Draw edges (limited to 2000 for terminal performance)
/// 4. Draw nodes on top, colored by community
///
/// # Performance
/// - Time: O(n + e) where e ≤ 2000
/// - Space: O(n) for position cache
/// - Suitable for graphs up to ~5000 nodes
///
/// # Arguments
/// - `f`: Ratatui frame for rendering
/// - `area`: Drawing area rectangle
/// - `graph`: Graph data (nodes, edges, communities)
/// - `block`: UI block container for the canvas
fn render_circular(f: &mut ratatui::Frame, area: Rect, graph: &GraphOutput, block: Block) {
    let canvas = Canvas::default()
        .block(block)
        .paint(|ctx| {
            let n = graph.nodes.len();
            if n == 0 {
                ctx.print(5.0, 5.0, "Graphe vide".red());
                return;
            }
            
            let center_x = 50.0;
            let center_y = 50.0;
            let radius = 40.0;
            
            let mut pos = HashMap::new();
            
            // 1. Calculer les positions circulaires de chaque nœud
            for (i, node) in graph.nodes.iter().enumerate() {
                let theta = (i as f64) * 2.0 * PI / (n as f64);
                let x = center_x + radius * theta.cos();
                let y = center_y + radius * theta.sin();
                pos.insert(node.id, (x, y, node.community));
            }
            
            // 2. Dessiner les arêtes
            // Dans un terminal, afficher trop de lignes peut ralentir, on limite à 2000 pour la fluidité
            let max_edges = 2000; 
            for (i, edge) in graph.edges.iter().enumerate() {
                if i > max_edges { break; }
                if let (Some(&(x1, y1, _)), Some(&(x2, y2, _))) = (pos.get(&edge.source), pos.get(&edge.target)) {
                    ctx.draw(&CanvasLine {
                        x1, y1, x2, y2,
                        color: Color::DarkGray,
                    });
                }
            }
            
            // 3. Dessiner les nœuds par dessus, avec la couleur de leur communauté
            let colors = [
                Color::Red, Color::Green, Color::Blue, Color::Yellow, 
                Color::Magenta, Color::Cyan, Color::White
            ];
            for (_id, (x, y, comm)) in pos {
                let color = colors[comm % colors.len()];
                ctx.print(x, y, "•".fg(color));
            }
        })
        .x_bounds([0.0, 100.0])
        .y_bounds([0.0, 100.0]);
    f.render_widget(canvas, area);
}

/// Placeholder for physics-based graph layout (Fruchterman-Reingold algorithm).
///
/// # Future Implementation
/// Will simulate forces:
/// - **Repulsion**: Nodes push each other away (k² / distance)
/// - **Attraction**: Connected nodes pull toward each other (distance² / k)
/// - **Cooling**: Gradually reduce node movement velocity
///
/// This reveals natural clustering and structural patterns in the graph.
///
/// # Current Status
/// In development. Currently displays a placeholder message.
///
/// # Expected Performance
/// - Time: O(iterations × n²) — computationally expensive
/// - Space: O(n) for node positions
/// - Suitable for graphs up to ~10,000 nodes
///
/// # Challenge
/// Ratatui is optimized for static rendering; iterative animations
/// require careful frame rate management in terminal environment.
///
/// # Arguments
/// - `f`: Ratatui frame for rendering
/// - `area`: Drawing area rectangle
/// - `graph`: Graph data (currently unused, for future use)
/// - `block`: UI block container for the canvas
fn render_force_directed(f: &mut ratatui::Frame, area: Rect, graph: &GraphOutput, block: Block) {
    let _ = graph; // Evite le warning de variable non utilisée
    let canvas = Canvas::default()
        .block(block)
        .paint(|ctx| {
            ctx.print(5.0, 50.0, "Algo de force en cours de dev...".green());
            ctx.print(5.0, 45.0, "=> Utilisez le mode Circulaire ou MacroStats en attendant.".gray());
        })
        .x_bounds([0.0, 100.0])
        .y_bounds([0.0, 100.0]);
    f.render_widget(canvas, area);
}

/// Renders aggregated graph statistics and community distribution as ASCII histograms.
///
/// # Display Components
/// 1. **Global Statistics**:
///    - Total node count
///    - Total edge count
///    - Measured homophily (how much edges respect community boundaries)
///
/// 2. **Community Distribution**:
///    - Bar chart (using "█" character) showing node distribution per community
///    - Sorted by community ID
///    - Bar length proportional to community size
///
/// # Algorithm
/// 1. Count nodes per community (O(n))
/// 2. Sort communities by ID
/// 3. Render each community as a bar with proportional length
///
/// # Performance
/// - Time: O(n) single pass through nodes
/// - Space: O(k) where k = number of communities
/// - **Scalable to 1M+ nodes** — ideal for very large graphs
///
/// # Use Case
/// When graph is too large to visualize spatially, this shows the
/// **composition** and **balance** of communities at a glance.
///
/// # Arguments
/// - `f`: Ratatui frame for rendering
/// - `area`: Drawing area rectangle
/// - `graph`: Graph data (metadata, nodes)
/// - `block`: UI block container for the paragraph
fn render_stats(f: &mut ratatui::Frame, area: Rect, graph: &GraphOutput, block: Block) {
    // Histogram and text information for large graphs
    let meta = &graph.metadata;
    
    // Calculate community distribution
    let mut comm_counts = HashMap::new();
    for node in &graph.nodes {
        *comm_counts.entry(node.community).or_insert(0) += 1;
    }
    
    let mut stats_text = vec![
        Line::from(vec![Span::raw("📊 ").green(), Span::styled("Statistiques globales du graphe", Style::default().bold())]),
        Line::from(""),
        Line::from(format!("• Nœuds générés : {}", meta.n_nodes)),
        Line::from(format!("• Arêtes générées : {}", meta.n_edges)),
        Line::from(format!("• Homophilie réelle mesurée : {:.2}", meta.homophily)),
        Line::from(""),
        Line::from(Span::styled("Répartition des Blocs / Communautés :", Style::default().underlined())),
    ];
    
    let mut comms: Vec<_> = comm_counts.keys().collect();
    comms.sort();
    
    for &c in comms {
        let count = comm_counts[&c];
        let bar_len = (count as f64 / meta.n_nodes as f64 * 30.0) as usize;
        let bar = "█".repeat(bar_len);
        stats_text.push(Line::from(format!("  Communauté {} : {} ({} nœuds)", c, bar.cyan(), count)));
    }
    
    let para = Paragraph::new(stats_text).block(block);
    f.render_widget(para, area);
}
