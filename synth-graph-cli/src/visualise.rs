use crate::config::RenderMode;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, canvas::{Canvas, MapResolution, Map}},
    Terminal,
};
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use serde::{Deserialize, Serialize};
use std::io::{self, stdout};
use std::collections::HashMap;
use synth_graph_rs::GraphOutput;

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

fn render_circular(f: &mut ratatui::Frame, area: Rect, graph: &GraphOutput, block: Block) {
    // Basic circular layout
    let canvas = Canvas::default()
        .block(block)
        .paint(|ctx| {
            // Implémentation du rendu circulaire basique (à développer plus tard)
            ctx.print(5.0, 5.0, "Visualisation circulaire en cours de dev...".yellow());
        })
        .x_bounds([0.0, 100.0])
        .y_bounds([0.0, 100.0]);
    f.render_widget(canvas, area);
}

fn render_force_directed(f: &mut ratatui::Frame, area: Rect, graph: &GraphOutput, block: Block) {
    // Fruchterman-Reingold Force-Directed Layout
    let canvas = Canvas::default()
        .block(block)
        .paint(|ctx| {
            // Implémentation de Force Directed (à développer plus tard)
            ctx.print(5.0, 5.0, "Algorithme de force en cours de dev...".green());
        })
        .x_bounds([0.0, 100.0])
        .y_bounds([0.0, 100.0]);
    f.render_widget(canvas, area);
}

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
