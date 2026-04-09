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
