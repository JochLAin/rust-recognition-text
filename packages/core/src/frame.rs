use eframe::egui;
use eframe::egui::Color32;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use plotters::prelude::{
    BitMapBackend, ChartBuilder, GREEN, IntoDrawingArea, LineSeries, PathElement, RED, Text, WHITE,
};
use plotters::style::{Color, IntoFont, ShapeStyle, TextStyle};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

struct MetricsApp {
    losses: Vec<f64>,
    accuracies: Vec<f64>,
}

impl MetricsApp {
    fn new(losses: Vec<f64>, accuracies: Vec<f64>) -> Self {
        Self { losses, accuracies }
    }

    fn to_points(&self, values: &[f64]) -> PlotPoints<'_> {
        values
            .iter()
            .enumerate()
            .map(|(idx, value)| [idx as f64, *value])
            .collect::<Vec<_>>()
            .into()
    }
}

impl eframe::App for MetricsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Training metrics");

            if self.losses.is_empty() && self.accuracies.is_empty() {
                ui.label("No metrics recorded yet.");
                return;
            }

            Plot::new("loss_plot")
                .legend(Legend::default())
                .show(ui, |plot_ui| {
                    if !self.losses.is_empty() {
                        let points = self.to_points(&self.losses);
                        plot_ui.line(Line::new("Loss", points).color(Color32::LIGHT_RED));
                    }
                });

            ui.separator();

            Plot::new("accuracy_plot")
                .legend(Legend::default())
                .show(ui, |plot_ui| {
                    if !self.accuracies.is_empty() {
                        let points = self.to_points(&self.accuracies);
                        plot_ui.line(Line::new("Accuracy", points).color(Color32::LIGHT_GREEN));
                    }
                });
        });
    }
}

pub fn show_metrics_window(losses: &Vec<f64>, accuracies: &Vec<f64>) -> Result<(), Box<dyn Error>> {
    let window_losses = losses.clone();
    let window_accuracies = accuracies.clone();

    match eframe::run_native(
        "Training Plot",
        eframe::NativeOptions::default(),
        Box::new(move |_| {
            Ok(Box::new(MetricsApp::new(
                window_losses.clone(),
                window_accuracies.clone(),
            )))
        }),
    ) {
        Ok(_) => Ok(()),
        Err(err) => {
            eprintln!(
                "Unable to open an interactive metrics window ({err}). Saving the plot instead."
            );
            let path = save_metrics_plot(&losses, &accuracies)?;
            println!(
                "Training metrics saved to {} (loss = red, left axis · accuracy = green, right axis 0-1). Open the file locally to inspect the curves.",
                path.display()
            );
            Ok(())
        }
    }
}

pub fn save_metrics_plot(losses: &[f64], accuracies: &[f64]) -> Result<PathBuf, Box<dyn Error>> {
    let output_dir = Path::new("assets");
    fs::create_dir_all(output_dir)?;
    let path = output_dir.join("training_metrics.png");
    let output_path = path.to_string_lossy().into_owned();
    let root = BitMapBackend::new(&output_path, (1200, 720)).into_drawing_area();
    root.fill(&WHITE)?;
    let (chart_area, legend_area) = root.split_horizontally(980);
    legend_area.fill(&WHITE)?;

    let max_len = losses.len().max(accuracies.len());
    let (loss_min_raw, loss_max_raw) = compute_range(losses, 0.0, 1.0);
    let (loss_min, loss_max) = expand_range(loss_min_raw, loss_max_raw, 0.1, None, None);
    let (acc_min_raw, acc_max_raw) = (0.0, 1.0);
    let (acc_min, acc_max) = expand_range(acc_min_raw, acc_max_raw, 0.1, Some(0.0), Some(1.0));

    if max_len == 0 {
        chart_area.titled("Training Metrics", ("sans-serif", 36).into_font())?;
        root.present()?;
        return Ok(path);
    }

    let x_end = (max_len - 1) as f64;
    let mut chart = ChartBuilder::on(&chart_area)
        .caption("Training Metrics", ("sans-serif", 36).into_font())
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .right_y_label_area_size(70)
        .build_cartesian_2d(0f64..x_end, loss_min..loss_max)?
        .set_secondary_coord(0f64..x_end, acc_min..acc_max);

    chart
        .configure_mesh()
        .x_desc("Iteration")
        .y_desc("Loss")
        .x_labels(8)
        .y_labels(8)
        .max_light_lines(4)
        .light_line_style(&WHITE.mix(0.1))
        .label_style(("sans-serif", 22))
        .axis_style(&WHITE.mix(0.8))
        .draw()?;

    chart
        .configure_secondary_axes()
        .y_desc("Accuracy")
        .label_style(("sans-serif", 22))
        .axis_style(&WHITE.mix(0.8))
        .draw()?;

    if !losses.is_empty() {
        let series = losses
            .iter()
            .enumerate()
            .map(|(idx, value)| (idx as f64, *value));
        chart
            .draw_series(LineSeries::new(series, &RED))?
            .label("Loss")
            .legend(|(x, y)| {
                use plotters::prelude::PathElement;
                PathElement::new(vec![(x, y), (x + 20, y)], &RED)
            });
    }

    if !accuracies.is_empty() {
        let series = accuracies
            .iter()
            .enumerate()
            .map(|(idx, value)| (idx as f64, *value));
        chart
            .draw_secondary_series(LineSeries::new(series, &GREEN))?
            .label("Accuracy")
            .legend(|(x, y)| {
                use plotters::prelude::PathElement;
                PathElement::new(vec![(x, y), (x + 20, y)], &GREEN)
            });
    }

    draw_legend(&legend_area)?;

    root.present()?;
    Ok(path)
}

fn compute_range(values: &[f64], fallback_min: f64, fallback_max: f64) -> (f64, f64) {
    if values.is_empty() {
        return (fallback_min, fallback_max);
    }

    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for value in values {
        min_v = min_v.min(*value);
        max_v = max_v.max(*value);
    }

    if !min_v.is_finite() || !max_v.is_finite() {
        return (fallback_min, fallback_max);
    }

    if (max_v - min_v).abs() < f64::EPSILON {
        let eps = if max_v == 0.0 { 1.0 } else { max_v.abs() * 0.1 };
        min_v -= eps;
        max_v += eps;
    }

    (min_v, max_v)
}

fn expand_range(
    min_v: f64,
    max_v: f64,
    ratio: f64,
    floor: Option<f64>,
    ceil: Option<f64>,
) -> (f64, f64) {
    let mut min_v = min_v;
    let mut max_v = max_v;
    if !min_v.is_finite() || !max_v.is_finite() {
        return (min_v, max_v);
    }

    let span = (max_v - min_v).abs();
    let pad = if span < f64::EPSILON {
        if max_v == 0.0 {
            0.0
        } else {
            max_v.abs() * ratio
        }
    } else {
        span * ratio
    };

    if pad > 0.0 {
        if !floor
            .map(|target| (min_v - target).abs() < f64::EPSILON)
            .unwrap_or(false)
        {
            min_v -= pad;
        }

        if !ceil
            .map(|target| (max_v - target).abs() < f64::EPSILON)
            .unwrap_or(false)
        {
            max_v += pad;
        }
    }

    if let Some(f) = floor {
        min_v = min_v.max(f);
    }
    if let Some(c) = ceil {
        max_v = max_v.min(c);
    }

    if (max_v - min_v).abs() < f64::EPSILON {
        max_v = min_v + 1.0;
    }

    (min_v, max_v)
}

fn draw_legend(
    area: &plotters::drawing::DrawingArea<BitMapBackend<'_>, plotters::coord::Shift>,
) -> Result<(), Box<dyn Error>> {
    let title_font = TextStyle::from(("sans-serif", 30).into_font());
    let entry_font = TextStyle::from(("sans-serif", 24).into_font());

    area.draw(&Text::new("Legend", (10, 40), title_font))?;

    draw_legend_entry(area, 80, "Loss", &RED, &entry_font)?;
    draw_legend_entry(area, 130, "Accuracy", &GREEN, &entry_font)?;

    Ok(())
}

fn draw_legend_entry(
    area: &plotters::drawing::DrawingArea<BitMapBackend<'_>, plotters::coord::Shift>,
    y: i32,
    label: &str,
    color: &plotters::style::RGBColor,
    font: &TextStyle<'_>,
) -> Result<(), Box<dyn Error>> {
    area.draw(&PathElement::new(
        vec![(10, y), (90, y)],
        ShapeStyle::from(color).stroke_width(4),
    ))?;
    area.draw(&Text::new(label, (100, y + 5), font.clone()))?;
    Ok(())
}
