use chrono::Local;
use plotters::prelude::{
    BitMapBackend, ChartBuilder, GREEN, IntoDrawingArea, LineSeries, PathElement, RED, Text, WHITE,
};
use plotters::style::{Color, IntoFont, ShapeStyle, TextStyle};
use std::error::Error;
use std::path::{PathBuf};
use crate::internal::dataset::get_root_dir;

pub fn save_metrics_plot(losses: &[f64], accuracies: &[f64]) -> Result<PathBuf, Box<dyn Error>> {
    let path = get_root_dir().join(
        format!(
            "datasets/metrics/training.{}.png",
            Local::now().format("%Y%m%d%H%M%S%f").to_string()
        )
        .to_string(),
    );

    let output_path = path.to_string_lossy().into_owned();
    let root = BitMapBackend::new(&output_path, (1200, 720)).into_drawing_area();
    root.fill(&WHITE)?;
    let (chart_area, legend_area) = root.split_horizontally(980);
    legend_area.fill(&WHITE)?;

    let max_len = losses.len().max(accuracies.len());
    let (loss_min, loss_max) = (0.0, 20.0);
    let (acc_min, acc_max) = (0.0, 1.0);

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
