use eframe::egui;
use eframe::egui::Color32;
use egui_plot::{Legend, Line, Plot, PlotPoints};

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

pub fn show_metrics_window(losses: Vec<f64>, accuracies: Vec<f64>) -> Result<(), eframe::Error> {
    eframe::run_native(
        "Training Plot",
        eframe::NativeOptions::default(),
        Box::new(move |_| {
            Ok(Box::new(MetricsApp::new(
                losses.clone(),
                accuracies.clone(),
            )))
        }),
    )
}
