use candle_core::{Device};
use eframe::egui;
use egui_plot::{Plot, Line, PlotPoints};
use neural::{Activation, Network};

mod dataset;
pub mod neural;

struct App {
  x: Vec<f64>,
  y: Vec<f64>,
}

impl eframe::App for App {
  fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
    egui::CentralPanel::default().show(ctx, |ui| {
      let points: PlotPoints = self.loss
        .iter()
        .enumerate()
        .map(|(i, v)| [i as f64, *v])
        .collect::<Vec<_>>()
        .into();

      Plot::new("loss_plot").show(ui, |plot_ui| {
        plot_ui.line(Line::new(points));
      });
    });

    ctx.request_repaint(); // ← essentiel pour rafraîchir en continu
  }
}

fn get_app() -> App {
  let app = App { x: vec![], y: vec![] };
  eframe::run_native(
    "Training Plot",
    eframe::NativeOptions::default(),
    Box::new(|_| Box::new(app)),
  );

  app
}

fn get_device() -> Device {
  // #[cfg(feature = "cuda")]
  // { Device::new_cuda(0).unwrap() }
  // #[cfg(not(feature = "cuda"))]
  // { Device::Cpu }

  Device::Cpu
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let device = get_device();
  let learning_rate = 0.1f64;
  let nb_iter = 10000;

  println!("### Retrieve train data and labels from CSV files");
  let (train_data, train_label) = dataset::get_train(&device)?;

  println!("# Train csv");
  println!("Labels: {:?}", train_label);
  println!("Datas: {:?}", train_data);

  let fan_in = train_data.dim(0)?;
  let fan_out = train_label.dim(0)?;

  let (_losses, _accs) = Network::new(fan_in, fan_out, Activation::Softmax, &device)
    .add_layer(32, Activation::ReLU)
    .add_layer(32, Activation::ReLU)
    .build()?
    .quiet(true)
    .train(&train_data, &train_label, learning_rate, nb_iter)?;

  // println!("Retrieve test data from CSV file");
  // let test_data = dataset::get_test(&device)?;
  // println!("test data shape: {:?}", test_data.shape());

  Ok(())
}
