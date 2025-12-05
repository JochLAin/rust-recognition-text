use candle_core::Device;
use neural::{Activation, Network};

mod dataset;
mod frame;
mod logger;
pub mod neural;

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
    let nb_iter = 100;

    println!("### Retrieve train data and labels from CSV files");
    let (train_data, train_label) = dataset::get_train(&device)?;

    println!("# Train csv");
    println!("Labels: {:?}", train_label);
    println!("Datas: {:?}", train_data);

    let fan_in = train_data.dim(0)?;
    let fan_out = train_label.dim(0)?;

    let mut logger = logger::Logger::new(nb_iter, false, logger::Level::Debug)?;
    let (losses, accuracies) = Network::new(&logger, fan_in, fan_out, Activation::Softmax, &device)
        .add_layer(32, Activation::ReLU)
        .add_layer(32, Activation::ReLU)
        .build()?
        .train(&train_data, &train_label, learning_rate, nb_iter)?;

    if !losses.is_empty() || !accuracies.is_empty() {
        let path = frame::save_metrics_plot(&losses, &accuracies)?;
        println!(
            "Training metrics saved to {} (loss = red, left axis · accuracy = green, right axis 0-1). Open the file locally to inspect the curves.",
            path.display()
        );
    }

    // println!("Retrieve test data from CSV file");
    // let test_data = dataset::get_test(&device)?;
    // println!("test data shape: {:?}", test_data.shape());

    Ok(())
}
