use candle_core::Device;
use internal::*;
use neural::{Activation, Network};

mod internal;
pub mod logger;
pub mod neural;
mod error;

const DECAY_RATE: f64 = 0.95;
const DECAY_STEP: usize = 10;
const LEARNING_RATE: f64 = 0.00001;
const LEVEL: logger::Level = logger::Level::Message;
const NB_ITER: usize = 1000;

fn get_device() -> Device {
    #[cfg(feature = "cuda")]
    { Device::new_cuda(0).unwrap() }
    // #[cfg(feature = "metal")]
    // { Device::new_metal(0).unwrap() }
    #[cfg(not(feature = "cuda"))]
    { Device::Cpu }
}

fn main() -> internal::error::Result<()> {
    let logger = logger::Logger::new(NB_ITER, false, LEVEL)?;
    let device = get_device();

    logger.log("Retrieve train data and labels from CSV files");
    let (train_data, train_label) = dataset::get_train(&device)?;

    logger.log(format!("- Train labels: {:?}", train_label));
    logger.log(format!("- Train datas: {:?}", train_data));

    let fan_in = train_data.dim(0)?;
    let fan_out = train_label.dim(0)?;

    let mut net = match memory::load(&logger, &device) {
        Ok(net) => net,
        Err(error) => {
            logger.error(format!("{:?}", error));
            Network::new(fan_in, fan_out, Activation::None, &device)
                .add_layer(32, Activation::ReLU)?
                .add_layer(32, Activation::ReLU)?
                .build(&logger)?
        }
    };

    let (losses, accuracies) = net.train(
        &logger,
        &train_data,
        &train_label,
        NB_ITER,
        LEARNING_RATE,
        DECAY_RATE,
        DECAY_STEP
    )?;

    if !losses.is_empty() || !accuracies.is_empty() {
        let path = frame::save_metrics_plot(&losses, &accuracies)?;
        println!(
            "Training metrics saved to {} (loss = red, left axis · accuracy = green, right axis 0-1). Open the file locally to inspect the curves.",
            path.display()
        );
    }

    memory::save(&net)?;

    // println!("Retrieve test data from CSV file");
    // let test_data = dataset::get_test(&device)?;
    // println!("test data shape: {:?}", test_data.shape());

    Ok(())
}
