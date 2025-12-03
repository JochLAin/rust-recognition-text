use candle_core::{Device};

mod dataset;
mod neural;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let device = {
    #[cfg(feature = "cuda")]
    { Device::new_cuda(0)? }
    #[cfg(not(feature = "cuda"))]
    { Device::Cpu }
  };

  println!("Retrieve train data and labels from CSV files");
  let (train_data, train_label) = dataset::get_train(&device)?;

  println!("train data shape: {:?}", train_data.shape());
  println!("train data dims: {:?}", train_data.dims());

  println!("train labels shape: {:?}", train_label.shape());
  println!("train labels dims: {:?}", train_label.dims());

  println!("Retrieve test data from CSV file");
  let test_data = dataset::get_test(&device)?;

  println!("test data shape: {:?}", test_data.shape());
  println!("test data dims: {:?}", test_data.dims());

  let net = neural::Network::new(vec![784, 32, 32, 10], &device)?;
  net.print();

  let result = net.forward(&train_data)?;
  println!("result shape: {:?}", result.shape());

  Ok(())
}
