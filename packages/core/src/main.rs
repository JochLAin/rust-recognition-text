use candle_core::Device;

mod dataset;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let device = {
    #[cfg(feature = "cuda")]
    { Device::new_cuda(0)? }
    #[cfg(not(feature = "cuda"))]
    { Device::Cpu }
  };

  let (train_data, train_label) = dataset::get_train(&device)?;
  let test_data = dataset::get_test(&device)?;
  println!("train data shape: {:?}", train_data.shape());
  println!("train labels shape: {:?}", train_label.shape());
  println!("test data shape: {:?}", test_data.shape());

  Ok(())
}
