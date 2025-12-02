use candle_core::{Device, DType, Error, Result, Tensor};
use csv::ReaderBuilder;
use std::fs::File;
use std::path::Path;

fn read_csv<P: AsRef<Path>>(path: P, is_training: bool) -> Result<(usize, usize, Vec<u8>, Vec<u8>)> {
  let file = File::open(path)?;
  let mut reader = ReaderBuilder::new()
    .has_headers(true)
    .from_reader(file);

  let mut data: Vec<u8> = Vec::new();
  let mut labels: Vec<u8> = Vec::new();
  let mut nb_rows = 0usize;
  let mut nb_cols = None::<usize>;

  for result in reader.records() {
    let record = result.map_err(|e| Error::Msg(e.to_string()))?;
    if nb_cols.is_none() {
      nb_cols = Some(record.len());
    } else if nb_cols != Some(record.len()) {
      return Err(
        Error::Msg("Inconsistent number of columns in CSV".to_string()
      ));
    }

    let mut is_first_line = is_training;
    for cell in record.iter() {
      let value: u8 = cell.parse().map_err(|e| {
        Error::Msg(format!("Error parsing field {}: {}", cell, e))
      })?;

      if !is_first_line {
        is_first_line = false;
        labels.push(value);
      } else {
        data.push(value);
      }
    }

    nb_rows += 1;
  }

  let nb_cols = nb_cols.unwrap_or(0);

  Ok((nb_rows, nb_cols, data, labels))

  // let shape = (nb_rows, nb_cols);
  //
  // Tensor::from_vec(data, shape, device)?
  //   .to_dtype(DType::U8)
}

fn create_tensor(nb_rows: usize, nb_cols: usize, data: Vec<u8>, device: &Device) -> Result<Tensor> {
  let side =(nb_cols as f32).sqrt() as usize;

  if side * side != nb_cols {
    return Err(Error::Msg(
      format!("Column count {nb_cols} is not a perfect square").into()
    ));
  }

  Tensor::from_vec(data, (nb_rows, nb_cols), device)?
    .reshape((nb_rows, side, side))?
    .unsqueeze(1)?
    .to_dtype(DType::U8)
}

fn get_root_dir() -> &'static Path {
  let root = env!("CARGO_MANIFEST_DIR");
  let mut dir = Path::new(root);
  for _ in 0..2 {
    dir = dir.parent().expect("Failed to read parent directory");
  }

  dir
}

pub fn get_train(device: &Device) -> Result<(Tensor, Tensor)> {
  // 1258 lignes avec 1 ligne de header
  // 784 colonnes
  let path = get_root_dir().join("datasets/train.csv");
  println!("path: {}", path.to_str().expect("Failed to read path for train data"));

  let (nb_rows, nb_cols, data, labels) = read_csv(path, true)?;

  Ok((
    create_tensor(nb_rows, nb_cols - 1, data, device)?,
    Tensor::from_vec(labels, (nb_rows,), device)?.to_dtype(DType::U8).expect("Failed to create labels tensor")
  ))
}

pub fn get_test(device: &Device) -> Result<Tensor> {
  // 1258 lignes avec 1 ligne de header
  // 784 colonnes
  let path = get_root_dir().join("datasets/test.csv");
  let (nb_rows, nb_cols, data, _) = read_csv(path, false)?;

  create_tensor(nb_rows, nb_cols, data, device)
}
