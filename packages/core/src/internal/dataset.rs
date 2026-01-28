use candle_core::{DType, Device, Tensor};
use crate::internal::error::{Error, Result};
use csv::ReaderBuilder;
use std::fs::File;
use std::path::Path;

const MAX_NB_LINE: usize = 0;

fn read_csv<P: AsRef<Path>>(
    path: P,
    is_training: bool,
) -> Result<(usize, usize, Vec<f64>, Vec<f64>)> {
    let file = File::open(path)?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);

    let mut data: Vec<f64> = Vec::new();
    let mut labels: Vec<f64> = Vec::new();
    let mut nb_rows = 0usize;
    let mut nb_cols = None::<usize>;

    for result in reader.records() {
        let record = result.map_err(|e| Error::Msg(e.to_string()))?;
        if nb_cols.is_none() {
            nb_cols = Some(record.len());
        } else if nb_cols != Some(record.len()) {
            return Err(Error::Msg(
                "Inconsistent number of columns in CSV".to_string(),
            ));
        }

        let mut is_first_col = is_training;
        for cell in record.iter() {
            let value: f64 = cell
                .parse()
                .map_err(|e| Error::Msg(format!("Error parsing field {}: {}", cell, e)))?;

            if is_first_col {
                labels.push(value);
                is_first_col = false;
            } else {
                data.push(value);
            }
        }

        nb_rows += 1;
        if MAX_NB_LINE == nb_rows {
            break;
        }
    }

    let nb_cols = nb_cols.unwrap_or(0);

    Ok((nb_rows, nb_cols, data, labels))
}

fn create_tensor_data(
    nb_rows: usize,
    nb_cols: usize,
    data: Vec<f64>,
    device: &Device,
) -> candle_core::Result<Tensor> {
    Tensor::from_vec(data, (nb_cols, nb_rows), device)?.to_dtype(DType::F64)
}

fn create_tensor_expected(data: Vec<f64>, device: &Device) -> candle_core::Result<Tensor> {
    let max = 10;
    let mut vec: Vec<f64> = vec![0f64; data.len() * max];
    for (idx, value) in data.iter().enumerate() {
        vec[idx * max + (*value as usize)] = 1f64;
    }

    Tensor::from_vec(vec, (max, data.len()), device)?.to_dtype(DType::F64)
}

pub fn get_root_dir() -> &'static Path {
    let root = env!("CARGO_MANIFEST_DIR");
    let mut dir = Path::new(root);
    for _ in 0..2 {
        dir = dir.parent().expect("Failed to read parent directory");
    }

    dir
}

fn normalize(data: &[f64], min: f64, max: f64) -> Vec<f64> {
    data.iter()
        .map(|value| (*value - min) / (max - min))
        .collect()
}

pub fn get_train(device: &Device) -> Result<(Tensor, Tensor)> {
    let path = get_root_dir().join("datasets/train.csv");
    let (nb_rows, nb_cols, data, labels) = read_csv(path, true)?;
    let data = normalize(&data, 0f64, 255f64);

    Ok((
        create_tensor_data(nb_rows, nb_cols - 1, data, device)?,
        create_tensor_expected(labels, device)?,
    ))
}

// pub fn get_test(device: &Device) -> Result<Tensor> {
//   let path = get_root_dir().join("datasets/test.csv");
//   let (nb_rows, nb_cols, data, _) = read_csv(path, false)?;
//   let data = normalize(&data, 0f64, 255f64);
//
//   create_tensor_data(nb_rows, nb_cols, data, device)
// }
