use candle_core::Device;
use chrono::Local;
use crate::internal::dataset::get_root_dir;
use crate::internal::error::Error;
use crate::logger::Logger;
use crate::neural::{LayerData, Model, Network};
use std::result::Result;

pub fn save(net: &Network) -> Result<(), Error> {
    let mut layers: Vec<LayerData> = Vec::new();
    for layer in net.layers.iter().as_slice() {
        layers.push(LayerData {
            idx: layer.idx,
            activation: layer.activation,
            biases: layer
                .biases
                .to_vec2::<f64>()?
                .into_iter()
                .flatten()
                .collect(),
            shape: layer.weights.dims2()?,
            weights: layer
                .weights
                .to_vec2::<f64>()?
                .into_iter()
                .flatten()
                .collect(),
        });
    }

    let model = Model {
        dimensions: net.dimensions.clone(),
        layers,
    };

    let json = serde_json::to_string(verify(&model)?)?;
    let path = get_root_dir().join(format!("datasets/models/{}.json", Local::now().format("%Y%m%d%H%M%S%f")));
    println!("Saving model to {}", path.display());
    std::fs::write(path, &json)?;

    let path = get_root_dir().join("datasets/model.json");
    println!("Saving model to {}", path.display());
    std::fs::write(path, &json)?;

    Ok(())
}

pub fn load(logger: &Logger, device: &Device) -> Result<Network, Error> {
    logger.log("Loading model from disk");

    let path = get_root_dir().join("datasets/model.json");
    let json = std::fs::read_to_string(path)?;
    let model: Model = serde_json::from_str(&json)?;
    let net = Network::load(logger, verify(&model)?, device)?;

    logger.log("Model loaded from disk");

    Ok(net)
}

fn verify(model: &Model) -> Result<&Model, Error> {
    if 0 == model.layers.len() {
        if model.dimensions.len() > 0 {
            Err(Error::from("Model dimensions are not empty but layers are"))?
        }
    } else {
        let mut dimensions: Vec<usize> = vec![model.layers[0].shape.1];
        for layer in model.layers.iter().as_slice() {
            dimensions.push(layer.shape.0);
        }

        if dimensions != model.dimensions {
            Err(Error::from(
                "Model dimensions are not equal to layers dimensions",
            ))?
        }
    }

    Ok(model)
}
