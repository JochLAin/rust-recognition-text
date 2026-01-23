use crate::logger::Logger;
use crate::error::{Error, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::loss::cross_entropy;
use candle_nn::ops::{sigmoid, softmax};
use serde::{Deserialize, Serialize};
use std::cmp::max;

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    ReLU,
    Softmax,
    Sigmoid,
    None,
}

#[derive(Serialize, Deserialize)]
pub struct Model {
    pub dimensions: Vec<usize>,
    pub layers: Vec<LayerData>,
}

#[derive(Serialize, Deserialize)]
pub struct LayerData {
    pub idx: usize,
    pub activation: Activation,
    pub biases: Vec<f64>,
    pub weights: Vec<f64>,
    pub shape: (usize, usize),
}

struct Computer;
impl Computer {
    fn assert_shape(msg: String, expected: &[usize], result: &[usize]) -> Result<()> {
        if expected != result {
            Err(format!("{:?}, expected {:?} got {:?}", msg, expected, result))?
        }
        Ok(())
    }

    fn assert_activation(x: &Tensor, w: &Tensor, a: &Tensor) -> Result<()> {
        Computer::assert_shape(
            "Shape mismatch on forward activation (a)".into(),
            [w.dim(0)?, x.dim(1)?].as_slice(),
            a.dims(),
        )
    }

    fn assert_derivative_activation(a: &Tensor, da: &Tensor) -> Result<()> {
        Computer::assert_shape(
            "Shape mismatch on backward derivative activation (da)".into(),
            a.dims(),
            da.dims(),
        )
    }

    fn assert_bias(b: &Tensor, r: &Tensor) -> Result<()> {
        Computer::assert_shape(
            "Shape mismatch on backward bias (b)".into(),
            b.dims(),
            r.dims(),
        )
    }

    fn assert_derivative_bias(b: &Tensor, db: &Tensor) -> Result<()> {
        Computer::assert_shape(
            "Shape mismatch on backward derivative bias (db)".into(),
            b.dims(),
            db.dims(),
        )
    }

    fn assert_result(x: &Tensor, w: &Tensor, z: &Tensor) -> Result<()> {
        Computer::assert_shape(
            "Shape mismatch on forward result (z)".into(),
            [w.dim(0)?, x.dim(1)?].as_slice(),
            z.dims(),
        )
    }

    fn assert_derivative_result(dz: &Tensor) -> Result<()> {

        Ok(())
    }

    fn assert_weight(w: &Tensor, r: &Tensor) -> Result<()> {
        Computer::assert_shape(
            "Shape mismatch on backward weight (w)".into(),
            w.dims(),
            r.dims(),
        )
    }

    fn assert_derivative_weight(w: &Tensor, dw: &Tensor) -> Result<()> {
        Computer::assert_shape(
            "Shape mismatch on backward derivative weight (dw)".into(),
            w.dims(),
            dw.dims(),
        )
    }

    fn calculate_activation(method: Activation, z: &Tensor) -> candle_core::Result<Tensor> {
        match method {
            Activation::ReLU => z.relu(),
            Activation::Softmax => softmax(&z, 1),
            Activation::Sigmoid => sigmoid(&z),
            Activation::None => Ok(z.clone()),
        }
    }

    fn calculate_bias(b: &Tensor, db: &Tensor, lr: f64) -> candle_core::Result<Tensor> {
        let lr = &Tensor::new(lr, db.device())?;
        b.sub(&lr.broadcast_mul(db)?)
    }

    fn calculate_result(x: &Tensor, w: &Tensor, b: &Tensor) -> candle_core::Result<Tensor> {
        w.matmul(&x)?.broadcast_add(b)
    }

    fn calculate_weight(w: &Tensor, dw: &Tensor, lr: f64) -> candle_core::Result<Tensor> {
        let lr = &Tensor::new(lr, dw.device())?;
        w.sub(&lr.broadcast_mul(dw)?)
    }

    fn drift_activation(method: Activation, a: &Tensor) -> candle_core::Result<Tensor> {
        match method {
            Activation::ReLU => a.gt(0f64)?.to_dtype(DType::F64),
            Activation::Sigmoid => a.mul(&Tensor::ones_like(a)?.sub(a)?),
            Activation::Softmax => Tensor::ones_like(a),
            Activation::None => Tensor::ones_like(a),
        }
    }

    fn drift_bias(dz: &Tensor, m: f64) -> candle_core::Result<Tensor> {
        let m = Tensor::new(1f64 / m, dz.device())?;
        dz.sum_keepdim(1)?.broadcast_mul(&m)
    }

    fn drift_weight(dz: &Tensor, a: &Tensor, m: f64) -> candle_core::Result<Tensor> {
        let m = Tensor::new(1f64 / m, dz.device())?;
        dz.matmul(&a.t()?)?.broadcast_mul(&m)
    }

    fn drift_result(w: &Tensor, dz: &Tensor, da: &Tensor) -> candle_core::Result<Tensor> {
        w.t()?.matmul(&dz)?.mul(&da)
    }

    pub fn get_a(logger: &Logger, method: Activation, x: &Tensor, w: &Tensor, z: &Tensor, idx: usize) -> Result<Tensor> {
        let a = Computer::calculate_activation(method, z)?;
        logger.debug(format!("a{}: {:?}", idx, a));
        Computer::assert_activation(x, w, &a)?;
        Ok(a)
    }

    pub fn get_b(logger: &Logger, b: &Tensor, db: &Tensor, lr: f64, idx: usize) -> Result<Tensor> {
        let r = Computer::calculate_bias(b, db, lr)?;
        logger.debug(format!("b{}: {:?}", idx, b));
        Computer::assert_bias(b, &r)?;
        Ok(r)
    }

    pub fn get_w(logger: &Logger, w: &Tensor, dw: &Tensor, lr: f64, idx: usize) -> Result<Tensor> {
        let r = Computer::calculate_weight(w, dw, lr)?;
        logger.debug(format!("w{}: {:?}", idx, r));
        Computer::assert_weight(w, &r)?;
        Ok(r)
    }

    pub fn get_z(logger: &Logger, x: &Tensor, w: &Tensor, b: &Tensor, idx: usize) -> Result<Tensor> {
        let z = Computer::calculate_result(x, w, b)?;
        logger.debug(format!("z{}: {:?}", idx, z));
        Computer::assert_result(x, w, &z)?;
        Ok(z)
    }

    pub fn get_da(logger: &Logger, method: Activation, a: &Tensor, idx: usize) -> Result<Tensor> {
        let da = Computer::drift_activation(method, a)?;
        logger.debug(format!("da{}: {:?}", idx, da));
        Computer::assert_derivative_activation(a, &da)?;
        Ok(da)
    }

    pub fn get_db(logger: &Logger, b: &Tensor, dz: &Tensor, m: f64, idx: usize) -> Result<Tensor> {
        let db = Computer::drift_bias(dz, m)?;
        logger.debug(format!("db{}: {:?}", idx, db));
        Computer::assert_derivative_bias(&b, &db)?;
        Ok(db)
    }

    pub fn get_dw(logger: &Logger, w: &Tensor, dz: &Tensor, a: &Tensor, m: f64, idx: usize) -> Result<Tensor> {
        let r = Computer::drift_weight(dz, a, m)?;
        logger.debug(format!("dw{}: {:?}", idx, r));
        Computer::assert_derivative_weight(w, &r)?;
        Ok(r)
    }

    pub fn get_dz(logger: &Logger, w: &Tensor, dz: &Tensor, da: &Tensor, idx: usize) -> Result<Tensor> {
        let dz = Computer::drift_result(w, dz, da)?;
        logger.debug(format!("dz{}: {:?}", idx, dz));
        Ok(dz)
    }

    pub fn get_dz_next(logger: &Logger, method: Activation, a: &Tensor, w: &Tensor, dz: &Tensor, idx: usize) -> Result<Tensor> {
        let da = Computer::get_da(logger, method, a, idx)?;
        Computer::get_dz(logger, w, dz, &da, idx)
    }
}

#[derive(Debug, Clone)]
pub struct Layer {
    pub idx: usize,
    pub weights: Tensor,
    pub biases: Tensor,
    pub activation: Activation,
}

impl Layer {
    pub fn new(
        logger: &Logger,
        idx: usize,
        fan_in: usize,
        fan_out: usize,
        activation: Activation,
        device: &Device,
        weights: Option<Vec<f64>>,
        biases: Option<Vec<f64>>,
    ) -> Result<Self> {
        let biases = match biases {
            Some(biases) => Tensor::from_vec(biases, (fan_out, 1), device)?,
            None => Tensor::zeros((fan_out, 1), DType::F64, device)?,
        };

        let weights = match weights {
            Some(weights) => Tensor::from_vec(weights, (fan_out, fan_in), device)?,
            None => Tensor::randn(
                0f64,
                match activation {
                    Activation::ReLU => (2f64 / fan_in as f64).sqrt(),
                    _ => (2f64 / (fan_in + fan_out) as f64).sqrt(),
                },
                (fan_out, fan_in),
                device,
            )?,
        };

        let layer = Self {
            idx,
            weights,
            biases,
            activation,
        };

        logger.debug(format!("w{}: {:?}", layer.idx, layer.weights));
        logger.debug(format!("b{}: {:?}", layer.idx, layer.biases));

        Ok(layer)
    }

    pub fn backward(
        &mut self,
        logger: &Logger,
        dz: &Tensor,
        a: &Tensor,
        m: f64,
    ) -> Result<(Tensor, Tensor)> {
        Ok((
            Computer::get_dw(logger, &self.weights, dz, a, m, self.idx)?,
            Computer::get_db(logger, &self.biases, dz, m, self.idx)?,
        ))
    }

    pub fn forward(&self, logger: &Logger, x: &Tensor) -> Result<Tensor> {
        let z = Computer::get_z(logger, x, &self.weights, &self.biases, self.idx)?;
        Computer::get_a(logger, self.activation, x, &self.weights, &z, self.idx)
    }

    pub fn update(&mut self, logger: &Logger, dw: &Tensor, db: &Tensor, lr: f64) -> Result<()> {
        self.weights = Computer::get_w(logger, &self.weights, dw, lr, self.idx)?;
        self.biases = Computer::get_b(logger, &self.biases, db, lr, self.idx)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Network {
    pub layers: Vec<Layer>,
    pub dimensions: Vec<usize>,
    pub activations: Vec<Activation>,
    pub device: Device,
    built: bool,
}

impl Network {
    pub fn new(fan_in: usize, fan_out: usize, activation: Activation, device: &Device) -> Self {
        Self {
            dimensions: vec![fan_in, fan_out],
            activations: vec![activation],
            device: device.clone(),
            layers: vec![],
            built: false,
        }
    }

    pub fn load(logger: &Logger, model: &Model, device: &Device) -> Result<Self> {
        let activations = model
            .layers
            .iter()
            .map(|layer| Activation::from(layer.activation))
            .collect();

        let net = Self {
            activations,
            dimensions: model.dimensions.clone(),
            device: device.clone(),
            built: true,
            layers: vec![],
        };

        let layers: Vec<Layer> = model
            .layers
            .iter()
            .map(|layer| {
                Layer::new(
                    logger,
                    layer.idx,
                    layer.shape.1,
                    layer.shape.0,
                    layer.activation,
                    &device.clone(),
                    Some(layer.weights.clone()),
                    Some(layer.biases.clone()),
                )
                .unwrap()
            })
            .collect();

        Ok(net.with_layers(layers)?)
    }

    pub fn add_layer(&mut self, nb_neuron: usize, activation: Activation) -> Result<&mut Self> {
        self.verify_network_built(false)?;
        self.dimensions.insert(self.dimensions.len() - 1, nb_neuron);
        self.activations
            .insert(self.activations.len() - 1, activation);

        Ok(self)
    }

    pub fn with_layers(self, layers: Vec<Layer>) -> Result<Self> {
        let mut net = self.clone();
        net.layers = layers.clone();
        net.verify_layers()?;
        net.built = true;
        Ok(net)
    }

    pub fn build(&mut self, logger: &Logger) -> Result<Self> {
        self.verify_network_built(false)?;
        let len = self.dimensions.len();
        logger.info(format!("🎉 Setup network with {} layers", len - 1));

        for idx in 1..len {
            let fan_in = self.dimensions[idx - 1];
            let fan_out = self.dimensions[idx];
            let activation = self.activations[idx - 1];
            self.layers.push(Layer::new(
                logger,
                idx,
                fan_in,
                fan_out,
                activation,
                &self.device,
                None,
                None,
            )?);
        }

        self.built = true;
        self.verify_layers()?;

        Ok(self.clone())
    }

    pub fn backward(
        &mut self,
        logger: &Logger,
        y: &Tensor,
        a: &Vec<Tensor>,
    ) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        self.verify_network_built(true)?;
        let len = self.dimensions.len() - 1;
        logger.info("⏮️ Backward network");
        logger.debug(format!("y: {:?}", y));

        let m = y.dim(1)? as f64;
        logger.debug(format!("m: {:?}", m));

        let mut dws: Vec<Tensor> = Vec::new();
        let mut dbs: Vec<Tensor> = Vec::new();

        let mut dz = a[a.len() - 1].sub(&y)?;
        for idx in (0..len).rev() {
            logger.info(format!("- Backward layer {} ￫ {}", idx + 1, idx));
            let (dw, db) = self.layers[idx].backward(logger, &dz, &a[idx], m)?;
            dws.push(dw);
            dbs.push(db);

            if idx != 0 {
                dz = Computer::get_dz_next(logger, self.layers[idx].activation, &a[idx], &self.layers[idx].weights, &dz, idx + 1)?
            }
        }

        dws.reverse();
        dbs.reverse();
        Ok((dws, dbs))
    }

    pub fn forward(&self, logger: &Logger, input: &Tensor) -> Result<Vec<Tensor>> {
        self.verify_network_built(true)?;
        if !self.built {
            Err(Error::Msg("Network is not built".into()))?
        }

        let mut x = self.normalize(&input)?;
        logger.debug(format!("x: {:?}", x));

        let len = self.dimensions.len() - 1;
        logger.info("⏩ Forward network");

        let mut acs: Vec<Tensor> = vec![x.clone()];
        for idx in 0..len {
            logger.info(format!("- Forward layer {} ￫ {}", idx, idx + 1));

            x = self.layers[idx].forward(logger, &x)?;
            acs.push(x.clone());
        }

        Ok(acs)
    }

    pub fn predict(&self, logger: &Logger, x: &Tensor) -> Result<Tensor> {
        self.verify_network_built(true)?;
        Ok(
            self.forward(logger, x)?
                .last()
                .unwrap()
                .argmax(0)?
                .to_dtype(DType::F64)?
        )
    }

    pub fn train(
        &mut self,
        logger: &Logger,
        x: &Tensor,
        y: &Tensor,
        learning_rate: f64,
        nb_iter: usize,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        self.verify_network_built(true)?;
        let step = max(1, nb_iter / 100);
        logger.log(format!(
            "Training network with {} iterations, save step each {}",
            nb_iter, step
        ));

        let mut losses: Vec<f64> = Vec::new();
        let mut accuracies: Vec<f64> = Vec::new();
        logger.progress_bar.set_position(0);
        for idx in 0..nb_iter {
            logger.debug(format!("## Iteration {}", idx));

            let a = self.forward(logger, x)?;
            let (dws, dbs) = self.backward(logger, y, &a)?;
            self.update(logger, dws, dbs, learning_rate)?;

            if 0 == nb_iter % step {
                let loss = self.calculate_loss(&a[a.len() - 1], y)?;
                logger.dump(format!("------ loss: {}", loss));
                losses.push(loss);

                let acc = self.calculate_accuracy(&a[a.len() - 1], y)?;
                logger.dump(format!("------ acc: {}", acc));
                accuracies.push(acc);
            }

            logger.progress_bar.inc(1);
        }
        logger.progress_bar.finish();

        Ok((losses, accuracies))
    }

    pub fn update(&mut self, logger: &Logger, dws: Vec<Tensor>, dbs: Vec<Tensor>, learning_rate: f64) -> Result<()> {
        self.verify_network_built(true)?;
        let len = self.layers.len();
        logger.info("🔄 Update network");
        for idx in 0..len {
            logger.info(format!("- Update layer {}", idx));
            self.layers[idx].update(logger, &dws[idx], &dbs[idx], learning_rate)?;
        }
        Ok(())
    }

    fn calculate_accuracy(&self, a: &Tensor, y: &Tensor) -> Result<f64> {
        let y = y.argmax(0)?.to_dtype(DType::F64)?;
        let batch = y.dim(0)? as f64;
        let r = a.eq(&y)?
                .to_dtype(DType::F64)?
                .sum_all()?
                .to_scalar::<f64>()?;

        Ok(r / batch)
    }

    fn calculate_loss(&self, a: &Tensor, y: &Tensor) -> candle_core::Result<f64> {
        cross_entropy(&a, &y.t()?.argmax(0)?)?
                .to_dtype(DType::F64)?
                .to_scalar()
    }

    fn normalize(&self, input: &Tensor) -> Result<Tensor> {
        let dimensions = input.dims();
        let (batch, flattened) = match dimensions {
            [batch, nb_row, nb_col, nb_channel] => (*batch, nb_row * nb_col * nb_channel),
            [batch, nb_row, nb_col] => (*batch, nb_row * nb_col),
            [batch, nb_cell] => (*batch, *nb_cell),
            _ => Err(Error::Msg(
                format!("Unexpected tensor dimensions: {:?}", dimensions).into(),
            ))?
        };

        Ok(input.reshape((batch, flattened))?)
    }

    fn verify_network_built(&self, is_built: bool) -> Result<()> {
        if self.built != is_built {
            if is_built {
                Err("Network is already built")?
            }
            Err("Network is not built")?
        }

        Ok(())
    }

    fn verify_layers(&self) -> Result<()> {
        let len = self.layers.len();
        if self.dimensions.len() != len + 1 {
            Err("Layers count mismatch")?
        }

        for idx in 0..len {
            Computer::assert_shape(
                format!("Weights shape on layer {} shape mismatch", idx),
                [self.dimensions[idx + 1], self.dimensions[idx]].as_slice(),
                self.layers[idx].weights.dims(),
            )?;

            Computer::assert_shape(
                format!("Biases shape on layer {} shape mismatch", idx),
                [self.dimensions[idx + 1], 1].as_slice(),
                self.layers[idx].biases.dims(),
            )?;
        }

        Ok(())
    }
}
