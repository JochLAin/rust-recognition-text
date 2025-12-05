use candle_core::{DType, Device, Error, Result, Tensor};
use candle_nn::loss::cross_entropy;
use candle_nn::ops::{sigmoid, softmax};
use std::cmp::max;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    ReLU,
    Softmax,
    Sigmoid,
}

pub struct Layer {
    idx: usize,
    weights: Tensor,
    biases: Tensor,
    activation: Activation,
    debug: bool,
}

impl Layer {
    fn create(
        idx: usize,
        fan_in: usize,
        fan_out: usize,
        activation: Activation,
        device: &Device,
    ) -> Result<Self> {
        let std = match activation {
            Activation::ReLU => (2f64 / fan_in as f64).sqrt(),
            Activation::Softmax => (2f64 / (fan_in + fan_out) as f64).sqrt(),
            Activation::Sigmoid => (2f64 / (fan_in + fan_out) as f64).sqrt(),
        };

        let biases = Tensor::zeros((fan_out, 1), DType::F64, device)?;
        let weights = Tensor::randn(0f64, std, (fan_out, fan_in), device)?;

        let layer = Self {
            idx,
            weights,
            biases,
            activation,
            debug: true,
        };

        println!("w{}: {:?}", layer.idx, layer.weights);
        println!("b{}: {:?}", layer.idx, layer.biases);

        Ok(layer)
    }

    fn backward(
        &mut self,
        dz: &Tensor,
        a: &Tensor,
        m: f64,
        learning_rate: f64,
        is_last: bool,
    ) -> Result<Tensor> {
        if self.debug {
            println!("dz{}: {:?}", self.idx, dz);
        }

        let dw = self.get_weight_derivative(dz, a, m)?;
        let db = self.get_bias_derivative(dz)?;
        if self.debug {
            println!("dw{}: {:?}", self.idx, dw);
            println!("db{}: {:?}", self.idx, db);
        }

        let dz = if is_last {
            Tensor::zeros(dz.shape(), DType::F64, dz.device())?
        } else {
            self.weights
                .t()?
                .matmul(&dz)?
                .mul(&self.get_activation_derivative(a)?)?
        };

        self.update(&dw, &db, learning_rate)?;

        Ok(dz)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let z = self.calculate_result(x)?;
        if self.debug {
            println!("z{}: {:?}", self.idx, z);
        }

        let a = self.calculate_activation(&z)?;
        if self.debug {
            println!("a{}: {:?}", self.idx, a);
        }

        Ok(a)
    }

    fn calculate_activation(&self, z: &Tensor) -> Result<Tensor> {
        match self.activation {
            Activation::ReLU => z.relu(),
            Activation::Softmax => softmax(&z, 1),
            Activation::Sigmoid => sigmoid(&z),
        }
    }

    fn calculate_result(&self, x: &Tensor) -> Result<Tensor> {
        self.weights.matmul(&x)?.broadcast_add(&self.biases)
    }

    fn get_activation_derivative(&self, a: &Tensor) -> Result<Tensor> {
        match self.activation {
            Activation::ReLU => a.gt(0f64),
            Activation::Sigmoid => a.mul(&Tensor::ones_like(a)?.sub(a)?),
            _ => Ok(Tensor::ones_like(a)?),
        }
    }

    fn get_bias_derivative(&self, dz: &Tensor) -> Result<Tensor> {
        // println!(" Calculate bias derivative");
        dz.sum_keepdim(1)
    }

    fn get_weight_derivative(&self, dz: &Tensor, a: &Tensor, m: f64) -> Result<Tensor> {
        // println!(" Calculate weight derivative, dz {:?}, a {:?}", dz.shape(), a.shape());
        let m = Tensor::new(1f64 / m, dz.device())?;
        m.broadcast_mul(&dz.matmul(&a.t()?)?)
    }

    fn update(&mut self, dw: &Tensor, db: &Tensor, learning_rate: f64) -> Result<()> {
        self.weights = self
            .weights
            .sub(&dw.broadcast_mul(&Tensor::new(learning_rate, dw.device())?)?)?;
        self.biases = self
            .biases
            .sub(&db.broadcast_mul(&Tensor::new(learning_rate, db.device())?)?)?;

        Ok(())
    }
}

pub struct Network {
    pub layers: Vec<Layer>,
    pub dimensions: Vec<usize>,
    pub activations: Vec<Activation>,
    pub device: Device,
    pub built: bool,
    debug: bool,
}

impl Network {
    pub fn new(fan_in: usize, fan_out: usize, activation: Activation, device: &Device) -> Self {
        Self {
            dimensions: vec![fan_in, fan_out],
            activations: vec![activation],
            device: device.clone(),
            layers: vec![],
            built: false,
            debug: true,
        }
    }

    pub fn add_layer(&mut self, nb_neuron: usize, activation: Activation) -> &mut Self {
        self.dimensions.insert(self.dimensions.len() - 1, nb_neuron);
        self.activations
            .insert(self.activations.len() - 1, activation);

        self
    }

    pub fn build(&mut self) -> Result<&mut Self> {
        let len = self.dimensions.len();
        println!("🎉 Setup network with {} layers", len - 1);

        for idx in 1..len {
            let fan_in = self.dimensions[idx - 1];
            let fan_out = self.dimensions[idx];
            let activation = self.activations[idx - 1];
            self.layers.push(Layer::create(
                idx,
                fan_in,
                fan_out,
                activation,
                &self.device,
            )?);
        }

        self.built = true;
        Ok(self)
    }

    fn backward(&mut self, y: &Tensor, a: Vec<Tensor>, learning_rate: f64) -> Result<Tensor> {
        let len = self.dimensions.len() - 1;
        if self.debug {
            println!("⏮️ Backward network on {} layers", len);
            println!("y: {:?}", y);
        }

        let m = y.dim(1)? as f64;
        if self.debug {
            println!("m: {:?}", m);
        }
        let mut dz = a[a.len() - 1].sub(&y)?;
        for idx in (1..len).rev() {
            if self.debug {
                println!("- Backward layer {}, dz {:?}", idx - 1, dz.shape());
            }
            dz = self.layers[idx].backward(&dz, &a[idx - 1], m, learning_rate, idx == 1)?;
        }

        Ok(dz)
    }

    fn forward(&self, input: &Tensor) -> Result<Vec<Tensor>> {
        if !self.built {
            Err(Error::Msg("Network is not built".into()))?
        }

        let mut x = self.flatten(&input)?;
        if self.debug {
            println!("x: {:?}", x.shape());
        }

        let len = self.dimensions.len() - 1;
        if self.debug {
            println!("⏩ Forward network on {} layers", len);
        }

        let mut acs: Vec<Tensor> = Vec::new();
        for idx in 1..(len + 1) {
            if self.debug {
                println!("- Forward layer {}", idx);
            }

            x = self.layers[idx - 1].forward(&x)?;
            acs.push(x.clone());
        }

        Ok(acs)
    }

    fn flatten(&self, input: &Tensor) -> Result<Tensor> {
        let dimensions = input.dims();
        let (batch, flattened) = match dimensions {
            [batch, nb_row, nb_col, nb_channel] => (*batch, nb_row * nb_col * nb_channel),
            [batch, nb_row, nb_col] => (*batch, nb_row * nb_col),
            [batch, nb_cell] => (*batch, *nb_cell),
            _ => {
                return Err(Error::Msg(
                    format!("Unexpected tensor dimensions: {:?}", dimensions).into(),
                ));
            }
        };

        input.reshape((batch, flattened))
    }

    fn get_accuracy(&self, x: &Tensor, y: &Tensor) -> Result<f64> {
        let x = self.predict(x)?;
        let y = y.argmax(0)?.to_dtype(DType::F64)?;
        let batch = y.dim(0)? as f64;

        Ok(x.eq(&y)?
            .to_dtype(DType::F64)?
            .sum_all()?
            .to_scalar::<f64>()?
            / batch)
    }

    fn get_loss(&self, a: &Tensor, y: &Tensor) -> Result<f64> {
        cross_entropy(&a, &y.t()?.argmax(0)?)?
            .to_dtype(DType::F64)?
            .to_scalar()
    }

    pub fn predict(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)?
            .last()
            .unwrap()
            .argmax(0)?
            .to_dtype(DType::F64)
    }

    pub fn train(
        &mut self,
        x: &Tensor,
        y: &Tensor,
        learning_rate: f64,
        nb_iter: usize,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let step = max(1, nb_iter / 100);
        let quiet = !self.debug;
        println!(
            "Training network with {} iterations, save step each {}",
            nb_iter, step
        );

        let mut losses: Vec<f64> = Vec::new();
        let mut accuracies: Vec<f64> = Vec::new();
        for idx in 0..nb_iter {
            println!("## Iteration {}", idx);

            let a = self.forward(x)?;
            self.backward(y, a.clone(), learning_rate)?;

            if 0 == nb_iter % step {
                self.quiet(true);
                let loss = self.get_loss(&a[a.len() - 1], y)?;
                println!("------ loss: {}", loss);
                losses.push(loss);

                let acc = self.get_accuracy(x, y)?;
                println!("------ acc: {}", acc);
                accuracies.push(acc);
                self.quiet(quiet);
            }
        }

        Ok((losses, accuracies))
    }

    pub fn quiet(&mut self, value: bool) -> &mut Self {
        self.debug = !value;
        for layer in &mut self.layers {
            layer.debug = !value;
        }

        self
    }
}
