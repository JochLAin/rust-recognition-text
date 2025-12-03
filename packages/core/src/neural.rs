use candle_core::{Device, DType, Error, Result, Tensor};
use candle_nn::ops::{sigmoid, softmax};

enum Activation {
  ReLU,
  Softmax,
  Sigmoid,
}

struct Layer {
  idx: usize,
  weights: Tensor,
  biases: Tensor,
  activation_method: Option<Activation>,
  activation: Option<Tensor>,
  result: Option<Tensor>,
}

impl Layer {
  fn new_hidden(idx: usize, fan_in: usize, fan_out: usize, activation_method: Option<Activation>, device: &Device) -> Result<Self> {
    let std = (2f64 / fan_in as f64).sqrt();
    Ok(Self {
      idx,
      weights: Tensor::randn(0f64, std, (fan_out, fan_in), device)?,
      biases: Tensor::zeros((fan_out,), DType::F64, device)?,
      activation_method,
      activation: None,
      result: None,
    })
  }

  fn new_end(idx: usize, fan_in: usize, fan_out: usize, device: &Device) -> Result<Self> {
    let std = (2f64 / (fan_in + fan_out) as f64).sqrt();
    Ok(Self {
      idx,
      weights: Tensor::randn(0f64, std, (fan_out, fan_in), device)?,
      biases: Tensor::zeros((fan_out,), DType::F64, device)?,
      activation_method: None,
      activation: None,
      result: None,
    })
  }

  fn backward(&mut self, dz: &Tensor, a: &Tensor, m: f64, is_last: bool) -> Result<Tensor> {
    let dw = self.get_derived_weight(dz, a, m)?;
    let db = self.get_derived_bias(dz)?;

    if is_last {
      Ok(Tensor::zeros(dz.shape(), DType::F64, dz.device())?)
    } else {
      self.get_back_result(dz, a)
    }
  }

  fn forward(&self, x: &Tensor) -> Result<Tensor> {
    self.get_activation(&self.get_result(x)?)
  }

  fn get_activation(&self, z: &Tensor) -> Result<Tensor> {
    match self.activation_method {
      Some(Activation::ReLU) => z.relu(),
      Some(Activation::Softmax) => softmax(&z, 1),
      Some(Activation::Sigmoid) => sigmoid(&z),
      None => Ok(z.clone()),
    }
  }

  fn get_derived_weight(&self, dz: &Tensor, a: &Tensor, m: f64) -> Result<Tensor> {
    let m = Tensor::new(1f64 / m, dz.device())?;
    dz.matmul(&a.t()?)?.broadcast_mul(&m)
  }

  fn get_derived_bias(&self, dz: &Tensor) -> Result<Tensor> {
    dz.sum(0)
  }

  fn get_result(&self, x: &Tensor) -> Result<Tensor> {
    x
      .matmul(&self.weights.t()?)?
      .broadcast_add(&self.biases)
  }

  fn get_back_result(&self, dz: &Tensor, a: &Tensor) -> Result<Tensor> {
    self.weights.t()?.matmul(&dz)?.mul(a)?.mul(&Tensor::new(1f64, dz.device())?.sub(&a)?)
  }
}

pub struct Network {
  layers: Vec<Layer>,
}

impl Network {
  pub fn new(dimensions: Vec<usize>, device: &Device) -> Result<Self> {
    let mut layers: Vec<Layer> = Vec::new();
    let len = dimensions.len();
    for idx in 1..len {
      let fan_in = dimensions[idx - 1];
      let fan_out = dimensions[idx];
      if idx == len - 1 {
        layers.push(Layer::new_end(idx, fan_in, fan_out, device)?);
      } else {
        layers.push(Layer::new_hidden(idx, fan_in, fan_out, Some(Activation::ReLU), device)?);
      }
    }

    Ok(Self { layers })
  }

  pub fn backward(&mut self, y: &Tensor, a: Vec<Tensor>) -> Result<Tensor> {
    let len = self.layers.len();
    let m = y.shape().dim(1)? as f64;

    let mut dz = a[a.len() - 1].sub(y)?;
    for idx in (1..len + 1).rev() {
      dz = self.layers[idx].backward(&dz, &a[idx - 1], m, idx == 1)?;
    }

    Ok(dz)
  }

  pub fn forward(&self, input: &Tensor) -> Result<Vec<Tensor>> {
    let dimensions = input.dims();
    let (batch, flattened) = match dimensions {
      [batch, nb_row, nb_col] => (*batch, nb_row * nb_col),
      [batch, nb_row, nb_col, nb_channel] => (*batch, nb_row * nb_col * nb_channel),
      _ => return Err(Error::Msg(
        format!("Unexpected tensor dimensions: {:?}", dimensions).into()
      ))
    };

    let mut x = input.reshape((batch, flattened))?;
    let mut activations: Vec<Tensor> = Vec::new();
    let len = self.layers.len();
    for idx in 0..len {
      x = self.layers[idx].forward(&x)?;
      activations.push(x.clone());
    }

    Ok(activations)
  }

  pub fn print(&self) {
    for layer in &self.layers {
      println!("{:?}", layer.weights.shape());
    }
  }
}
