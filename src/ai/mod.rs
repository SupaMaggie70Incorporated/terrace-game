use std::path::Path;

use burn::{backend::candle::CandleDevice, prelude::*, tensor::activation::softmax};
use nn::{Linear, LinearConfig, Relu};

mod net;
mod eval;
mod train;
mod train_loop;
pub mod data;

const TRAINING_DATA_FOLDER: &str = "/media/FastSSD/Databases/Terrace_Training/";

pub type BACKEND = burn::backend::Candle;
pub type AUTODIFF_BACKEND = burn::backend::Autodiff<burn::backend::Candle>;
pub const DEVICE: CandleDevice = burn::backend::candle::CandleDevice::Cuda(0);
/// 11 possible pieces including none, 8 possible heights. The heights are only necessary for conv nets, but we will keep them for the sake of simplicity
pub const INPUT_SIZE: usize = 64 * 19; 
pub const OUTPUT_SIZE: usize = 3;

pub struct NetworkInput<B: Backend> {
    /// Dims: [batch_size, rows, columns, channels(1 channel for each possible type of piece)]
    value: Tensor<B, 4, Bool>
}
pub struct NetworkOutput<B: Backend> {
    /// Dims: [batch_size, channels]
    ///
    /// 1 channel for each possible type of result, so 3 for win, loss, draw. Channels are normalized such that they add up to 1
    value: Tensor<B, 2, Float>
}
pub struct NetworkTarget<B: Backend> {
    /// Dims: [batch_size]
    /// Each value is the index of the correct option(0 for win, 1 for draw, 2 for loss)
    value: Tensor<B, 1, Int>
}

#[derive(Config, Debug)]
pub struct MlpConfig {
    layer_sizes: Vec<usize>,
}
impl MlpConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mlp<B> {
        let mut layers = Vec::new();
        assert!({
            let mut val = true;
            for size in &self.layer_sizes {
                if *size <= 0 {val = false;break}
            }
            val
        });
        if layers.len() > 0 {
            layers.push(LinearConfig::new(INPUT_SIZE, self.layer_sizes[0]).init(device));
            for i in 0..self.layer_sizes.len() - 1 {
                layers.push(LinearConfig::new(self.layer_sizes[i], self.layer_sizes[i + 1]).init(device));
            }
            layers.push(LinearConfig::new(self.layer_sizes[self.layer_sizes.len() - 1], OUTPUT_SIZE).init(device));
        } else {
            layers.push(LinearConfig::new(INPUT_SIZE, OUTPUT_SIZE).init(device));
        }
        Mlp {
            layers,
            activation: Relu::new()
        }
    }
}
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    layers: Vec<Linear<B>>,
    activation: Relu,
}
impl<B: Backend> Mlp<B> {
    pub fn forward(&self, input: NetworkInput<B>) -> NetworkOutput<B> {
        // We will reshape it to be a single dimension, as we are using a basic feedforward
        let [batch_size, x_width, y_height, channel_count] = input.value.shape().dims;
        let mut x = input.value.reshape([batch_size, x_width * y_height * channel_count]).float();
        for layer in &self.layers {
            x = layer.forward(x);
            x = self.activation.forward(x);
        }
        x = softmax(x, 1);
        NetworkOutput {
            value: x
        }
    }
}