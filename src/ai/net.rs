use std::fs;

use burn::{config::Config, module::{AutodiffModule, Module}, nn::{conv::{Conv2d, Conv2dConfig}, loss::CrossEntropyLossConfig, BatchNorm, BatchNormConfig, Linear, LinearConfig, Relu}, tensor::{activation::{relu, softmax}, backend::{AutodiffBackend, Backend}, Bool, Data, Float, Int, Shape, Tensor}};

use crate::rules::{self, AbsoluteGameResult, Piece, Player, Square, TerraceGameState};

use super::data;


/// 11 possible pieces including none, 8 possible heights. The heights are only necessary for conv nets, but we will keep them for the sake of simplicity
pub const INPUT_CHANNELS: usize = 19;
pub const OUTPUT_SIZE: usize = 3;

#[derive(Clone, Debug)]
pub struct NetworkInput<B: Backend> {
    /// Dims: [batch_size, rows, columns, channels(1 channel for each possible type of piece)]
    value: Tensor<B, 4, Bool>
}
impl<B: Backend> NetworkInput<B> {
    pub fn from_state(state: &TerraceGameState, dev: &B::Device) -> Self {
        let flip = state.player_to_move() == Player::P2;
        let mut vec = Vec::new();
        for x in 0..8 {
            for y in 0..8 {
                let pc = if flip {
                    state.square(Square::from_xy((7 - x, 7 - y)))
                } else {
                    state.square(Square::from_xy((x, y)))
                };
                vec.push(!pc.is_any());
                vec.push(pc == Piece::new(crate::rules::PieceType::T, Player::P1));
                vec.push(pc == Piece::new(crate::rules::PieceType::T, Player::P2));
                for size in 0..4 {
                    vec.push(pc == Piece::new(crate::rules::PieceType::from_size(size), Player::P1));
                    vec.push(pc == Piece::new(crate::rules::PieceType::from_size(size), Player::P2));
                }
                for i in 0..8 {
                    vec.push(i == rules::TOPOGRAPHICAL_BOARD_MAP[x as usize][y as usize]);
                }
            }
        }
        let data = Data::new(vec, Shape::new([1, INPUT_CHANNELS, 8, 8]));
        let tensor = Tensor::<B, 4, Bool>::from_data(data, dev);
        Self {
            value: tensor
        }
    }
    pub fn from_states(states: &[TerraceGameState], dev: &B::Device) -> Self {
        let mut vec = Vec::new();
        for state in states {
            let flip = state.player_to_move() == Player::P2;
            for x in 0..8 {
                for y in 0..8 {
                    let pc = if flip {
                        state.square(Square::from_xy((7 - x, 7 - y)))
                    } else {
                        state.square(Square::from_xy((x, y)))
                    };
                    vec.push(!pc.is_any());
                    vec.push(pc == Piece::new(crate::rules::PieceType::T, Player::P1));
                    vec.push(pc == Piece::new(crate::rules::PieceType::T, Player::P2));
                    for size in 0..4 {
                        vec.push(pc == Piece::new(crate::rules::PieceType::from_size(size), Player::P1));
                        vec.push(pc == Piece::new(crate::rules::PieceType::from_size(size), Player::P2));
                    }
                    for i in 0..8 {
                        vec.push(i == rules::TOPOGRAPHICAL_BOARD_MAP[x as usize][y as usize]);
                    }
                }
            }
        }
        let data = Data::new(vec, Shape::new([states.len(), INPUT_CHANNELS, 8, 8]));
        let tensor = Tensor::<B, 4, Bool>::from_data(data, dev);
        Self {
            value: tensor
        }
    }
}
#[derive(Clone, Debug)]
pub struct NetworkOutput<B: Backend> {
    /// Dims: [batch_size, channels]
    ///
    /// 1 channel for each possible type of result, so 3 for win, loss, draw. Channels are normalized such that they add up to 1
    value: Tensor<B, 2, Float>
}
impl<B: Backend> NetworkOutput<B> {
    pub fn get_single_probabilities(&self) -> [f32; 3] {
        assert!(self.value.dims()[0] == 1);
        let v: Vec<f32> = self.value.clone().into_data().convert().value;
        [v[0], v[1], v[2]]
    }
    pub fn get_probabilities(&self) -> Vec<[f32; 3]> {
        let v: Vec<f32> = self.value.clone().into_data().convert().value;
        let mut vec = Vec::with_capacity(v.len() / 3);
        for i in 0..v.len() / 3 {
            vec.push([v[i * 3 + 0], v[i * 3 + 1], v[i * 3 + 2]]);
        }
        vec
    }
}
#[derive(Clone, Debug)]
pub struct NetworkTarget<B: Backend> {
    /// Dims: [batch_size]
    /// Each value is the index of the correct option(0 for win, 1 for draw, 2 for loss)
    value: Tensor<B, 1, Int>
}
impl<B: Backend> NetworkTarget<B> {
    pub fn loss(&self, output: NetworkOutput<B>) -> Tensor<B, 1> {
        CrossEntropyLossConfig::new().init(&output.value.device()).forward(output.value, self.value.clone())
    }
    pub fn num_targets(&self) -> usize {
        self.value.shape().dims[0]
    }
}
pub trait TrainableData {
    fn to_training_values<B: Backend>(&self, dev: &B::Device) -> (NetworkInput<B>, NetworkTarget<B>);
}
impl TrainableData for &[data::PositionInfo] {
    fn to_training_values<B: Backend>(&self, dev: &B::Device) -> (NetworkInput<B>, NetworkTarget<B>) {
        let mut input_vec = Vec::new();
        let mut target_vec = Vec::new();
        for &item in self.iter() {
            for x in 0..8 {
                for y in 0..8 {
                    let pc = item.get_piece_at(x, y);
                    input_vec.push(!pc.is_any());
                    input_vec.push(pc == Piece::new(crate::rules::PieceType::T, Player::P1));
                    input_vec.push(pc == Piece::new(crate::rules::PieceType::T, Player::P2));
                    for size in 0..4 {
                        input_vec.push(pc == Piece::new(crate::rules::PieceType::from_size(size), Player::P1));
                        input_vec.push(pc == Piece::new(crate::rules::PieceType::from_size(size), Player::P2));
                    }
                    for i in 0..8 {
                        input_vec.push(i == rules::TOPOGRAPHICAL_BOARD_MAP[x][y]);
                    }
                }
            }
            target_vec.push(match item.get_game_result() {
                AbsoluteGameResult::P1Win => 0,
                AbsoluteGameResult::Draw => 1,
                AbsoluteGameResult::P2Win => 2
            });
        }
        let input_data = Data::new(input_vec, Shape::new([self.len(), INPUT_CHANNELS, 8, 8]));
        let target_data = Data::new(target_vec, Shape::new([self.len()]));
        let input = NetworkInput {
            value: Tensor::<B, 4, Bool>::from_data(input_data, dev),
        };
        let target = NetworkTarget {
            value: Tensor::<B, 1, Int>::from_data(target_data.convert(), dev)
        };
        (input, target)
    }
}



pub(crate) trait Network<B: Backend>: Module<B> {
    fn forward(&self, x: NetworkInput<B>) -> NetworkOutput<B>;
    fn device(&self) -> B::Device;
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
            layers.push(LinearConfig::new(INPUT_CHANNELS * 64, self.layer_sizes[0]).init(device));
            for i in 0..self.layer_sizes.len() - 1 {
                layers.push(LinearConfig::new(self.layer_sizes[i], self.layer_sizes[i + 1]).init(device));
            }
            layers.push(LinearConfig::new(self.layer_sizes[self.layer_sizes.len() - 1], OUTPUT_SIZE).init(device));
        } else {
            layers.push(LinearConfig::new(INPUT_CHANNELS * 64, OUTPUT_SIZE).init(device));
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
impl<B: Backend> Network<B> for Mlp<B> {
    fn forward(&self, input: NetworkInput<B>) -> NetworkOutput<B> {
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
    fn device(&self) -> B::Device {
        self.devices()[0].clone()
    }
}


#[derive(Config, Debug)]
pub struct ResnetConfig {
    channel_count: usize,
    block_count: usize,
    /// The number of channels in the output of the block conv layer, which will be fed to a linear layer
    final_output_size: usize,
}
impl ResnetConfig {
    pub fn init<B: Backend>(&self, dev: &B::Device) -> Resnet<B> {
        let input_layer = Conv2dConfig::new([INPUT_CHANNELS, self.channel_count], [3, 3]).with_padding(burn::nn::PaddingConfig2d::Same).init::<B>(dev);
        //println!("Input layer shape: {:?}", input_layer.weight.shape().dims);
        let input_normalizer = BatchNormConfig::new(self.channel_count).init::<B, 2>(dev);
        let mut blocks = Vec::new();
        for i in 0..self.block_count {
            blocks.push(ResnetBlock::<B>::new(dev, self.channel_count));
        }
        let final_conv_layer = Conv2dConfig::new([self.channel_count, self.final_output_size], [3, 3]).with_padding(burn::nn::PaddingConfig2d::Same).init::<B>(dev);
        let final_conv_normalizer = BatchNormConfig::new(self.final_output_size).init::<B, 2>(dev);
        let output_layer = LinearConfig::new(self.final_output_size * 64, OUTPUT_SIZE).init::<B>(dev);
        Resnet {
            input_layer,
            input_normalizer,
            blocks,
            final_conv_layer,
            final_conv_normalizer,
            output_layer
        }
    }
}
#[derive(Module, Debug)]
pub struct Resnet<B: Backend> {
    input_layer: Conv2d<B>,
    input_normalizer: BatchNorm<B, 2>,
    blocks: Vec<ResnetBlock<B>>,
    final_conv_layer: Conv2d<B>,
    final_conv_normalizer: BatchNorm<B, 2>,
    output_layer: Linear<B>,
}
impl<B: Backend> Network<B> for Resnet<B> {
    fn forward(&self, x: NetworkInput<B>) -> NetworkOutput<B> {
        let mut x = x.value.float();
        x = self.input_layer.forward(x); // The devil
        
        x = relu(self.input_normalizer.forward(x));
        for block in &self.blocks {
            x = block.forward(x);
        }
        x = relu(self.final_conv_normalizer.forward(self.final_conv_layer.forward(x)));
        let [batch_size, final_size, _, _] = x.shape().dims;
        let mut x = x.reshape([batch_size, 64 * final_size]);
        x = self.output_layer.forward(x);
        x = softmax(x, 1);
        NetworkOutput {
            value: x
        }
    } 
    fn device(&self) -> <B as Backend>::Device {
        self.devices()[0].clone()
    }
}
#[derive(Module, Debug)]
pub struct ResnetBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
}
impl<B: Backend> ResnetBlock<B> {
    pub fn new(dev: &B::Device, channel_count: usize) -> Self {
        let conv_config = Conv2dConfig::new([channel_count, channel_count], [3, 3]).with_padding(burn::nn::PaddingConfig2d::Same);
        let bn_config = BatchNormConfig::new(channel_count);
        Self {
            conv1: conv_config.init(dev),
            bn1: bn_config.init(dev),
            conv2: conv_config.init(dev),
            bn2: bn_config.init(dev),
        }
    }
    pub fn forward(&self, mut x: Tensor<B, 4, Float>) -> Tensor<B, 4, Float> {
        let residual = x.clone();
        x = relu(self.bn1.forward(self.conv1.forward(x)));
        x = relu(self.bn2.forward(self.conv2.forward(x)));
        x = x.add(residual);
        x = relu(x);
        x
    }
}