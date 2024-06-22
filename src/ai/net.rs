use std::fs;
use std::marker::PhantomData;

use burn::{config::Config, module::{AutodiffModule, Module}, nn::{conv::{Conv2d, Conv2dConfig}, loss::CrossEntropyLossConfig, BatchNorm, BatchNormConfig, Linear, LinearConfig, Relu}, record::Record, tensor::{activation::{relu, softmax}, backend::{AutodiffBackend, Backend}, Bool, Data, Float, Int, Shape, Tensor}};
use once_cell::sync::Lazy;
use regex::Regex;
use crate::ai;

use crate::rules::{self, AbsoluteGameResult, Piece, Player, Square, TerraceGameState, ALL_POSSIBLE_MOVES};

use super::data;


/// 11 possible pieces including none, 8 possible heights. The heights are only necessary for conv nets, but we will keep them for the sake of simplicity
pub const INPUT_CHANNELS: usize = 19;
pub const VALUE_OUTPUT_SIZE: usize = 3;
pub const POLICY_OUTPUT_SIZE: usize = rules::NUM_POSSIBLE_MOVES;

#[derive(Clone, Debug)]
pub struct NetworkInput<B: Backend> {
    /// Dims: [batch_size, rows, columns, channels(1 channel for each possible type of piece)]
    pub(crate) state: Tensor<B, 4, Bool>,
    /// Dims: [batch_size, POLICY_OUTPUT_SIZE]
    pub(crate) illegal_moves: Tensor<B, 2, Bool>
}
impl<B: Backend> NetworkInput<B> {
    pub fn from_state(state: &TerraceGameState, dev: &B::Device) -> Self {
        let flip = state.player_to_move() == Player::P2;
        let mut state_vec = Vec::new();
        let mut legal_move_vec = Vec::new();
        for x in 0..8 {
            for y in 0..8 {
                let pc = if flip {
                    state.square(Square::from_xy((7 - x, 7 - y)))
                } else {
                    state.square(Square::from_xy((x, y)))
                };
                state_vec.push(!pc.is_any());
                state_vec.push(pc == Piece::new(crate::rules::PieceType::T, Player::P1));
                state_vec.push(pc == Piece::new(crate::rules::PieceType::T, Player::P2));
                for size in 0..4 {
                    state_vec.push(pc == Piece::new(crate::rules::PieceType::from_size(size), Player::P1));
                    state_vec.push(pc == Piece::new(crate::rules::PieceType::from_size(size), Player::P2));
                }
                for i in 0..8 {
                    state_vec.push(i == rules::TOPOGRAPHICAL_BOARD_MAP[x as usize][y as usize]);
                }
            }
        }
        for m in &*rules::ALL_POSSIBLE_MOVES {
            legal_move_vec.push(!state.is_move_valid(*m));
        }
        let state_data = Data::new(state_vec, Shape::new([1, INPUT_CHANNELS, 8, 8]));
        let legal_move_data = Data::new(legal_move_vec, Shape::new([1, POLICY_OUTPUT_SIZE]));
        let state_tensor = Tensor::<B, 4, Bool>::from_data(state_data, dev);
        let legal_move_tensor = Tensor::<B, 2, Bool>::from_data(legal_move_data, dev);
        Self {
            state: state_tensor,
            illegal_moves: legal_move_tensor
        }
    }
    pub fn from_states(states: &[TerraceGameState], dev: &B::Device) -> Self {
        let mut state_vec = Vec::new();
        let mut legal_move_vec = Vec::new();
        for state in states {
            let flip = state.player_to_move() == Player::P2;
            for x in 0..8 {
                for y in 0..8 {
                    let pc = if flip {
                        state.square(Square::from_xy((7 - x, 7 - y)))
                    } else {
                        state.square(Square::from_xy((x, y)))
                    };
                    state_vec.push(!pc.is_any());
                    state_vec.push(pc == Piece::new(crate::rules::PieceType::T, Player::P1));
                    state_vec.push(pc == Piece::new(crate::rules::PieceType::T, Player::P2));
                    for size in 0..4 {
                        state_vec.push(pc == Piece::new(crate::rules::PieceType::from_size(size), Player::P1));
                        state_vec.push(pc == Piece::new(crate::rules::PieceType::from_size(size), Player::P2));
                    }
                    for i in 0..8 {
                        state_vec.push(i == rules::TOPOGRAPHICAL_BOARD_MAP[x as usize][y as usize]);
                    }
                }
            }
            for m in &*rules::ALL_POSSIBLE_MOVES {
                legal_move_vec.push(!state.is_move_valid(*m));
            }
        }
        let state_data = Data::new(state_vec, Shape::new([states.len(), INPUT_CHANNELS, 8, 8]));
        let legal_move_data = Data::new(legal_move_vec, Shape::new([states.len(), POLICY_OUTPUT_SIZE]));
        let state_tensor = Tensor::<B, 4, Bool>::from_data(state_data, dev);
        let legal_move_tensor = Tensor::<B, 2, Bool>::from_data(legal_move_data, dev);
        Self {
            state: state_tensor,
            illegal_moves: legal_move_tensor
        }
    }
}
#[derive(Clone, Debug)]
pub struct NetworkOutput<B: Backend> {
    /// Dims: [batch_size, channels]
    ///
    /// 1 channel for each possible type of result, so 3 for win, loss, draw. Channels are normalized such that they add up to 1
    value: Tensor<B, 2, Float>,
    /// Dims: [batch_size, NUM_POSSIBLE_MOVES]
    policy: Tensor<B, 2, Float>
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
    pub fn get_single_policies(&self) -> [f32; POLICY_OUTPUT_SIZE] {
        assert!(self.value.dims()[0] == 1);
        let v = self.policy.clone().into_data().convert().value;
        let mut data = [0.0; POLICY_OUTPUT_SIZE];
        data.copy_from_slice(&v);
        data
    }
    pub fn get_policies(&self) -> Vec<[f32; POLICY_OUTPUT_SIZE]> {
        let v: Vec<f32> = self.policy.clone().into_data().convert().value;
        let mut vec = Vec::with_capacity(v.len() / POLICY_OUTPUT_SIZE);
        for i in 0..v.len() / POLICY_OUTPUT_SIZE {
            let mut data = [0.0; POLICY_OUTPUT_SIZE];
            data.copy_from_slice(&v[i * POLICY_OUTPUT_SIZE..(i + 1) * POLICY_OUTPUT_SIZE]);
            vec.push(data);
        }
        vec
    }
    pub fn get_values_policies(&self) -> Vec<([f32; 3], [f32; POLICY_OUTPUT_SIZE])> {
        let v = self.value.clone().into_data().convert().value;
        let p = self.policy.clone().into_data().convert().value;
        let mut vec = Vec::with_capacity(v.len() / 3);
        for i in 0..v.len() / 3 {
            let value = [v[i * 3 + 0], v[i * 3 + 1], v[i * 3 + 2]];
            let mut policy = [0.0; POLICY_OUTPUT_SIZE];
            policy.copy_from_slice(&p[i * POLICY_OUTPUT_SIZE..(i + 1) * POLICY_OUTPUT_SIZE]);
            vec.push((value, policy));
        }
        vec
    }
}
#[derive(Clone, Debug)]
pub struct NetworkTarget<B: Backend> {
    /// Dims: [batch_size]
    /// Each value is the index of the correct option(0 for win, 1 for draw, 2 for loss)
    pub(crate) value: Tensor<B, 1, Int>,
    /// Dims: [batch_size]
    pub(crate) policy: Tensor<B, 1, Int>,
}
impl<B: Backend> NetworkTarget<B> {
    pub fn loss(&self, output: NetworkOutput<B>) -> NetworkLoss<B> {
        let dev = output.value.device();
        let value_loss = CrossEntropyLossConfig::new().init(&dev).forward(output.value, self.value.clone());
        let policy_loss = CrossEntropyLossConfig::new().init(&dev).forward(output.policy, self.policy.clone());
        NetworkLoss {
            value_loss,
            policy_loss
        }
    }
    pub fn num_targets(&self) -> usize {
        self.value.shape().dims[0]
    }
}
#[derive(Clone, Debug)]
pub struct NetworkLoss<B: Backend> {
    pub value_loss: Tensor<B, 1>,
    pub policy_loss: Tensor<B, 1>,
}
pub(crate) trait Network<B: Backend>: Module<B> {
    fn forward(&self, x: NetworkInput<B>) -> NetworkOutput<B>;
    fn device(&self) -> B::Device;
    fn save_to_file(&self, file: &str);

    type Config: NetworkConfig where Self: Sized;
    fn init(cfg: &Self::Config, dev: &B::Device) -> Self where Self: Sized;
    fn init_from_file(cfg: &Self::Config, dev: &B::Device, file: &str) -> Self where Self: Sized;
}
pub trait NetworkConfig {
    //fn init<B: Backend>(&self, dev: &B::Device) -> NetworkEnum<B>;
    //fn init_from_file<B: Backend>(&self, dev: &B::Device, file: &str) -> NetworkEnum<B>;
}

#[derive(Config, Debug)]
pub struct MlpConfig {
    layer_sizes: Vec<usize>,
}
impl MlpConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mlp<B> {
        assert!({ // Make sure all layers have a positive number of neurons
            let mut val = true;
            for size in &self.layer_sizes {
                if *size <= 0 {val = false;break}
            }
            val
        });
        assert!(self.layer_sizes.len() > 0);// Make sure we have some layers
        let mut layers = Vec::new();
        layers.push(LinearConfig::new(INPUT_CHANNELS * 64, self.layer_sizes[0]).init(device));
        for i in 0..self.layer_sizes.len() - 1 {
            layers.push(LinearConfig::new(self.layer_sizes[i], self.layer_sizes[i + 1]).init(device));
        }
        let value_head = LinearConfig::new(self.layer_sizes[self.layer_sizes.len() - 1], VALUE_OUTPUT_SIZE).init(device);
        let policy_head = LinearConfig::new(self.layer_sizes[self.layer_sizes.len() - 1], POLICY_OUTPUT_SIZE).init(device);
        Mlp {
            layers,
            activation: Relu::new(),
            value_head,
            policy_head
        }
    }
}
impl NetworkConfig for MlpConfig {
    /*fn init<B: Backend>(&self, dev: &B::Device) -> NetworkEnum<B> {
        NetworkEnum::Mlp(self.init::<B>(dev))
    }
    fn init_from_file<B: Backend>(&self, dev: &<B as Backend>::Device, file: &str) -> NetworkEnum<B> {
        NetworkEnum::Mlp(self.init::<B>(dev).load_file(file, &*ai::RECORDER, dev).unwrap())
    }*/
}
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    layers: Vec<Linear<B>>,
    value_head: Linear<B>,
    policy_head: Linear<B>,
    activation: Relu,
}
impl<B: Backend> Network<B> for Mlp<B> {
    fn forward(&self, input: NetworkInput<B>) -> NetworkOutput<B> {
        // We will reshape it to be a single dimension, as we are using a basic feedforward
        let [batch_size, x_width, y_height, channel_count] = input.state.shape().dims;
        let mut x = input.state.reshape([batch_size, x_width * y_height * channel_count]).float();
        for layer in &self.layers {
            x = layer.forward(x);
            x = self.activation.forward(x);
        }
        let value = softmax(self.value_head.forward(x.clone()), 1);
        let mut policy = self.policy_head.forward(x);
        policy = policy.mask_fill(input.illegal_moves, f32::NEG_INFINITY);
        let policy = softmax(policy, 1);
        NetworkOutput {
            value,
            policy
        }
    }
    fn device(&self) -> B::Device {
        self.devices()[0].clone()
    }
    fn save_to_file(&self, file: &str) {
        self.clone().save_file(file, &*ai::RECORDER).unwrap();
    }
    
    type Config = MlpConfig;
    fn init(cfg: &Self::Config, dev: &B::Device) -> Self {
        cfg.init(dev)
    }
    fn init_from_file(cfg: &Self::Config, dev: &<B as Backend>::Device, file: &str) -> Self where Self: Sized {
        cfg.init(dev).load_file(file, &*ai::RECORDER, dev).unwrap()
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
        let value_head = LinearConfig::new(self.final_output_size * 64, VALUE_OUTPUT_SIZE).init::<B>(dev);
        let policy_head = LinearConfig::new(self.final_output_size * 64, POLICY_OUTPUT_SIZE).init::<B>(dev);
        Resnet {
            input_layer,
            input_normalizer,
            blocks,
            final_conv_layer,
            final_conv_normalizer,
            value_head,
            policy_head
        }
    }
}
impl NetworkConfig for ResnetConfig {
    /*fn init<B: Backend>(&self, dev: &B::Device) -> NetworkEnum<B> {
        NetworkEnum::Resnet(self.init::<B>(dev))
    }
    fn init_from_file<B: Backend>(&self, dev: &<B as Backend>::Device, file: &str) -> NetworkEnum<B> {
        NetworkEnum::Resnet(self.init::<B>(dev).load_file(file, &*ai::RECORDER, dev).unwrap())
    }*/
}
#[derive(Module, Debug)]
pub struct Resnet<B: Backend> {
    input_layer: Conv2d<B>,
    input_normalizer: BatchNorm<B, 2>,
    blocks: Vec<ResnetBlock<B>>,
    final_conv_layer: Conv2d<B>,
    final_conv_normalizer: BatchNorm<B, 2>,
    value_head: Linear<B>,
    policy_head: Linear<B>
}
impl<B: Backend> Network<B> for Resnet<B> {
    fn forward(&self, input: NetworkInput<B>) -> NetworkOutput<B> {
        let mut x = input.state.float();
        x = self.input_layer.forward(x); // The devil
        
        x = relu(self.input_normalizer.forward(x));
        for block in &self.blocks {
            x = block.forward(x);
        }
        x = relu(self.final_conv_normalizer.forward(self.final_conv_layer.forward(x)));
        let [batch_size, final_size, _, _] = x.shape().dims;
        let x = x.reshape([batch_size, 64 * final_size]);
        let value = softmax(self.value_head.forward(x.clone()), 1);
        let mut policy = self.policy_head.forward(x);
        policy = policy.mask_fill(input.illegal_moves, f32::NEG_INFINITY);
        policy = softmax(policy, 1);
        NetworkOutput {
            value,
            policy
        }
    } 
    fn device(&self) -> <B as Backend>::Device {
        self.devices()[0].clone()
    }
    fn save_to_file(&self, file: &str) {
        self.clone().save_file(file, &*ai::RECORDER).unwrap();
    }

    type Config = ResnetConfig;
    fn init(cfg: &Self::Config, dev: &B::Device) -> Self {
        cfg.init(dev)
    }
    fn init_from_file(cfg: &Self::Config, dev: &<B as Backend>::Device, file: &str) -> Self where Self: Sized {
        cfg.init(dev).load_file(file, &*ai::RECORDER, dev).unwrap()
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
#[derive(Debug, Module)]
pub enum NetworkEnum<B: Backend> {
    Mlp(Mlp<B>),
    Resnet(Resnet<B>)
}
impl<B: Backend> NetworkEnum<B> {
    pub fn load(desc: &str, dev: &B::Device, data_dir: &str) -> Self {
        const FAIL_MESSAGE: &str = "Invalid format for network config";
        let s_index = desc.chars().position(|a| a == ':').expect(FAIL_MESSAGE);
        let cfg = NetworkConfigEnum::parse(desc[s_index + 1..].trim());
        let net = cfg.init_from_file(dev, &format!("{data_dir}/nets/{}", desc[0..s_index].trim()));
        net
    }
}
impl<B: Backend> Network<B> for NetworkEnum<B> {
    fn device(&self) -> <B as Backend>::Device {
        match self {
            Self::Mlp(s) => s.device(),
            Self::Resnet(s) => s.device(),
        }
    }
    fn forward(&self, x: NetworkInput<B>) -> NetworkOutput<B> {
        match self {
            Self::Mlp(s) => s.forward(x),
            Self::Resnet(s) => s.forward(x),
        }
    }
    
    fn save_to_file(&self, file: &str) {
        match self {
            Self::Mlp(s) => s.save_to_file(file),
            Self::Resnet(s) => s.save_to_file(file)
        }
    }
    type Config = NetworkConfigEnum where Self: Sized;
    fn init(cfg: &Self::Config, dev: &<B as Backend>::Device) -> Self where Self: Sized {
        cfg.init(dev)
    }
    fn init_from_file(cfg: &Self::Config, dev: &<B as Backend>::Device, file: &str) -> Self where Self: Sized {
        cfg.init(dev).load_file(file, &*ai::RECORDER, dev).unwrap()
    }
}
#[derive(Clone, Debug)]
pub enum NetworkConfigEnum {
    Mlp(MlpConfig),
    Resnet(ResnetConfig)
}
impl NetworkConfigEnum {
    pub fn parse(desc: &str) -> Self {
        const FAIL_MESSAGE: &str = "Invalid format for network config";
        let s_index = desc.chars().position(|a| a == ' ').expect(FAIL_MESSAGE);
        assert!(s_index + 1 < desc.len());
        let main = &desc[s_index + 1..];
        match &desc[0..s_index] {
            "mlp" => {
                let trimmed = main.trim().trim_matches(|c| c == '[' || c == ']');

                let mut vec = Vec::new();
                for element in trimmed.split(',') {
                    let number = element.trim().parse().expect(FAIL_MESSAGE);
                    vec.push(number);
                }
                Self::Mlp(MlpConfig::new(vec))
            }
            "resnet" => {
                static RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\d+)x(\d+)->(\d+)").unwrap());
                let captures = RE.captures(main).expect(FAIL_MESSAGE);
                let (_, [a, b, c]) = captures.extract();
                let a = a.parse().unwrap();
                let b = b.parse().unwrap();
                let c = c.parse().unwrap();
                Self::Resnet(ResnetConfig::new(a, b, c))
            },
            a => panic!("Unrecognized network config type: {a}, must be one of {{ mlp | resnet }}")
        }
    }
}
impl NetworkConfigEnum {
    pub fn init<B: Backend>(&self, dev: &B::Device) -> NetworkEnum<B> {
        match self {
            Self::Mlp(s) => NetworkEnum::Mlp(s.init(dev)),
            Self::Resnet(s) => NetworkEnum::Resnet(s.init(dev))
        }
    }
    pub fn init_from_file<B: Backend>(&self, dev: &B::Device, file: &str) -> NetworkEnum<B> {
        match self {
            Self::Mlp(s) => NetworkEnum::Mlp(Mlp::init_from_file(s, dev, file)),
            Self::Resnet(s) => NetworkEnum::Resnet(Resnet::init_from_file(s, dev, file))
        }
    }
}
impl NetworkConfig for NetworkConfigEnum {

}