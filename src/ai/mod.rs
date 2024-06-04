use std::path::Path;

use burn::{backend::candle::CandleDevice, prelude::*, tensor::activation::softmax};
use nn::{Linear, LinearConfig, Relu};

pub mod net;
pub mod eval;
pub mod train;
pub mod train_loop;
pub mod data;
pub mod plot;

const TRAINING_DATA_FOLDER: &str = "/media/FastSSD/Databases/Terrace_Training/";

pub type BACKEND = burn::backend::Candle;
pub type AUTODIFF_BACKEND = burn::backend::Autodiff<burn::backend::Candle>;
pub const DEVICE: CandleDevice = burn::backend::candle::CandleDevice::Cuda(0);
