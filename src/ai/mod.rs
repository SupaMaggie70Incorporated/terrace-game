use std::{path::Path, time::{Duration, Instant}};

use burn::{backend::candle::CandleDevice, prelude::*, tensor::activation::softmax};
use burn::record::CompactRecorder;
use lazy_static::lazy_static;
use eval::PositionEvaluate;
use net::{MlpConfig, ResnetConfig};
use nn::{Linear, LinearConfig, Relu};
use crate::ai::net::{Mlp, NetworkEvaluator, Resnet};

use crate::rules::TerraceGameState;

pub mod net;
pub mod eval;
pub mod train;
pub mod train_loop;
pub mod data;
pub mod plot;
pub mod compare;

const TRAINING_DATA_FOLDER: &str = "/media/FastSSD/Databases/Terrace_Training/";

pub type BACKEND = burn::backend::Candle;
pub type AUTODIFF_BACKEND = burn::backend::Autodiff<burn::backend::Candle>;
pub const CPU: CandleDevice = burn::backend::candle::CandleDevice::Cpu;
pub const DEVICE: CandleDevice = burn::backend::candle::CandleDevice::Cuda(0);

//pub type CURRENT_NETWORK_CONFIG_TYPE = ResnetConfig;
pub type CURRENT_NETWORK_CONFIG_TYPE = MlpConfig;
//pub type CURRENT_NETWORK_TYPE<B: Backend> = Resnet<B>;
pub type CURRENT_NETWORK_TYPE<B: Backend> = Mlp<B>;
lazy_static! {
    //pub static ref CURRENT_NETWORK_CONFIG: CURRENT_NETWORK_CONFIG_TYPE = ResnetConfig::new(4, 1, 2); // Time to train: 262s, Elo: 259 +/- 67, Time for 128 games: 27s
    //pub static ref CURRENT_NETWORK_CONFIG: CURRENT_NETWORK_CONFIG_TYPE = MlpConfig::new(vec![1024]); // Time to train: 120s, Elo: 371 +/- 77, Time for 128 games: 32
    pub static ref CURRENT_NETWORK_CONFIG: CURRENT_NETWORK_CONFIG_TYPE = MlpConfig::new(vec![256]); // Time to train: 120s, Elo: 371 +/- 86, Time for 19 games: 14s
    //pub static ref CURRENT_NETWORK_CONFIG: CURRENT_NETWORK_CONFIG_TYPE = MlpConfig::new(vec![256, 256]); // Time to train: 120s, Elo: 320 +/- 63, Time for 19 games: <forgot to record>s
    //pub static ref CURRENT_NETWORK_CONFIG: CURRENT_NETWORK_CONFIG_TYPE = MlpConfig::new(vec![256, 256, 256]); // Time to train: 120s, Elo: 268 +/- 60, Time for 128 games: 27s
    //pub static ref CURRENT_NETWORK_CONFIG: CURRENT_NETWORK_CONFIG_TYPE = MlpConfig::new(vec![256, 256, 256, 256, 256]); // Time to train: 120s, Elo: 46 +/- 45, Time for 128 games: 25
    //pub static ref CURRENT_NETWORK_CONFIG: CURRENT_NETWORK_CONFIG_TYPE = MlpConfig::new(vec![128, 128, 128]); // Time to train: 120s, Elo: 191 +/- 53, Time for 128 games: 16
    //pub static ref CURRENT_NETWORK_CONFIG: CURRENT_NETWORK_CONFIG_TYPE = MlpConfig::new(vec![64, 64, 64]); // Time to train: 120s, Elo: 118 +/- 52, Time for 128 games: 14s
    pub static ref RECORDER: CompactRecorder = CompactRecorder::new();
}

pub fn do_perf_tests() {
    let devs = vec![
        ("CPU", CPU),
        ("CUDA", DEVICE)
    ];
    let mlp_configs = vec![
        ("Mlp [16]", MlpConfig::new(vec![16])),
        ("Mlp [16, 16, 16]", MlpConfig::new(vec![16, 16, 16])),
        ("Mlp [64]", MlpConfig::new(vec![64])),
        ("Mlp [64, 64, 64]", MlpConfig::new(vec![64, 64, 64])),
        
    ];
    let resnet_configs = vec![
        ("ResNet 4x1->1", ResnetConfig::new(4, 1, 1)),
        ("ResNet 16x1->1", ResnetConfig::new(16, 1, 1)),
        ("Resnet 4x4->1", ResnetConfig::new(4, 4, 1)),
        ("Resnet 16x4->1", ResnetConfig::new(16, 4, 1)),
        ("Resnet 4x1->16", ResnetConfig::new(4, 1, 16)),
    ];
    let duration = Duration::from_secs_f32(5.0);
    let mut writer = csv::Writer::from_path("net-perf.csv").unwrap();
    writer.write_record(&["Network Name", "CPU speed", "CUDA speed"]).unwrap();
    for (config_name, config) in &mlp_configs {
        let net = config.init::<BACKEND>(&CPU);
        let eval = NetworkEvaluator::new(net);
        let res = perf_test(eval.clone(), duration);
        let cpu_per_second = res.0 as f32 / res.1;
        let net = config.init::<BACKEND>(&DEVICE);
        let res = perf_test(eval, duration);
        let cuda_per_second = res.0 as f32 / res.1;
        writer.write_record(&[*config_name, cpu_per_second.to_string().as_str(), cuda_per_second.to_string().as_str()]).unwrap();
    }
    for (config_name, config) in &resnet_configs {
        let net = config.init::<BACKEND>(&CPU);
        let eval = NetworkEvaluator::new(net);
        let res = perf_test(eval.clone(), duration);
        let cpu_per_second = res.0 as f32 / res.1;
        let net = config.init::<BACKEND>(&DEVICE);
        let res = perf_test(eval, duration);
        let cuda_per_second = res.0 as f32 / res.1;
        writer.write_record(&[*config_name, cpu_per_second.to_string().as_str(), cuda_per_second.to_string().as_str()]).unwrap();
    }
    /*for &(dev_name, dev) in &devs {
        for (config_name, config) in &mlp_configs {
            let net = config.init::<BACKEND>(&dev);
            let res = perf_test(net, duration);
            let per_second = res.0 as f32 / res.1;
            println!("{dev_name}: {config_name} - {per_second} positions/second");
        }
        for (config_name, config) in &resnet_configs {
            let net = config.init::<BACKEND>(&dev);
            let res = perf_test(net, duration);
            let per_second = res.0 as f32 / res.1;
            println!("{dev_name}: {config_name} - {per_second} positions/second");
        }
    }*/
    writer.flush().unwrap();
}
/// Returns the number of positions searched and the time taken
pub fn perf_test<E: PositionEvaluate>(net: E, time: Duration) -> (usize, f32) {
    let pos = TerraceGameState::setup_new();
    let start_time = Instant::now();
    let mut num = 0;
    while Instant::now() - start_time < time {
        net.evaluate_on_position(&pos);
        num += 1;
    }
    let end_time = Instant::now();
    (num, (end_time - start_time).as_secs_f32())
}