use std::{path::Path, sync::Mutex, thread::{self, Thread}, time::{Duration, Instant}};

use burn::{prelude::*, tensor::activation::softmax};
use burn::record::CompactRecorder;
use indicatif::{ProgressBar, ProgressStyle};
use lazy_static::lazy_static;
use eval::{Evaluator};
use net::{MlpConfig, NetworkEnum, ResnetConfig};
use nn::{Linear, LinearConfig, Relu};
use plot::plot_net_perf;
use crate::ai::net::{Mlp, Resnet};

use crate::rules::TerraceGameState;

pub mod net;
pub mod eval;
pub mod train;
pub mod train_loop;
pub mod data;
pub mod plot;
pub mod compare;

const TRAINING_DATA_FOLDER: &str = "/media/FastSSD/Databases/Terrace_Training/";

pub const USE_CANDLE: bool = true;
pub const USE_LIBTORCH: bool = false;
pub const USE_WGPU: bool = false;

pub type BACKEND = burn::backend::LibTorch;
pub type AUTODIFF_BACKEND = burn::backend::Autodiff<burn::backend::LibTorch>;
pub const CPU: burn::backend::libtorch::LibTorchDevice = burn::backend::libtorch::LibTorchDevice::Cpu;
pub const DEVICE: burn::backend::libtorch::LibTorchDevice = burn::backend::libtorch::LibTorchDevice::Cuda(0);
/*pub type BACKEND = burn::backend::Wgpu;
pub type AUTODIFF_BACKEND = burn::backend::Autodiff<burn::backend::Wgpu>;
pub const CPU: burn::backend::wgpu::WgpuDevice = burn::backend::wgpu::WgpuDevice::Cpu;
pub const DEVICE: burn::backend::wgpu::WgpuDevice = burn::backend::wgpu::WgpuDevice::DiscreteGpu(0);*/
/*pub type BACKEND = burn::backend::Candle;
pub type AUTODIFF_BACKEND = burn::backend::Autodiff<burn::backend::Candle>;
pub const CPU: burn::backend::candle::CandleDevice = burn::backend::candle::CandleDevice::Cpu;
pub const DEVICE: burn::backend::candle::CandleDevice = burn::backend::candle::CandleDevice::Cuda(0);*/

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

pub fn do_perf_tests(duration: Duration, thread_count: usize, show_progress: bool) {
    let mlp_configs = vec![
        /*("Mlp [16]", MlpConfig::new(vec![16])),
        ("Mlp [16, 16, 16]", MlpConfig::new(vec![16, 16, 16])),
        ("Mlp [64]", MlpConfig::new(vec![64])),
        ("Mlp [64, 64, 64]", MlpConfig::new(vec![64, 64, 64])),
        ("Mlp [256]", MlpConfig::new(vec![256])),
        ("Mlp [256, 256]", MlpConfig::new(vec![256, 256])),
        ("Mlp [512]", MlpConfig::new(vec![512])),
        ("Mlp [512, 512]", MlpConfig::new(vec![512, 512])),*/
        ("Mlp [4096, 4096]", MlpConfig::new(vec![4096, 4096])),
    ];
    let resnet_configs = vec![
        ("ResNet 4x1->1", ResnetConfig::new(4, 1, 1)),
        ("ResNet 4x2->1", ResnetConfig::new(4, 2, 1)),
        ("ResNet 4x4->1", ResnetConfig::new(4, 4, 1)),
        ("ResNet 4x8->1", ResnetConfig::new(4, 8, 1)),
        ("ResNet 4x20->1", ResnetConfig::new(4, 20, 1)),
        
        ("ResNet 8x1->2", ResnetConfig::new(8, 1, 2)),
        ("ResNet 8x2->2", ResnetConfig::new(8, 2, 2)),
        ("ResNet 8x4->2", ResnetConfig::new(8, 4, 2)),
        ("ResNet 8x8->2", ResnetConfig::new(8, 8, 2)),
        ("ResNet 8x20->2", ResnetConfig::new(8, 20, 2)),

        ("ResNet 16x1->4", ResnetConfig::new(16, 1, 4)),
        ("ResNet 16x1->2", ResnetConfig::new(16, 1, 2)),
        ("ResNet 16x2->4", ResnetConfig::new(16, 2, 4)),
        ("ResNet 16x2->2", ResnetConfig::new(16, 2, 2)),
        ("ResNet 16x4->4", ResnetConfig::new(16, 4, 4)),
        ("ResNet 16x8->4", ResnetConfig::new(16, 8, 4)),
        ("ResNet 16x20->4", ResnetConfig::new(16, 20, 4)),

        ("Resnet 64x1->16", ResnetConfig::new(64, 1, 16)),
        ("Resnet 64x2->16", ResnetConfig::new(64, 2, 16)),
        ("Resnet 64x4->16", ResnetConfig::new(64, 4, 16)),
        ("ResNet 64x8->16", ResnetConfig::new(64, 8, 16)),
        ("ResNet 64x20->16", ResnetConfig::new(64, 20, 16)),

        ("Resnet 256x1->64", ResnetConfig::new(256, 1, 64)),
        ("Resnet 256x2->64", ResnetConfig::new(256, 2, 64)),
        ("Resnet 256x4->64", ResnetConfig::new(256, 4, 64)),
        ("ResNet 256x8->64", ResnetConfig::new(256, 8, 64)),
        ("ResNet 256x20->64", ResnetConfig::new(256, 20, 64)),
    ];
    let bar = if show_progress {
        ProgressBar::new((mlp_configs.len() + resnet_configs.len()) as u64 * 2).with_style(ProgressStyle::with_template("[{elapsed_precise}] {msg} {bar:100.cyan/blue} {pos:>7}/{len:7}").unwrap()).with_message("Doing network performance tests")
    } else {
        ProgressBar::hidden()
    };
    let mut writer = csv::Writer::from_path("net-perf.csv").unwrap();
    writer.write_record(&["Network Name", "CPU speed", "CUDA speed"]).unwrap();
    let mut data = Vec::new();
    for (config_name, config) in &mlp_configs {
        let net = NetworkEnum::Mlp(config.init::<BACKEND>(&CPU));
        let eval = Evaluator::network(net);
        let res = perf_test_multithreaded(eval.clone(), duration, thread_count);
        let cpu_per_second = res.0 as f32 / res.1;
        bar.inc(1);
        let net = NetworkEnum::Mlp(config.init::<BACKEND>(&DEVICE));
        let eval = Evaluator::network(net);
        let res = perf_test_multithreaded(eval, duration, thread_count);
        let cuda_per_second = res.0 as f32 / res.1;
        bar.inc(1);
        writer.write_record(&[*config_name, cpu_per_second.to_string().as_str(), cuda_per_second.to_string().as_str()]).unwrap();
        data.push((*config_name, cpu_per_second, cuda_per_second));
    }
    for (config_name, config) in &resnet_configs {
        let net = NetworkEnum::Resnet(config.init::<BACKEND>(&CPU));
        let eval = Evaluator::network(net);
        let res = perf_test_multithreaded(eval.clone(), duration, thread_count);
        let cpu_per_second = res.0 as f32 / res.1;
        bar.inc(1);
        let net = NetworkEnum::Resnet(config.init::<BACKEND>(&DEVICE));
        let eval = Evaluator::network(net);
        let res = perf_test_multithreaded(eval, duration, thread_count);
        let cuda_per_second = res.0 as f32 / res.1;
        bar.inc(1);
        writer.write_record(&[*config_name, cpu_per_second.to_string().as_str(), cuda_per_second.to_string().as_str()]).unwrap();
        data.push((*config_name, cpu_per_second, cuda_per_second));
    }
    bar.finish();
    writer.flush().unwrap();
    plot_net_perf("net-perf.svg", data);
}
#[derive(Clone, Copy)]
pub struct PerfTestState {
    total: usize,
    stop: bool,
}
pub fn perf_test_thread(net: Evaluator<impl Backend>, state: &Mutex<PerfTestState>) {
    let pos = TerraceGameState::setup_new();
    let mut last_update_time = Instant::now();
    let mut num_since_last_update = 0;
    let mut cont = true;
    while cont {
        while Instant::now() - last_update_time < Duration::from_millis(10) {
            net.evaluate_on_position(&pos);
            num_since_last_update += 1;
        }
        let mut lock = state.lock().unwrap();
        cont = !lock.stop;
        lock.total += num_since_last_update;
        drop(lock);
        last_update_time = Instant::now();
        num_since_last_update = 0;
    }
}
pub fn perf_test_multithreaded(net: Evaluator<impl Backend>, time: Duration, thread_count: usize) -> (usize, f32) {
    let state = Mutex::new(PerfTestState {
        total: 0,
        stop: false,
    });
    let state_ref = &state as *const _;
    let mut threads = Vec::with_capacity(thread_count);
    for _ in 0..thread_count {
        let net = net.clone();
        let state_ref = unsafe{&*state_ref};
        threads.push(thread::spawn(move || {
            perf_test_thread(net, state_ref)
        }))
    }
    let start_time = Instant::now();
    while Instant::now() - start_time < time {
        thread::sleep(Duration::from_millis(1));
    }
    let mut lock = state.lock().unwrap();
    lock.stop = true;
    drop(lock);
    for thread in threads {
        thread.join().unwrap();
    }
    let end_time = Instant::now();
    let num = state.lock().unwrap().total;
    (num, (end_time - start_time).as_secs_f32())
}
/// Returns the number of positions searched and the time taken
pub fn perf_test(net: Evaluator<impl Backend>, time: Duration) -> (usize, f32) {
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