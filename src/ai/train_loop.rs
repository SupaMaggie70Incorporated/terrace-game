use std::{fmt::Debug, fs, path::{Path, PathBuf}};
use std::marker::PhantomData;
use std::process::exit;
use std::time::Instant;

use burn::{module::AutodiffModule, optim::{momentum::MomentumConfig, SgdConfig}, record::{CompactRecorder, DefaultFileRecorder}, tensor::backend::AutodiffBackend};
use burn::module::Module;
use burn::prelude::Backend;
use env_logger::Target;
use indicatif::{ProgressBar, ProgressStyle};
use log::{error, info, LevelFilter, warn};
use serde::Deserialize;
use uuid::Uuid;
use crate::ai;
use crate::ai::compare::EloComparisonMode;
use crate::ai::data::DataLoader;
use crate::ai::eval::RandomEvaluator;
use crate::ai::net::{Mlp, MlpConfig, NetworkEvaluator};

use crate::mcts::{MctsSearchConfig, MctsStopCondition};

use super::{data::load_data_from_file, compare::{compare_singlethreaded, elo_comparison}, net::Network, plot::{plot_elo_graphs, plot_loss_graph}, train::{train_on_data}};

pub fn train_networks_old<P: AsRef<Path>, B: AutodiffBackend, N: Network<B> + AutodiffModule<B>, F: Fn() -> N>(data_dir: P, data_epochs: usize, graph_dir: P, progress_bar: bool, num_networks: usize, dev: &B::Device, create_network: F, learning_rate: f64, momentum: f64) {
    let mut graph_dir: PathBuf = graph_dir.as_ref().to_owned();
    let mut networks = Vec::new();
    let mut network_histories: Vec<(Vec<(f32, f32, f32)>, Vec<(f32, f32)>)> = Vec::new();
    let data_dir: PathBuf = data_dir.as_ref().to_owned();
    let (data_loader, set_count) = DataLoader::<B>::new(dev, data_dir, data_epochs);
    for i in 0..num_networks {
        let net = create_network();
        //let net = net.load_file(format!("nets/net{i}.mpk"), &*ai::RECORDER, dev).unwrap();
        networks.push((net, SgdConfig::new().with_momentum(Some(MomentumConfig::new().with_momentum(momentum))).init()));
        network_histories.push((Vec::new(), Vec::new()));
    }
    let bar = if progress_bar {
        ProgressBar::new(set_count as u64).with_style(ProgressStyle::with_template("[{elapsed_precise}] {bar:100.cyan/blue} {pos:>7}/{len:7} {msg}").unwrap())
    } else {ProgressBar::hidden()};
    bar.set_position(0);
    for (iter, (inputs, targets)) in data_loader.enumerate() {
        for (i, (net, optim)) in networks.iter_mut().enumerate() {
            let (n, value_loss, policy_loss) = train_on_data(dev, net.clone(), inputs.clone(), targets.clone(), optim, learning_rate);
            *net = n;
            network_histories[i].1.push((value_loss, policy_loss));
            graph_dir.push(format!("loss-net{}.svg", i));
            //plot_loss_graph(&graph_dir, &network_histories[i].1);
            graph_dir.pop();
            if iter % 32 == 0 {
                net.clone().save_file(format!("nets/net{}.bin", i), &*super::RECORDER).unwrap();
            }
        }
        bar.inc(1);
    }
    for (i, (net, optim)) in networks.into_iter().enumerate() {
        net.save_file(format!("nets/net{}.bin", i), &*super::RECORDER).unwrap();
    }
    bar.finish();
}
#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct TrainLoopConfig {
    train_folder: String,
    #[serde(default)]
    clear_data: bool,
    #[serde(default)]
    log_file: String,
    networks: TrainLoopNetworkConfig,
    training: TrainLoopTrainConfig,
}
#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct TrainLoopNetworkConfig {
    #[serde(default = "TrainLoopNetworkConfig::default_count")]
    count: usize,
    mlp_shape: Vec<usize>,
    #[serde(default)]
    start_networks: Vec<String>,
}
impl TrainLoopNetworkConfig {
    fn default_count() -> usize {1}
}
#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct TrainLoopTrainConfig {
    #[serde(default = "TrainLoopTrainConfig::default_train_iterations")]
    iterations: usize,
    data_epochs: usize,
    data_generation_threads: usize,
    data_per_file: usize,
    data_generation_rounds: usize,
    max_data_per_game: usize,
    random_move_count: usize,
    policy_deviance: f64,
    mcts_use_value: bool,
    max_datapoints_trained_on: usize,
    learning_rate: f64,
    momentum: f64,

    initial_data: Vec<String>,
    initial_epochs: usize,
    initial_learning_rate: f64,
    initial_momentum: f64,
    evaluations_allowed: usize,
}
impl TrainLoopTrainConfig {
    fn default_train_iterations() -> usize {usize::MAX}
}
/// We must replace to_device with saving and loading from a file, which is ridiculous
struct NetworkInstance<B: AutodiffBackend, N: Network<B>> {
    cpu: N,
    dev: Option<N>,
    uuid: Uuid,
    _p: PhantomData<B>
}
impl<B: AutodiffBackend, N: Network<B>> NetworkInstance<B, N> {
    fn new(cfg: &N::Config, cpu: &B::Device) -> Self {
        let uuid = Uuid::new_v4();
        let cpu = N::init(cfg, cpu);
        Self {
            cpu,
            dev: None,
            uuid,
            _p: Default::default()
        }
    }
    fn load_from_file<P: AsRef<Path>>(cfg: &N::Config, uuid: Uuid, train_folder: P, cpu: &B::Device) -> Self {
        let uuid = Uuid::new_v4();
        let mut path = train_folder.as_ref().to_owned();
        path.push(format!("nets/{}.mpk", uuid));
        let net = N::init(cfg, cpu);
        let cpu = net.load_file(path, &*ai::RECORDER, cpu).unwrap();
        Self {
            cpu,
            dev: None,
            uuid,
            _p: Default::default()
        }
    }
    fn save_to_file<P: AsRef<Path>>(&self, train_folder: P) {
        let mut path = train_folder.as_ref().to_owned();
        path.push(format!("nets/{}.mpk", self.uuid));
        println!("Saving file to path {:?}", &path);
        self.cpu.clone().save_file(path, &*ai::RECORDER).unwrap();
    }
    fn from_net(net: N, cpu: &B::Device) -> Self {
        if net.device() == cpu.clone() {
            let uuid = Uuid::new_v4();
            let cpu = net;
            Self {
                uuid,
                cpu,
                dev: None,
                _p: Default::default()
            }
        } else {
            println!("Network started on GPU");
            let uuid = Uuid::new_v4();
            let cpu = net.clone().fork(cpu);
            let dev = Some(net);
            Self {
                cpu,
                dev,
                uuid,
                _p: Default::default()
            }
        }
    }
    fn cpu_net(&self) -> N {
        self.cpu.clone()
    }
    fn dev_net(&mut self, dev: &B::Device) -> N {
        if let Some(n) = &self.dev {
            n.clone()
        } else {
            let n = self.cpu.clone().fork(dev);
            self.dev = Some(n.clone());
            n
        }
    }
    fn update_from_device(&mut self, cpu: &B::Device) {
        self.cpu = self.dev.as_ref().unwrap().clone().fork(cpu);
    }
    fn uuid(&self) -> Uuid {
        self.uuid
    }
}
pub fn train_networks<B: AutodiffBackend, N: Network<B> + AutodiffModule<B>>(nets: &mut [NetworkInstance<B, N>], net_config: &N::Config, train_folder: &Path, data_subdirectory: &str, epochs: usize, learning_rate: f64, momentum: f64, cpu: &B::Device, dev: &B::Device) {
    // Whats not the problem: graphs, data(folder and # of datapoints), what else could it be!?
    let mut graph_dir: PathBuf = train_folder.to_owned();
    let mut networks = Vec::new();
    let mut network_histories: Vec<Vec<(f32, f32)>> = Vec::new();
    let mut data_dir: PathBuf = graph_dir.clone();
    data_dir.push(format!("data/{data_subdirectory}"));
    graph_dir.push("graphs");
    let (data_loader, set_count) = DataLoader::<B>::new(dev, data_dir, epochs);
    for i in 0..nets.len() {
        let net = nets[i].dev_net(dev);
        //let net = ai::CURRENT_NETWORK_CONFIG.init::<B>(cpu).to_device(dev);
        //let net = ai::CURRENT_NETWORK_CONFIG.init(dev);
        networks.push((net, SgdConfig::new().with_momentum(Some(MomentumConfig::new().with_momentum(momentum))).init()));
        network_histories.push(Vec::new());
    }
    let bar = ProgressBar::new(set_count as u64).with_style(ProgressStyle::with_template("[{elapsed_precise}] {msg} {bar:100.cyan/blue} {pos:>7}/{len:7}").unwrap()).with_message("Training networks");
    bar.set_position(0);
    for (iter, (inputs, targets)) in data_loader.enumerate() {
        for (i, (net, optim)) in networks.iter_mut().enumerate() {
            let (n, value_loss, policy_loss) = train_on_data(dev, net.clone(), inputs.clone(), targets.clone(), optim, learning_rate);
            *net = n;
            network_histories[i].push((value_loss, policy_loss));
        }
        bar.inc(1);
    }
    bar.finish();
    for (i, (net, _)) in networks.into_iter().enumerate() {
        let old_uuid = nets[i].uuid();
        let net = NetworkInstance::from_net(net, cpu);
        let new_uuid = net.uuid();
        net.save_to_file(train_folder);
        graph_dir.push(format!("{}.loss.svg", net.uuid()));
        plot_loss_graph(&graph_dir, &network_histories[i]);
        graph_dir.pop();
        info!("Trained network {new_uuid} from {old_uuid} on data {data_subdirectory}");
    }
}
pub fn training_loop() {
    type Backend = ai::AUTODIFF_BACKEND;
    const CPU: burn::backend::libtorch::LibTorchDevice = ai::CPU;
    const DEVICE: burn::backend::libtorch::LibTorchDevice = ai::DEVICE;

    let config = toml::from_str::<TrainLoopConfig>(&std::fs::read_to_string("config.toml").unwrap()).expect("Failed to parse config.toml:");
    let train_folder = PathBuf::from(&config.train_folder);
    let target = if config.log_file.len() == 0 {
        Target::Stdout
    } else {
        let file = std::fs::File::create(&config.log_file).expect("Unable to open log-file specified in config.toml");
        Target::Pipe(Box::new(file))
    };
    env_logger::Builder::from_default_env()
        .filter_level(LevelFilter::Info)
        .target(target)
        .init();
    println!("Initializing networks");
    let network_init_start = Instant::now();
    if config.networks.mlp_shape.len() == 0 {
        error!("No layers in mlp-shape in config.toml");
        exit(1);
    }
    for layer in &config.networks.mlp_shape {
        if *layer == 0 {
            error!("Layers with size 0 in mlp-shape in config.toml");
            exit(1);
        }
    }
    let net_config = MlpConfig::new(config.networks.mlp_shape.clone());
    let mut nets: Vec<NetworkInstance<Backend, Mlp<Backend>>> = Vec::new();
    if config.networks.start_networks.len() != 0 {
        if config.networks.start_networks.len() != config.networks.count {
            error!("start-networks in config.toml should have either 0 or <count> networks where count is the number of networks as defined in config.toml");
            exit(1);
        }
        for net_uuid in &config.networks.start_networks {
            nets.push(NetworkInstance::load_from_file(&net_config, Uuid::parse_str(net_uuid).unwrap(), &train_folder, &CPU));
            info!("Loaded network {}", net_uuid);
        }
    } else {
        for _ in 0..config.networks.count {
            let net = NetworkInstance::new(&net_config, &CPU);
            info!("Created network {}", net.uuid());
            nets.push(net);
        }
        for dataset in &config.training.initial_data {
            let train_start = Instant::now();
            train_networks(&mut nets, &net_config, &train_folder, dataset, config.training.initial_epochs, config.training.initial_learning_rate, config.training.initial_momentum, &CPU, &DEVICE);
            info!("Trained initial networks on dataset \"{dataset}\" in {} seconds", (Instant::now() - train_start).as_secs_f32());
        }
    }
    info!("Initialized networks in {} seconds", (Instant::now() - network_init_start).as_secs_f32());
    for _ in 0..config.training.iterations {
        println!("Starting data generation");
        let iteration_start_time = Instant::now();
        let data_uuid = Uuid::new_v4();
        let eval = NetworkEvaluator::new(nets[0].cpu_net().valid());
        let mut data_dir = train_folder.clone();
        data_dir.push(format!("data/{data_uuid}"));
        ai::data::generate_data_multithreaded(data_dir, config.training.data_generation_threads, Some(config.training.data_generation_threads * config.training.data_generation_rounds), true, eval, MctsSearchConfig {
            stop_condition: MctsStopCondition::Evaluations(config.training.evaluations_allowed),
            initial_list_size: config.training.evaluations_allowed + 4, // Probably +1 would be fine but I'm just gonna be stupid
            use_value: config.training.mcts_use_value,
            policy_deviance: config.training.policy_deviance as f32,
        }, config.training.data_per_file, config.training.max_data_per_game, config.training.random_move_count);
        info!("Generated dataset {data_uuid} in {} seconds", (Instant::now() - iteration_start_time).as_secs_f32());
        let train_start = Instant::now();
        train_networks(&mut nets, &net_config, &train_folder, &data_uuid.to_string(), config.training.initial_epochs, config.training.initial_learning_rate, config.training.initial_momentum, &CPU, &DEVICE);
        info!("Trained networks on dataset {data_uuid} in {} seconds", (Instant::now() - train_start).as_secs_f32());
    }
}