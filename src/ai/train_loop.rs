use std::{fmt::Debug, fs, path::{Path, PathBuf}};
use std::marker::PhantomData;
use std::process::exit;
use std::time::Instant;

use burn::{module::AutodiffModule, optim::{momentum::MomentumConfig, SgdConfig}, record::{CompactRecorder, DefaultFileRecorder}, tensor::backend::AutodiffBackend};
use burn::module::Module;
use burn::prelude::Backend;
use chrono::{Datelike, Local, Timelike, Utc};
use env_logger::Target;
use indicatif::{ProgressBar, ProgressStyle};
use log::{error, info, LevelFilter, warn};
use serde::Deserialize;
use uuid::Uuid;
use crate::ai::{self, eval::Evaluator, net::NetworkConfigEnum};
use crate::ai::compare::{compare_multithreaded, EloComparisonMode};
use crate::ai::data::DataLoader;
use crate::ai::net::{Mlp, MlpConfig};

use crate::mcts::{MctsSearchConfig, MctsStopCondition};

use super::{compare::{compare_singlethreaded, elo_comparison}, data::load_data_from_file, net::{Network, NetworkEnum}, plot::{plot_elo_graphs, plot_loss_graph}, train::train_on_data};

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct TrainLoopConfig {
    #[serde(default)]
    log_file: String,
    networks: TrainLoopNetworkConfig,
    training: TrainLoopTrainConfig,
    compare: TrainLoopCompareConfig,
}
#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct TrainLoopNetworkConfig {
    #[serde(default = "TrainLoopNetworkConfig::default_count")]
    count: usize,
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
    pairs_per_matchup: usize,
    max_data_per_file: usize,
    max_data_per_game: usize,
    random_move_count: usize,
    evaluations_allowed: usize,
    policy_deviance: f64,
    mcts_use_value: bool,
    max_data_per_batch: usize,
    learning_rate: f64,
    momentum: f64,

    initial_data: Vec<String>,
    initial_epochs: usize,
    initial_learning_rate: f64,
    initial_momentum: f64,
}
impl TrainLoopTrainConfig {
    fn default_train_iterations() -> usize {usize::MAX}
}
#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct TrainLoopCompareConfig {
    // Number of training iterations per compare, for example 1 to compare every time, 0 would be infinitely often but instead we will use as never
    initial_compare: bool,
    compare_0elo_networks: usize,
    compare_previous: bool,
    random_move_count: usize,
    evaluations_allowed: usize,
    policy_deviance: f64,
    mcts_use_value: bool,
    pair_count: usize
}
/// We must replace to_device with saving and loading from a file, which is ridiculous
#[derive(Clone)]
struct NetworkInstance<B: AutodiffBackend> {
    cpu: NetworkEnum<B>,
    dev: Option<NetworkEnum<B>>,
    uuid: Uuid,
}
impl<B: AutodiffBackend> NetworkInstance<B> {
    fn new(cfg: &NetworkConfigEnum, cpu: &B::Device) -> Self {
        let uuid = Uuid::new_v4();
        let cpu = NetworkEnum::init(cfg, cpu);
        Self {
            cpu,
            dev: None,
            uuid,
        }
    }
    fn load_from_file<P: AsRef<Path>>(cfg: &NetworkConfigEnum, uuid: Uuid, train_folder: P, cpu: &B::Device) -> Self {
        let uuid = Uuid::new_v4();
        let mut path = train_folder.as_ref().to_owned();
        path.push(format!("nets/{}.mpk", uuid));
        let net = NetworkEnum::init(cfg, cpu);
        let cpu = net.load_file(path, &*ai::RECORDER, cpu).unwrap();
        Self {
            cpu,
            dev: None,
            uuid,
        }
    }
    fn save_to_file<P: AsRef<Path>>(&self, train_folder: P) {
        let mut path = train_folder.as_ref().to_owned();
        path.push(format!("nets/{}.mpk", self.uuid));
        self.cpu.clone().save_file(path, &*ai::RECORDER).unwrap();
    }
    fn from_net(net: NetworkEnum<B>, cpu: &B::Device) -> Self {
        if net.device() == cpu.clone() {
            let uuid = Uuid::new_v4();
            let cpu = net;
            Self {
                uuid,
                cpu,
                dev: None,
            }
        } else {
            let uuid = Uuid::new_v4();
            let cpu = net.clone().fork(cpu);
            let dev = Some(net);
            Self {
                cpu,
                dev,
                uuid,
            }
        }
    }
    fn cpu_net(&self) -> NetworkEnum<B> {
        self.cpu.clone()
    }
    fn dev_net(&mut self, dev: &B::Device) -> NetworkEnum<B> {
        if let Some(n) = &self.dev {
            n.clone()
        } else {
            let n = self.cpu.clone().fork(dev);
            self.dev = Some(n.clone());
            n
        }
    }
    pub fn inference(&mut self, use_cpu: bool, dev: &B::Device) -> NetworkEnum<B::InnerBackend> {
        if use_cpu {
            self.cpu_net().valid()
        } else {
            self.dev_net(dev).valid()
        }
    }
    fn update_from_device(&mut self, cpu: &B::Device) {
        self.cpu = self.dev.as_ref().unwrap().clone().fork(cpu);
    }
    fn uuid(&self) -> Uuid {
        self.uuid
    }
}
fn train_networks<B: AutodiffBackend>(nets: &[NetworkInstance<B>], net_config: &NetworkConfigEnum, train_folder: &Path, data_subdirectory: &str, epochs: usize, max_datapoints_per_batch: usize, learning_rate: f64, momentum: f64, cpu: &B::Device, dev: &B::Device) -> Vec<NetworkInstance<B>> {
    // Whats not the problem: graphs, data(folder and # of datapoints), what else could it be!?
    let mut graph_dir: PathBuf = train_folder.to_owned();
    let mut networks = Vec::new();
    let mut network_histories: Vec<Vec<(f32, f32)>> = Vec::new();
    let mut data_dir: PathBuf = graph_dir.clone();
    data_dir.push(format!("data/{data_subdirectory}"));
    graph_dir.push("graphs");
    let (data_loader, set_count) = DataLoader::<B>::new(dev, data_dir, epochs, max_datapoints_per_batch);
    for i in 0..nets.len() {
        let net = nets[i].cpu_net().fork(dev);
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
    let mut new_nets = Vec::new();
    for (i, (net, _)) in networks.iter().enumerate() {
        let old_uuid = nets[i].uuid();
        let net = NetworkInstance::from_net(net.clone(), cpu);
        let new_uuid = net.uuid();
        net.save_to_file(train_folder);
        graph_dir.push(format!("{}.loss.svg", net.uuid()));
        plot_loss_graph(&graph_dir, &network_histories[i]);
        graph_dir.pop();
        new_nets.push(net);
        info!("Trained network {new_uuid} from {old_uuid} on data {data_subdirectory}");
    }
    new_nets
}
pub fn training_loop(config: crate::Config) {
    let datetime = Local::now();
    let session = format!("{}.{}.{} {}:{}:{}", datetime.year(), datetime.month(), datetime.day(), datetime.hour(), datetime.minute(), datetime.second());
    println!("Session: {session}");

    let train_folder = PathBuf::from(&config.train_folder);
    let target = if config.train_loop.log_file.len() == 0 {
        //Target::Stdout
        let mut filepath = train_folder.clone();
        filepath.push(format!("sessions/{session}.log"));
        let file = std::fs::File::create(&filepath).expect("Unable to create session log file");
        println!("Writing to log file {filepath:?}");
        Target::Pipe(Box::new(file))
    } else {
        let file = std::fs::File::create(&config.train_loop.log_file).expect("Unable to open log-file specified in config.toml");
        Target::Pipe(Box::new(file))
    };
    env_logger::Builder::from_default_env()
        .filter_level(LevelFilter::Info)
        .target(target)
        .init();
    println!("Initializing networks");
    let network_init_start = Instant::now();
    if config.train_loop.networks.count == 0 {
        error!("count in config.toml/networks is set to 0, but you need to train on at least one network!");
        exit(1);
    }
    let net_config = NetworkConfigEnum::parse(&config.network_config);
    let mut old_nets: Vec<NetworkInstance<ai::AUTODIFF_BACKEND>> = Vec::new();
    let mut new_nets: Vec<NetworkInstance<ai::AUTODIFF_BACKEND>> = Vec::new();
    if config.train_loop.networks.start_networks.len() != 0 {
        if config.train_loop.networks.start_networks.len() != config.train_loop.networks.count {
            error!("start-networks in config.toml should have either 0 or <count> networks where count is the number of networks as defined in config.toml");
            exit(1);
        }
        for _ in 0..config.train_loop.networks.count {
            let net = NetworkInstance::new(&net_config, &ai::CPU);
            info!("Created network {}", net.uuid());
            net.save_to_file(&train_folder);
            old_nets.push(net);
        }
        for net_uuid in &config.train_loop.networks.start_networks {
            new_nets.push(NetworkInstance::load_from_file(&net_config, Uuid::parse_str(net_uuid).unwrap(), &train_folder, &ai::CPU));
            info!("Loaded network {}", net_uuid);
        }
    } else {
        for _ in 0..config.train_loop.networks.count {
            let net = NetworkInstance::new(&net_config, &ai::CPU);
            info!("Created network {}", net.uuid());
            net.save_to_file(&train_folder);
            old_nets.push(net);
        }
        if config.train_loop.training.initial_data.len() == 0 {
            for _ in 0..config.train_loop.networks.count {
                let net = NetworkInstance::new(&net_config, &ai::CPU);
                info!("Created network {}", net.uuid());
                net.save_to_file(&train_folder);
                new_nets.push(net);
            }
        } else {
            for dataset in &config.train_loop.training.initial_data {
                let train_start = Instant::now();
                new_nets = train_networks(&old_nets, &net_config, &train_folder, dataset, config.train_loop.training.initial_epochs, config.train_loop.training.max_data_per_batch, config.train_loop.training.initial_learning_rate, config.train_loop.training.initial_momentum, &ai::CPU, &ai::DEVICE);
                info!("Trained initial networks on dataset \"{dataset}\" in {} seconds", (Instant::now() - train_start).as_secs_f32());
            }
        }
    }
    info!("Initialized networks in {} seconds", (Instant::now() - network_init_start).as_secs_f32());
    let mut prev_best = old_nets[0].clone();
    let mut elo_from_0_histories = Vec::new();
    let mut elo_from_previous_histories = Vec::new();
    let mut elo_total_histories = vec![(0.0, 0.0, 0.0)];
    let mut prev_best_elo = 0.0;
    if config.train_loop.compare.initial_compare && config.train_loop.training.initial_data.len() > 0 {
        for net in &mut new_nets[0..config.train_loop.compare.compare_0elo_networks] {
            let start_time = Instant::now();
            let results = compare_multithreaded(&Evaluator::Network(net.inference(config.cpu_inference, &ai::DEVICE)), &Evaluator::<ai::BACKEND>::Random, Some(config.train_loop.compare.pair_count), MctsSearchConfig {
                stop_condition: MctsStopCondition::Evaluations(config.train_loop.compare.evaluations_allowed),
                initial_list_size: config.train_loop.compare.evaluations_allowed + 4,
                use_value: config.train_loop.compare.mcts_use_value,
                policy_deviance: config.train_loop.compare.policy_deviance as f32,
            }, config.train_loop.compare.random_move_count, config.threads, true);
            let time = (Instant::now() - start_time).as_secs_f32();
            let mut elo = elo_comparison(results, EloComparisonMode::Games, 0.95);
            if elo.0 == f32::INFINITY {
                elo.0 = 1000.0;
            } else if elo.0 == f32::NEG_INFINITY {
                elo.0 = -1000.0;
            } else if elo.0.is_nan() { // If this occurs, its more likely to mean a very strong rating than none
                elo.0 = 0.0;
            }
            if !elo.1.is_finite() {
                elo.1 = 0.0;
            }
            elo_from_0_histories.push((elo.0, elo.0 + elo.1, elo.0 - elo.1));
            info!("Compared network {} to 0elo in {time} seconds:\n{}", net.uuid(), results);
        }
    }
    for iteration_index in 0..config.train_loop.training.iterations {
        info!("Starting iteration {iteration_index}");
        println!("Starting data generation");
        let iteration_start_time = Instant::now();
        let data_uuid = Uuid::new_v4();
        let mut evals = Vec::new();
        for net in &mut old_nets {
            evals.push(Evaluator::Network(net.inference(config.cpu_inference, &ai::DEVICE)));
        }
        for net in &mut new_nets {
            evals.push(Evaluator::Network(net.inference(config.cpu_inference, &ai::DEVICE)));
        }
        let mut data_dir = train_folder.clone();
        data_dir.push(format!("data/{data_uuid}"));
        let out = ai::data::multithreaded_tournament(data_dir, config.threads, config.train_loop.training.max_data_per_file, true, evals.clone(), MctsSearchConfig {
            stop_condition: MctsStopCondition::Evaluations(config.train_loop.training.evaluations_allowed),
            initial_list_size: config.train_loop.training.evaluations_allowed + 4, // Probably +1 would be fine but I'm just gonna be stupid
            use_value: config.train_loop.training.mcts_use_value,
            policy_deviance: config.train_loop.training.policy_deviance as f32,
        }, config.train_loop.training.max_data_per_game, config.train_loop.training.random_move_count, config.train_loop.training.pairs_per_matchup);
        let mut results: Vec<_> = out.0.iter().enumerate().map(|(i, score)| {
            if i >= config.train_loop.networks.count {
                (score, true, new_nets[i % config.train_loop.networks.count].clone())
            } else {
                (score, false, old_nets[i].clone())
            }
        }).collect();
        results.sort_by_key(|(score, is_new, net)| *score);
        old_nets.clear();
        info!("Generated dataset {data_uuid} in {} seconds with {}/{} draws", (Instant::now() - iteration_start_time).as_secs_f32(), out.1, out.2);
        for i in 0..config.train_loop.networks.count {
            let (score, is_new, net) = results.pop().unwrap();
            info!("Net {} got {}th place out of {} networks, scoring {}, is new: {}", net.uuid(), i + 1, config.train_loop.networks.count * 2, score, is_new);
            old_nets.push(net);
        }
        let train_start = Instant::now();
        new_nets = train_networks(&old_nets, &net_config, &train_folder, &data_uuid.to_string(), config.train_loop.training.data_epochs, config.train_loop.training.max_data_per_batch, config.train_loop.training.learning_rate, config.train_loop.training.momentum, &ai::CPU, &ai::DEVICE);
        info!("Trained networks on dataset {data_uuid} in {} seconds", (Instant::now() - train_start).as_secs_f32());

        let compare_start_time = Instant::now();
        //let compared = config.compare.compare_0elo_networks > 0 || config.compare.compare_previous;
        let mut compared = config.train_loop.compare.compare_0elo_networks > 0;
        for net in &mut old_nets[0..config.train_loop.compare.compare_0elo_networks] {
            let start_time = Instant::now();
            let results = compare_multithreaded(&Evaluator::Network(net.inference(config.cpu_inference, &ai::DEVICE)), &Evaluator::<ai::BACKEND>::Random, Some(config.train_loop.compare.pair_count), MctsSearchConfig {
                stop_condition: MctsStopCondition::Evaluations(config.train_loop.compare.evaluations_allowed),
                initial_list_size: config.train_loop.compare.evaluations_allowed + 4,
                use_value: config.train_loop.compare.mcts_use_value,
                policy_deviance: config.train_loop.compare.policy_deviance as f32,
            }, config.train_loop.compare.random_move_count, config.threads, true);
            let time = (Instant::now() - start_time).as_secs_f32();
            let mut elo = elo_comparison(results, EloComparisonMode::Games, 0.95);
            if elo.0 == f32::INFINITY {
                elo.0 = 1000.0;
            } else if elo.0 == f32::NEG_INFINITY {
                elo.0 = -1000.0;
            } else if elo.0.is_nan() { // If this occurs, its more likely to mean a very strong rating than none
                elo.0 = 0.0;
            }
            if !elo.1.is_finite() {
                elo.1 = 0.0;
            }
            elo_from_0_histories.push((elo.0, elo.0 + elo.1, elo.0 - elo.1));
            plot_elo_graphs(format!("{}/graphs/{session}.elo-from-0.svg", train_folder.display()), &elo_from_0_histories);
            info!("Compared network {} to 0elo in {time} seconds:\n{}", net.uuid(), results);
        }
        if config.train_loop.compare.compare_previous {
            if old_nets[0].uuid() != prev_best.uuid() {
                compared = true;
                let start_time = Instant::now();
                let results = compare_multithreaded(&Evaluator::Network(old_nets[0].inference(config.cpu_inference, &ai::DEVICE)), &Evaluator::Network(prev_best.cpu_net().valid()), Some(config.train_loop.compare.pair_count), MctsSearchConfig {
                    stop_condition: MctsStopCondition::Evaluations(config.train_loop.compare.evaluations_allowed),
                    initial_list_size: config.train_loop.compare.evaluations_allowed + 4,
                    use_value: config.train_loop.compare.mcts_use_value,
                    policy_deviance: config.train_loop.compare.policy_deviance as f32,
                }, config.train_loop.compare.random_move_count, config.threads, true);
                let mut elo = elo_comparison(results, EloComparisonMode::Games, 0.95);
                if elo.0 == f32::INFINITY {
                    elo.0 = 1000.0;
                } else if elo.0 == f32::NEG_INFINITY {
                    elo.0 = -1000.0;
                } else if elo.0.is_nan() { // If this occurs, its more likely to mean a very strong rating than none
                    elo.0 = 0.0;
                }
                if !elo.1.is_finite() {
                    elo.1 = 0.0;
                }
                elo_from_previous_histories.push((elo.0, elo.0 + elo.1, elo.0 - elo.1));
                prev_best_elo += elo.0;
                elo_total_histories.push((prev_best_elo, prev_best_elo + elo.1, prev_best_elo - elo.1));
                plot_elo_graphs(format!("{}/graphs/{session}.total-elo.svg", train_folder.display()), &elo_total_histories);
                plot_elo_graphs(format!("{}/graphs/{session}.elo-vs-previous.svg", train_folder.display()), &elo_from_previous_histories);
                let time = (Instant::now() - start_time).as_secs_f32();
                info!("Compared network {} to previous best {} in {time} seconds:\n{}", old_nets[0].uuid(), prev_best.uuid(), results);
            } else {
                elo_total_histories.push(elo_total_histories[elo_total_histories.len() - 1]);
                elo_from_previous_histories.push((0.0, 0.0, 0.0));
                plot_elo_graphs(format!("{}/graphs/{session}.total-elo.svg", train_folder.display()), &elo_total_histories);
                plot_elo_graphs(format!("{}/graphs/{session}.elo-vs-previous.svg", train_folder.display()), &elo_from_previous_histories);
                info!("Skipped comparing to previous best as the current best is the previous best");
            }
        }
        prev_best = old_nets[0].clone();
        if compared {
            info!("Finished comparing networks in {} seconds", (Instant::now() - compare_start_time).as_secs_f32());
        }
        info!("Finished training iteration in {} seconds", (Instant::now() - iteration_start_time).as_secs_f32());
    }
}