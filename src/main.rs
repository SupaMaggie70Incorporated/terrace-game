use std::str::FromStr;
use std::time::Duration;
use std::{collections::HashMap, env::args};
use std::ops::Deref;
#[cfg(feature="ai")]
use ai::data::DataGenConfig;
#[cfg(feature="ai")]
use ai::eval::Evaluator;
#[cfg(feature="ai")]
use ai::net::{self, Network, NetworkConfig};
#[cfg(feature="ai")]
use ai::train_loop::TrainLoopConfig;
#[cfg(feature="ai")]
use burn::module::Module;
#[cfg(feature = "ai")]
use ai::{compare::{compare_singlethreaded, elo_comparison, CompareResult, EloComparisonMode}, net::{Mlp, MlpConfig, Resnet, ResnetConfig}};
#[cfg(feature = "ai")]
use mcts::MctsSearchConfig;
use rand::Rng;
use rules::{AbsoluteGameResult, GameResult, TerraceGameState};
use cfg_if::cfg_if;
#[cfg(feature="ai")]
use serde::Deserialize;
use uuid::Uuid;
#[cfg(feature="ai")]
use crate::ai::compare::compare_multithreaded;
#[cfg(feature="ai")]
use crate::ai::CURRENT_NETWORK_TYPE;
#[cfg(feature="ai")]
use crate::ai::train_loop::training_loop;

mod rules;
#[cfg(feature = "gui")]
mod gui;
#[cfg(feature = "ai")]
mod ai;
#[cfg(feature = "ai")]
mod mcts;
#[cfg(feature = "ai")]
mod eval_hce;
pub const USE_MCTS: bool = false;

pub struct SendablePointer<T> {
    pub inner: *const T
}
unsafe impl<T> Send for SendablePointer<T> {}

impl<T> Clone for SendablePointer<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner
        }
    }
}

impl<T> Copy for SendablePointer<T> {}
pub struct SpecificArgs {
    hash: HashMap<String, String>
}
impl SpecificArgs {
    pub fn new() -> Self {
        let mut args = std::env::args();
        args.next();
        args.next();
        let mut hash = HashMap::new();
        let mut current_arg_name = None;
        for arg in args {
            if let Some(a) = current_arg_name {
                hash.insert(a, arg);
                current_arg_name = None;
            } else {
                current_arg_name = Some(arg);
            }
        }
        if let Some(a) = &current_arg_name {
            panic!("Argument {a} named but not specified!");
        }
        Self {
            hash
        }
    }
    pub fn get<'a>(&self, arg_name: &str, default: Option<&'a str>) -> String {
        if let Some(v) = self.hash.get(&format!("--{arg_name}")) {
            v.to_owned()
        } else {
            if let Some(v) = default {
                v.to_owned()
            } else {
                panic!("Argument not specified: {arg_name}");
            }
        }
    }
    pub fn get_optional<'a>(&self, arg_name: &str) -> Option<String> {
        if let Some(v) = self.hash.get(&format!("--{arg_name}")) {
            Some(v.to_owned())
        } else {
            None
        }
    }
    pub fn get_as<'a, T: FromStr>(&self, arg_name: &str, default: Option<T>) -> T {
        if let Some(v) = self.hash.get(&format!("--{arg_name}")) {
            if let Ok(v) = v.parse::<T>() {
                v
            } else {
                panic!("Argument not formatted: {arg_name}={v}");
            }
        } else {
            if let Some(v) = default {
                v
            } else {
                panic!("Argument not specified: {arg_name}");
            }
        }
    }
    pub fn get_as_optional<'a, T: FromStr>(&self, arg_name: &str) -> Option<T> {
        if let Some(v) = self.hash.get(&format!("--{arg_name}")) {
            if let Ok(v) = v.parse::<T>() {
                Some(v)
            } else {
                panic!("Argument not formatted: {arg_name}={v}");
            }
        } else {
            None
        }
    }
}
#[cfg(feature="ai")]
#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct Config {
    threads: usize,
    train_folder: String,
    network_config: String,
    cpu_inference: bool,
    train_loop: TrainLoopConfig,
    data_gen: DataGenConfig,
}

fn main() {
    println!("Hello, world!");
    cfg_if! {
        if #[cfg(all(feature = "gui", not(feature = "ai")))] {
            gui::run(TerraceGameState::setup_new()).unwrap();
        } else if #[cfg(all(not(feature = "gui"), not(feature = "ai")))] {
            compile_error!("Must have either gui or ai feature enabled!");
        } else {
            let args: Vec<_> = args().collect();
            if args.len() < 2 {
                println!("No mode selected!");
            } else {
                let specific_args = SpecificArgs::new();
                let config = toml::from_str::<Config>(&std::fs::read_to_string(
                    &specific_args.get("config-file", Some("config.toml"))
                ).unwrap()).expect("Failed to parse config.toml:");
                match args[1].as_str() {
                    #[cfg(feature = "gui")]
                    "gui" => {
                        println!("Running terrace GUI.");
                        const NET_PATH: &str = "/media/supa/FastSSD/Databases/Terrace_Training/nets/a69e8725-d88e-4ee3-b25b-385cb5d36de0.mpk";
                        //let net = MlpConfig::new(vec![256, 256, 256]).init::<ai::BACKEND>(&ai::CPU);
                        //gui::run(TerraceGameState::setup_new(), Some(NetworkEvaluator::new(net))).unwrap();
                    }
                    #[cfg(feature = "ai")]
                    "gen-data" => {
                        println!("Generating data");
                        let mut evaluators = Vec::new();
                        for eval in &config.data_gen.evaluators {
                            evaluators.push(Evaluator::<ai::BACKEND>::load(eval, &config.train_folder, &ai::CPU));
                        }
                        let (a, b, c) = ai::data::multithreaded_tournament(format!("{}/data/{}", config.train_folder, config.data_gen.dataset_name), config.threads, config.data_gen.datapoints_per_file, true, evaluators, MctsSearchConfig {
                            stop_condition: mcts::MctsStopCondition::Evaluations(config.data_gen.mcts_evaluations),
                            initial_list_size: config.data_gen.mcts_evaluations + 2,
                            use_value: config.data_gen.mcts_use_value,
                            policy_deviance: config.data_gen.policy_deviance
                        }, config.data_gen.max_data_per_game, config.data_gen.num_rand_moves, config.data_gen.pairs_per_matchup);
                        println!("Draws: {b}/{c}, results:\n{a:?}");
                    }
                    #[cfg(feature = "ai")]
                    "train-loop" => {
                        println!("Starting training loop");
                        training_loop(
                            config
                        );
                    }
                    #[cfg(feature = "ai")]
                    "net-perf" => {
                        println!("Running performance tests");
                        ai::do_perf_tests(
                            Duration::from_secs_f32(specific_args.get_as("duration-per", None)),
                            specific_args.get_as("thread-count", None),
                            true
                        );
                    }
                    #[cfg(feature = "ai")]
                    "create-net" => {
                        let net = net::NetworkConfigEnum::parse(&config.network_config).init::<ai::BACKEND>(&ai::CPU);
                        let uuid = Uuid::new_v4();
                        net.save_to_file(&format!("{}/nets/{uuid}", config.train_folder));
                        println!("Created network {uuid}");
                    }
                    /*#[cfg(feature = "ai")]
                    "elo-from-zero" => {
                        println!("Comparing");
                        let net = ai::CURRENT_NETWORK_CONFIG.init::<ai::BACKEND>(&ai::CPU);
                        //let net = net.load_file("nets/net0.mpk", &*ai::RECORDER, &ai::CPU).unwrap();
                        let net_evaluator = NetworkEvaluator::new(net);
                        //let net_evaluator = HceEvaluator::default();
                        let random_evaluator = RandomEvaluator::default();
                        let results = compare_multithreaded(&net_evaluator, &random_evaluator, Some(64), MctsSearchConfig {
                            stop_condition: mcts::MctsStopCondition::Evaluations(500),
                            initial_list_size: 512,
                            use_value: true,
                            policy_deviance: 0.0,
                        }, 32, 10, true);
                        println!("Results: {:#?}", results);
                        let (elo_diff, elo_range) = elo_comparison(results, EloComparisonMode::Games, 0.95);
                        println!("Elo: {} +/- {}", elo_diff, elo_range);
                        let (elo_diff, elo_range) = elo_comparison(results, EloComparisonMode::DecisiveGames, 0.95);
                        println!("Decisive elo: {} +/- {}", elo_diff, elo_range);
                        let (elo_diff, elo_range) = elo_comparison(results, EloComparisonMode::Pairs, 0.95);
                        println!("Pair elo: {} +/- {}", elo_diff, elo_range);
                        let (elo_diff, elo_range) = elo_comparison(results, EloComparisonMode::DecisivePairs, 0.95);
                        println!("Decisive pair elo: {} +/- {}", elo_diff, elo_range);
                    }*/
                    _ => println!("Unrecognized mode!")
                };
            }
        }
    }
}
