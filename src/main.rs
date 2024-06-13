use std::env::args;
use std::ops::Deref;
use burn::module::Module;
#[cfg(feature = "ai")]
use ai::{compare::{compare_singlethreaded, elo_comparison, CompareResult, EloComparisonMode}, eval::RandomEvaluator, net::{Mlp, MlpConfig, Resnet, ResnetConfig, NetworkEvaluator}};
#[cfg(feature = "ai")]
use mcts::MctsSearchConfig;
use rand::Rng;
use rules::{AbsoluteGameResult, GameResult, TerraceGameState};
use cfg_if::cfg_if;
use crate::ai::CURRENT_NETWORK_TYPE;
use crate::ai::eval::HceEvaluator;
use crate::ai::train_loop::{train_networks_old, training_loop};

mod rules;
#[cfg(feature = "gui")]
mod gui;
#[cfg(feature = "ai")]
mod ai;
#[cfg(feature = "ai")]
mod mcts;
#[cfg(feature = "ai")]
mod eval_hce;

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
                match args[1].as_str() {
                    #[cfg(feature = "gui")]
                    "gui" => {
                        println!("Running terrace GUI.");
                        gui::run(TerraceGameState::setup_new()).unwrap();
                    }
                    #[cfg(feature = "ai")]
                    "gen-data" => {
                        println!("Generating data");
                        if args.len() < 3 {
                            let network = ai::CURRENT_NETWORK_CONFIG.init::<ai::BACKEND>(&ai::CPU);
                            let network = network.load_file("nets/net0.mpk", &*ai::RECORDER, &ai::CPU).unwrap();
                            let eval = NetworkEvaluator::new(network);
                            //let eval = RandomEvaluator::default();
                            // Expect 4000 datapoints every ~20 minutes
                            if true {
                                ai::data::generate_data_multithreaded("/media/supa/FastSSD/Databases/Terrace_Training/371elo/", 10, Some(400), true, eval, MctsSearchConfig {
                                    stop_condition: mcts::MctsStopCondition::Evaluations(500),
                                    initial_list_size: 512
                                }, 4000, 64, 32);
                            } else {
                                ai::data::generate_data("/media/supa/FastSSD/Databases/Terrace_Training/371elo/0.bin", &eval, MctsSearchConfig {
                                    stop_condition: mcts::MctsStopCondition::Evaluations(500),
                                    initial_list_size: 512
                                }, 4000, 64, 32, true).unwrap();
                            }
                        }
                    }
                    #[cfg(feature = "ai")]
                    "train" => {
                        println!("Training");
                        let config = ai::CURRENT_NETWORK_CONFIG.deref();
                        train_networks_old::<_, ai::AUTODIFF_BACKEND, ai::CURRENT_NETWORK_TYPE<ai::AUTODIFF_BACKEND>, _>("/media/supa/FastSSD/Databases/Terrace_Training/data/random/", 2, "./graphs", true, 1, &ai::DEVICE, move || {config.init(&ai::DEVICE)}, 0.01, 0.9);
                    }
                    #[cfg(feature = "ai")]
                    "train-loop" => {
                        println!("Starting training loop");
                        training_loop();
                    }
                    #[cfg(feature = "ai")]
                    "net-perf" => {
                        println!("Doing performance tests");
                        ai::do_perf_tests();
                    }
                    #[cfg(feature = "ai")]
                    "elo-from-zero" => {
                        println!("Comparing");
                        let net = ai::CURRENT_NETWORK_CONFIG.init::<ai::BACKEND>(&ai::CPU);
                        let net = net.load_file("nets/net0.mpk", &*ai::RECORDER, &ai::CPU).unwrap();
                        let net_evaluator = NetworkEvaluator::new(net);
                        //let net_evaluator = HceEvaluator::default();
                        let random_evaluator = RandomEvaluator::default();
                        let results = compare_singlethreaded(&net_evaluator, &random_evaluator, 64, 32, MctsSearchConfig {
                            stop_condition: mcts::MctsStopCondition::Evaluations(500),
                            initial_list_size: 512
                        });
                        println!("Results: {:#?}", results);
                        let (elo_diff, elo_range) = elo_comparison(results, EloComparisonMode::Games, 0.95);
                        println!("Elo: {} +/- {}", elo_diff, elo_range);
                        let (elo_diff, elo_range) = elo_comparison(results, EloComparisonMode::DecisiveGames, 0.95);
                        println!("Decisive elo: {} +/- {}", elo_diff, elo_range);
                        let (elo_diff, elo_range) = elo_comparison(results, EloComparisonMode::Pairs, 0.95);
                        println!("Pair elo: {} +/- {}", elo_diff, elo_range);
                        let (elo_diff, elo_range) = elo_comparison(results, EloComparisonMode::DecisivePairs, 0.95);
                        println!("Decisive pair elo: {} +/- {}", elo_diff, elo_range);
                    }
                    #[cfg(feature = "ai")]
                    "other" => {
                        println!("Comparing");
                        let net = ai::CURRENT_NETWORK_CONFIG.init::<ai::BACKEND>(&ai::CPU);
                        let net = net.load_file("nets/net0.mpk", &*ai::RECORDER, &ai::CPU).unwrap();
                        let net_evaluator = NetworkEvaluator::new(net);
                        let hce_evaluator = HceEvaluator::default();
                        let results = compare_singlethreaded(&net_evaluator, &hce_evaluator, 64, 32, MctsSearchConfig {
                            stop_condition: mcts::MctsStopCondition::Evaluations(500),
                            initial_list_size: 512
                        });
                        println!("Results: {:#?}", results);
                        let (elo_diff, elo_range) = elo_comparison(results, EloComparisonMode::Games, 0.95);
                        println!("Elo: {} +/- {}", elo_diff, elo_range);
                        let (elo_diff, elo_range) = elo_comparison(results, EloComparisonMode::DecisiveGames, 0.95);
                        println!("Decisive elo: {} +/- {}", elo_diff, elo_range);
                        let (elo_diff, elo_range) = elo_comparison(results, EloComparisonMode::Pairs, 0.95);
                        println!("Pair elo: {} +/- {}", elo_diff, elo_range);
                        let (elo_diff, elo_range) = elo_comparison(results, EloComparisonMode::DecisivePairs, 0.95);
                        println!("Decisive pair elo: {} +/- {}", elo_diff, elo_range);
                    }
                    _ => println!("Unrecognized mode!")
                };
            }
        }
    }
}
