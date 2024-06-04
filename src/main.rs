use std::env::args;
#[cfg(feature = "ai")]
use ai::{eval::{compare, elo_comparison, CompareResult, EloComparisonMode}, net::{Mlp, MlpConfig, Resnet, ResnetConfig}, train_loop::train_networks};
use rand::Rng;
use rules::{AbsoluteGameResult, GameResult, TerraceGameState};
use cfg_if::cfg_if;

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
                        //ai::data::generate_random_data("/media/FastSSD/Databases/Terrace_Training/random/0.bin", 100, 64, 8).unwrap();
                        ai::data::generate_random_data_multithreaded("/media/FastSSD/Databases/Terrace_Training/test/", 12, Some(240), true, 100_000, 64, 8);
                    }
                    #[cfg(feature = "ai")]
                    "train" => {

                        println!("Training");
                        //let config = MlpConfig::new(vec![16]);
                        let config = ResnetConfig::new(4, 1, 1);
                        train_networks::<_, ai::AUTODIFF_BACKEND, Resnet<ai::AUTODIFF_BACKEND>, _>("/media/FastSSD/Databases/Terrace_Training/random/", "./graphs", true, 1, &ai::DEVICE, move || {config.init(&ai::DEVICE)}, 0.01, 0.9);
                    }
                    #[cfg(feature = "ai")]
                    "elo-from-zero" => {
                        println!("Comparing");
                        let net = ResnetConfig::new(4, 1, 1).init::<ai::BACKEND>(&ai::DEVICE);
                        //let rand_evaluator = RandomEvaluator::default();
                        let rand_evaluator = MlpConfig::new(vec![]).init::<ai::BACKEND>(&ai::DEVICE);
                        let results = compare(&ai::DEVICE, &net, &rand_evaluator, 256, 64);
                        let (elo_diff, elo_range) = elo_comparison(results, EloComparisonMode::Games, 0.95);
                        println!("Results: {:#?}", results);
                        println!("Elo: {} +/- {}", elo_diff, elo_range);
                        let (elo_diff, elo_range) = elo_comparison(results, EloComparisonMode::DecisiveGames, 0.95);
                        println!("Decisive elo: {} +/- {}", elo_diff, elo_range);
                        println!("Evaluations: {}", unsafe {ai::eval::NUM_EVALUATIONS});
                    }
                    #[cfg(feature = "ai")]
                    "other" => {
                        let (diff, conf) = ai::eval::elo_comparison(CompareResult {
                            net1_wins: 125,
                            draws: 199,
                            net2_wins: 148,
                            net1_pair_wins: 0,
                            drawn_pairs: 0,
                            net2_pair_wins: 0,
                        }, EloComparisonMode::Games, 0.95);
                        println!("{} +/- {}", diff, conf);
                    }
                    _ => println!("Unrecognized mode!")
                };
            }
        }
    }
}
