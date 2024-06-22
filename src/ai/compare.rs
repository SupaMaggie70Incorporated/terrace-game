use std::f32::consts::{LN_10, PI, SQRT_2};
use std::fmt::{Display, Formatter};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};
use burn::tensor::backend::Backend;
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use crate::mcts::{MctsSearch, MctsSearchConfig};
use crate::rules::{AbsoluteGameResult, GameResult, Player, TerraceGameState};

use super::eval::Evaluator;

#[derive(Clone, Copy, Debug)]
pub struct CompareResult {
    pub net1_wins: usize,
    pub draws: usize,
    pub net2_wins: usize,
    pub net1_pair_wins: usize,
    pub drawn_pairs: usize,
    pub net2_pair_wins: usize,
}
impl Display for CompareResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        const MARGIN_OF_ERROR: f32 = 0.95;
        let elo = elo_comparison(*self, EloComparisonMode::Games, MARGIN_OF_ERROR);
        let decisive_elo = elo_comparison(*self, EloComparisonMode::DecisiveGames, MARGIN_OF_ERROR);
        let pair_elo = elo_comparison(*self, EloComparisonMode::Pairs, MARGIN_OF_ERROR);
        let decisive_pair_elo = elo_comparison(*self, EloComparisonMode::DecisivePairs, MARGIN_OF_ERROR);
        write!(f, "Games: +{}={}-{}\nPairs: +{}={}-{}\nElo difference: {} +/- {}\nDecisive elo difference: {} +/- {}\nPair elo difference: {} +/- {}\nDecisive pair elo difference: {} +/- {}",
            self.net1_wins, self.draws, self.net2_wins,
            self.net1_pair_wins, self.drawn_pairs, self.net2_pair_wins,
            elo.0, elo.1,
            decisive_elo.0, decisive_elo.1,
            pair_elo.0, pair_elo.1,
            decisive_pair_elo.0, decisive_pair_elo.1
        )
    }
}
pub fn compare_singlethreaded(eval1: &Evaluator<impl Backend>, eval2: &Evaluator<impl Backend>, pair_count: usize, random_move_count: usize, mcts_config: MctsSearchConfig) -> CompareResult {
    let mut results = CompareResult {
        net1_wins: 0,
        draws: 0,
        net2_wins: 0,
        net1_pair_wins: 0,
        drawn_pairs: 0,
        net2_pair_wins: 0,
    };
    let mut mov_vec = Vec::new();
    let bar = ProgressBar::new(pair_count as u64 * 2).with_style(ProgressStyle::with_template("[{elapsed_precise}] {bar:100.cyan/blue} {pos:>7}/{len:7} {msg}").unwrap());
    let start_time = Instant::now();
    for pair in 0..pair_count {
        let mut pair_start = TerraceGameState::setup_new();
        let mut pair_result = 0;
        loop {
            let mut success = true;
            let a =  for i in 0..random_move_count {
                if pair_start.result() != GameResult::Ongoing {
                    success = false;
                    break;
                }
                mov_vec.clear();
                pair_start.generate_moves(&mut mov_vec);
                if mov_vec.len() == 0 {
                    success = false;
                    break;
                }
                pair_start.make_move(mov_vec[rand::thread_rng().gen_range(0..mov_vec.len())]);
            };
            if success {
                break;
            } else {
                pair_start = TerraceGameState::setup_new();
            }
        }
        for is_second_game in [false, true] {
            let mut state = pair_start;
            while state.result() == GameResult::Ongoing {
                mov_vec.clear();
                state.generate_moves(&mut mov_vec);
                if mov_vec.len() == 0 {
                    state.skip_turn();
                } else if mov_vec.len() == 1 {
                    state.make_move(mov_vec[0]);
                } else {
                    let mov = if is_second_game == (state.player_to_move() == Player::P2) {
                        let mut mcts = MctsSearch::new(state, mcts_config, eval1);
                        mcts.search().mov
                        //eval1.find_best_value(state)
                    } else {
                        let mut mcts = MctsSearch::new(state, mcts_config, eval2);
                        mcts.search().mov
                        //eval2.find_best_value(state)
                    };
                    state.make_move(mov);
                }
            }
            let mut result = state.result().into_absolute();
            if is_second_game {
                result = result.other();
            }
            match result {
                AbsoluteGameResult::P1Win => {
                    results.net1_wins += 1;
                    pair_result += 2;
                }
                AbsoluteGameResult::Draw => {
                    results.draws += 1;
                    pair_result += 1;
                }
                AbsoluteGameResult::P2Win => {
                    results.net2_wins += 1;
                    pair_result += 0;
                }
            }
            bar.inc(1);
            let (elo_diff, elo_range) = elo_comparison(results, EloComparisonMode::Games, 0.95);
            let (decisive_elo_diff, decisive_elo_range) = elo_comparison(results, EloComparisonMode::DecisiveGames, 0.95);
            let time = (Instant::now() - start_time).as_secs_f32();
            bar.set_message(format!("Results: +{}={}-{}, elo {} +/- {}, decisive elo {} +/- {}", results.net1_wins, results.draws, results.net2_wins, elo_diff, elo_range, decisive_elo_diff, decisive_elo_range));
        }
        if pair_result > 2 {
            results.net1_pair_wins += 1;
        } else if pair_result < 2 {
            results.net2_pair_wins += 1;
        } else {
            results.drawn_pairs += 1;
        }
    }
    bar.finish();
    results
}
pub struct CompareState {
    current_index: usize,
    pairs_completed: usize,
    results: CompareResult,
    required: Option<usize>,
    stop: bool,
}
pub fn compare_thread(state: &Mutex<CompareState>, eval1: &Evaluator<impl Backend>, eval2: &Evaluator<impl Backend>, mcts_config: MctsSearchConfig, random_move_count: usize) {
    let mut mov_vec = Vec::new();
    loop {
        let mut lock = state.lock().unwrap();
        if Some(lock.current_index) == lock.required || lock.stop {
            break;
        }
        let i = lock.current_index;
        lock.current_index += 1;
        drop(lock);

        let mut pair_start = TerraceGameState::setup_new();
        loop {
            let mut success = true;
            let a =  for i in 0..random_move_count {
                if pair_start.result() != GameResult::Ongoing {
                    success = false;
                    break;
                }
                mov_vec.clear();
                pair_start.generate_moves(&mut mov_vec);
                if mov_vec.len() == 0 {
                    success = false;
                    break;
                }
                pair_start.make_move(mov_vec[rand::thread_rng().gen_range(0..mov_vec.len())]);
            };
            if success {
                break;
            } else {
                pair_start = TerraceGameState::setup_new();
            }
        }
        let mut net1_wins = 0;
        let mut net2_wins = 0;
        let mut draws = 0;
        let mut pair_result = 0;
        for is_second_game in [false, true] {
            let mut state = pair_start;
            while state.result() == GameResult::Ongoing {
                mov_vec.clear();
                state.generate_moves(&mut mov_vec);
                if mov_vec.len() == 0 {
                    state.skip_turn();
                } else if mov_vec.len() == 1 {
                    state.make_move(mov_vec[0]);
                } else {
                    let mov = if is_second_game == (state.player_to_move() == Player::P2) {
                        if crate::USE_MCTS {
                            let mut mcts = MctsSearch::new(state, mcts_config, eval1 as *const _);
                            mcts.search().mov
                        } else {
                            eval1.find_best_value(state, mcts_config.policy_deviance)
                        }
                    } else {
                        if crate::USE_MCTS {
                            let mut mcts = MctsSearch::new(state, mcts_config, eval2 as *const _);
                            mcts.search().mov
                        } else {
                            eval2.find_best_value(state, mcts_config.policy_deviance)
                        }
                    };
                    state.make_move(mov);
                }
            }
            let mut result = state.result().into_absolute();
            if is_second_game {
                result = result.other();
            }
            match result {
                AbsoluteGameResult::P1Win => {
                    net1_wins += 1;
                    pair_result += 2;
                }
                AbsoluteGameResult::Draw => {
                    draws += 1;
                    pair_result += 1;
                }
                AbsoluteGameResult::P2Win => {
                    net2_wins += 1;
                    pair_result += 0;
                }
            }
        }
        let mut lock = state.lock().unwrap();
        if pair_result > 2 {
            lock.results.net1_pair_wins += 1;
        } else if pair_result < 2 {
            lock.results.net2_pair_wins += 1;
        } else {
            lock.results.drawn_pairs += 1;
        }
        lock.results.net1_wins += net1_wins;
        lock.results.net2_wins += net2_wins;
        lock.results.draws += draws;
        lock.pairs_completed += 1;
        drop(lock);
    }
}
pub fn compare_multithreaded(eval1: &Evaluator<impl Backend>, eval2: &Evaluator<impl Backend>, pair_count: Option<usize>, mcts_config: MctsSearchConfig, random_move_count: usize, thread_count: usize, show_progress: bool) -> CompareResult {
    let state = Mutex::new(CompareState {
        current_index: 0,
        pairs_completed: 0,
        results: CompareResult {
            net1_wins: 0,
            draws: 0,
            net2_wins: 0,
            net1_pair_wins: 0,
            drawn_pairs: 0,
            net2_pair_wins: 0,
        },
        required: pair_count,
        stop: false,
    });
    let mut threads = Vec::new();
    let state_ref = &state as *const _;
    for i in 0..thread_count {
        let state_ref = unsafe {&*state_ref};
        let eval1 = eval1.clone();
        let eval2 = eval2.clone();
        threads.push(thread::spawn(move || {
            compare_thread(state_ref, &eval1, &eval2, mcts_config, random_move_count);
        }));
    }
    if show_progress {
        let bar = if let Some(pair_count) = pair_count {
            ProgressBar::new(pair_count as u64).with_style(ProgressStyle::with_template("[{elapsed_precise}] Comparing networks {bar:100.cyan/blue} {pos:>7}/{len:7}\n{msg}").unwrap())
        } else {
            ProgressBar::new(u64::MAX).with_style(ProgressStyle::with_template("[{elapsed_precise}] Comparing networks {pos:>7}\n{msg}").unwrap())
        };
        loop {
            let lock = state.lock().unwrap();
            bar.set_position(lock.pairs_completed as u64);
            if let Some(pair_count) = pair_count {
                if pair_count == lock.pairs_completed {
                    break;
                }
            }
            bar.set_message(format!("{}", lock.results));
            drop(lock);
            thread::sleep(Duration::from_millis(100));
        }
        bar.finish();
    }
    for thread in threads {
        thread.join().unwrap();
    }
    let results = state.lock().unwrap().results;
    results
}
pub enum EloComparisonMode {
    Games,
    Pairs,
    DecisiveGames,
    DecisivePairs,
}
/// Stolen from https://3dkingdoms.com/chess/elo.htm
///
/// Margin of error should be between 0 and 1, 0.95 is recommended. It is used to calculate the elo range
pub fn elo_comparison(result: CompareResult, mode: EloComparisonMode, margin_of_error: f32) -> (f32, f32) {
    fn elo_difference(percent: f32) -> f32{
        -400.0 * (1.0 / percent - 1.0).ln() / LN_10
    }
    fn phi_inv(x: f32) -> f32 {
        let x = 2.0 * x - 1.0;
        let a = 8.0 * (PI - 3.0) / (3.0 * PI * (4.0 - PI));
        let y = (1.0 - x * x).ln();
        let z = 2.0 / (PI * a) + y / 2.0;
        let ret = SQRT_2 * f32::sqrt(f32::sqrt(z * z - y / a) - z);
        if x < 0.0 {-ret} else {ret}
    }
    let (wins, draws, losses) = match mode {
        EloComparisonMode::Games => (result.net1_wins, result.draws, result.net2_wins),
        EloComparisonMode::DecisiveGames => (result.net1_wins, 0, result.net2_wins),
        EloComparisonMode::Pairs => (result.net1_pair_wins, result.drawn_pairs, result.net2_pair_wins),
        EloComparisonMode::DecisivePairs => (result.net1_pair_wins, 0, result.net2_pair_wins)
    };
    let total = wins + draws + losses;
    let score = wins as f32 + draws as f32 / 2.0;
    let percentage = score / total as f32;
    let elo_diff = -400.0 * (1.0 / percentage - 1.0).ln() / LN_10;

    let win_p = wins as f32 / total as f32;
    let draw_p = draws as f32 / total as f32;
    let loss_p = losses as f32 / total as f32;
    let wins_dev = win_p * (1.0 - percentage).powi(2);
    let draws_dev = draw_p * (0.5 - percentage).powi(2);
    let losses_dev = loss_p * (0.0 - percentage).powi(2);
    let std_dev = (wins_dev + draws_dev + losses_dev).sqrt() / (total as f32).sqrt();

    let min_confidence = (1.0 - margin_of_error) / 2.0;
    let max_confidence = 1.0 - min_confidence;
    let dev_min = percentage + phi_inv(min_confidence) * std_dev;
    let dev_max = percentage + phi_inv(max_confidence) * std_dev;

    let difference = elo_difference(dev_max) - elo_difference(dev_min);

    (elo_diff, difference / 2.0)
}