use std::{cmp::Ordering, f32::consts::{LN_10, PI, SQRT_2}, marker::PhantomData, time::Instant};
use std::ops::Deref;

use burn::{nn::loss, tensor::backend::Backend};
use indicatif::{ProgressBar, ProgressStyle};
use rand::{random, thread_rng, Rng};

use crate::{ai::net::{NetworkInput, NetworkOutput}, mcts::{MctsSearch, MctsSearchConfig}, rules::{self, AbsoluteGameResult, GameResult, Move, Player, TerraceGameState}};
use crate::rules::NUM_POSSIBLE_MOVES;

use super::net::{Network, NetworkEnum};

#[derive(Clone, Debug)]
pub enum Evaluator<B: Backend> {
    Random,
    Hce,
    Network(NetworkEnum<B>)
}
unsafe impl<B: Backend> Send for Evaluator<B> {}
impl<B: Backend> Evaluator<B> {
    pub fn random() -> Self {
        Self::Random
    }
    pub fn hce() -> Self {
        Self::Hce
    }
    pub fn network(net: NetworkEnum<B>) -> Self {
        Self::Network(net)
    }
    pub fn load(desc: &str, data_dir: &str, dev: &B::Device) -> Self{
        match desc {
            "random" => Self::Random,
            "hce" => Self::Hce,
            desc => {
                Self::Network(NetworkEnum::<B>::load(desc, dev, data_dir))
            }
        }
    }
    const RANDOM_MAX_POLICY: f32 = 20.0;
    pub fn evaluate_on_position(&self, state: &TerraceGameState) -> ([f32; 3], [f32; rules::NUM_POSSIBLE_MOVES]) {
        match self {
            Self::Random => {
                let value: [f32; 3] = [random(), random(), random()];
                let value_total = value[0] + value[1] + value[2];
                let value = [value[0] / value_total, value[1] / value_total, value[2] / value_total];
                let mut policy = [0.0; rules::NUM_POSSIBLE_MOVES];
                for i in 0..rules::NUM_POSSIBLE_MOVES {
                    policy[i] = Self::RANDOM_MAX_POLICY.powf(random::<f32>())
                }
                (value, policy)
            },
            Self::Hce => {
                let eval = crate::eval_hce::eval_board(state, &mut Vec::new());
                let sigmoid = 1.0 / (1.0 + (eval / 10000.0).exp());
                let value = if sigmoid < 0.5 {
                    [0.0, 0.0, (0.5 - sigmoid)]
                } else {
                    [(sigmoid - 0.5) * 2.0, 0.0, 0.0]
                };
                let policy = [1.0; NUM_POSSIBLE_MOVES];
                (value, policy)
            },
            Self::Network(net) => {
                let output = net.forward(NetworkInput::<B>::from_state(state, &net.device()));
                let value = output.get_single_probabilities();
                let policy = output.get_single_policies();
                (value, policy)
            }
        }
    }
    pub fn evaluate_on_positions(&self, states: &[TerraceGameState]) -> Vec<([f32; 3], [f32; rules::NUM_POSSIBLE_MOVES])> {
        match self {
            Self::Random | Self::Hce => {
                let mut vec = Vec::with_capacity(states.len());
                for state in states {
                    vec.push(self.evaluate_on_position(state));
                }
                vec
            }
            Self::Network(net) => {
                let output = net.forward(NetworkInput::<B>::from_states(states, &net.device()));
                output.get_values_policies()
            }
        }
    }
    pub fn find_best_value(&self, state: TerraceGameState, policy_deviance: f32) -> Move {
        match self {
            Self::Random => {
                let mut legal_moves = Vec::new();
                state.generate_moves(&mut legal_moves);
                legal_moves[thread_rng().gen_range(0..legal_moves.len())]
            }
            Self::Hce => {
                let mut legal_moves = Vec::new();
                state.generate_moves(&mut legal_moves);
                let mut best_mov = Move::NONE;
                let mut best_value = f32::NEG_INFINITY;
                for mov in legal_moves {
                    let mut new_state = state;
                    new_state.make_move(mov);
                    let score = if new_state.result() == GameResult::Ongoing {
                        -crate::eval_hce::eval_board(&new_state, &mut Vec::new()) * (1.0 + policy_deviance * 2.0 * (random::<f32>() - 0.5))
                        /*let value = self.evaluate_on_position(&new_state).0;
                        ((value[2] - value[0]) + ((random::<f32>() * 2.0 - 1.0) * policy_deviance)).clamp(-1.0, 1.0)*/
                    } else {
                        match new_state.result().into_absolute() {
                            AbsoluteGameResult::P1Win | AbsoluteGameResult::P2Win => f32::INFINITY,
                            AbsoluteGameResult::Draw => f32::NEG_INFINITY
                        }
                    };
                    if score >= best_value {
                        best_value = score;
                        best_mov = mov;
                    }
                }
                best_mov
            }
            Self::Network(net) => {
                let mut legal_moves = Vec::new();
                state.generate_moves(&mut legal_moves);
                let mut best_mov = Move::NONE;
                let mut best_value = f32::NEG_INFINITY;
                for mov in legal_moves {
                    let mut new_state = state;
                    new_state.make_move(mov);
                    let score = if new_state.result() == GameResult::Ongoing {
                        let value = self.evaluate_on_position(&new_state).0;
                        ((value[2] - value[0]) + ((random::<f32>() * 2.0 - 1.0) * policy_deviance)).clamp(-1.0, 1.0)
                    } else {
                        match new_state.result().into_absolute() {
                            AbsoluteGameResult::P1Win | AbsoluteGameResult::P2Win => f32::INFINITY,
                            AbsoluteGameResult::Draw => f32::NEG_INFINITY
                        }
                    };
                    if score >= best_value {
                        best_value = score;
                        best_mov = mov;
                    }
                }
                best_mov
            }
        }
    }
}