use std::{cmp::Ordering, f32::consts::{LN_10, PI, SQRT_2}, marker::PhantomData, time::Instant};

use burn::{nn::loss, tensor::backend::Backend};
use indicatif::{ProgressBar, ProgressStyle};
use rand::{random, Rng};

use crate::{ai::net::{NetworkInput, NetworkOutput}, mcts::{MctsSearch, MctsSearchConfig}, rules::{self, AbsoluteGameResult, GameResult, Move, Player, TerraceGameState}};
use crate::rules::NUM_POSSIBLE_MOVES;

use super::net::Network;

pub trait PositionEvaluate: Clone + Send {
    fn evaluate_on_position(&self, state: &TerraceGameState) -> ([f32; 3], [f32; rules::NUM_POSSIBLE_MOVES]);
    fn evaluate_on_positions(&self, states: &[TerraceGameState]) -> Vec<([f32; 3], [f32; rules::NUM_POSSIBLE_MOVES])>;
    fn find_best_value(&self, state: TerraceGameState) -> Move {
        let mut legal_moves = Vec::new();
        state.generate_moves(&mut legal_moves);
        let mut best_mov = Move::NONE;
        let mut best_value = f32::NEG_INFINITY;
        for mov in legal_moves {
            let mut new_state = state;
            new_state.make_move(mov);
            let score = if new_state.result() == GameResult::Ongoing {
                let value = self.evaluate_on_position(&new_state).0;
                value[2] - value[0]
            } else {
                match new_state.result().into_absolute() {
                    AbsoluteGameResult::P1Win | AbsoluteGameResult::P2Win => 2.0,
                    AbsoluteGameResult::Draw => 0.0
                }
            };
            if score > best_value {
                best_value = score;
                best_mov = mov;
            }
        }
        best_mov
    }
}
#[derive(Clone, Copy)]
pub struct RandomEvaluator;
impl RandomEvaluator {
    const MAX_POLICY: f32 = 20.0;
}
impl Default for RandomEvaluator {
    fn default() -> Self {
        Self
    }
}
impl PositionEvaluate for RandomEvaluator {
    fn evaluate_on_position(&self, state: &TerraceGameState) -> ([f32; 3], [f32; rules::NUM_POSSIBLE_MOVES]) {
        let value: [f32; 3] = [random(), random(), random()];
        let value_total = value[0] + value[1] + value[2];
        let value = [value[0] / value_total, value[1] / value_total, value[2] / value_total];
        let mut policy = [0.0; rules::NUM_POSSIBLE_MOVES];
        for i in 0..rules::NUM_POSSIBLE_MOVES {
            policy[i] = Self::MAX_POLICY.powf(random::<f32>())
        }
        (value, policy)
    }
    fn evaluate_on_positions(&self, states: &[TerraceGameState]) -> Vec<([f32; 3], [f32; rules::NUM_POSSIBLE_MOVES])> {
        let mut out = Vec::with_capacity(states.len());
        for _ in 0..states.len() {
            let value: [f32; 3] = [random(), random(), random()];
            let value_total = value[0] + value[1] + value[2];
            let value = [value[0] / value_total, value[1] / value_total, value[2] / value_total];
            let mut policy = [0.0; rules::NUM_POSSIBLE_MOVES];
            for i in 0..rules::NUM_POSSIBLE_MOVES {
                policy[i] = Self::MAX_POLICY.powf(random::<f32>())
            }
            out.push((value, policy));
        }
        out
    }
}
#[derive(Copy, Clone)]
pub struct HceEvaluator;
impl Default for HceEvaluator {
    fn default() -> Self {
        Self
    }
}
impl PositionEvaluate for HceEvaluator {
    fn evaluate_on_position(&self, state: &TerraceGameState) -> ([f32; 3], [f32; NUM_POSSIBLE_MOVES]) {
        let eval = crate::eval_hce::eval_board(state, &mut Vec::new());
        let sigmoid = 1.0 / (1.0 + (eval / 100.0).exp());
        let value = if sigmoid < 0.5 {
            [0.0, 0.0, (0.5 - sigmoid)]
        } else {
            [(sigmoid - 0.5) * 2.0, 0.0, 0.0]
        };
        let policy = [1.0; NUM_POSSIBLE_MOVES];
        (value, policy)
    }
    fn evaluate_on_positions(&self, states: &[TerraceGameState]) -> Vec<([f32; 3], [f32; NUM_POSSIBLE_MOVES])> {
        let mut vec = Vec::with_capacity(states.len());
        for state in states {
            vec.push(self.evaluate_on_position(state));
        }
        vec
    }
}

