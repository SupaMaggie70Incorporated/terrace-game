// https://github.com/foersterrobert/AlphaZeroFromScratch/blob/main/4.AlphaMCTS.ipynb

use std::{marker::PhantomData, panic::catch_unwind, sync::Mutex, time::{Duration, Instant}};

use burn::tensor::backend::Backend;
use rand::random;

use crate::{ai::{eval::PositionEvaluate, net::{Network, NetworkOutput}}, rules::{self, AbsoluteGameResult, GameResult, Move, Player, TerraceGameState, ALL_POSSIBLE_MOVES}};
pub struct MctsNode {
    mov: Move,
    num_visits: usize,
    /// The total of all network evaluation values, W,D,L
    total_value: (f32, f32, f32),
    /// This should be sorted so that the first elements have a higher policy
    children: Vec<usize>,
    /// Contains the move and the policy, this should be sortest from lowest to highest policy
    expandable_moves: Vec<(Move, f32)>,
    policy: f32,
    is_terminal: bool,
    /*
    /// When a thread is modifying or using this node, this lock should be held by the thread.
    /// When the node is queued to be searched(but not have one of its children searched), the inner value should be set to true so that it is not searched again.
    in_use: Mutex<bool>*/
}
impl MctsNode {
    pub fn total_score(&self) -> f32 {
        evaluation_to_score(self.total_value)
    }
    pub fn average_value(&self) -> (f32, f32, f32) {
        // The +1 comes from the first search we do. Starting with num_visits=1 would mess up the policy exploration calculations, though not by any substantial amount
        (self.total_value.0 / self.num_visits as f32, self.total_value.1 / self.num_visits as f32, self.total_value.2 / self.num_visits as f32)
    }
    pub fn select_child<E: PositionEvaluate>(&self, ctx: *const MctsSearch<E>) -> usize {
        assert!(self.expandable_moves.len() == 0);
        assert!(!self.is_terminal);
        assert!(self.children.len() > 0);
        let search = unsafe {&*ctx};
        let mut best_ucb = -f32::INFINITY;
        let mut best_child = usize::MAX;
        for child_index in &self.children {
            let child = &search.nodes[*child_index];
            // The child score won't be negated because the viewpoint is automatically shifted when evaluating. Therefore, the child's evaluation should be roughly the reciprocal of the parent's evaluation
            // Thus, we simply choose the best move, not the one we like the most.
            let ucb = ucb(child.policy, child.total_score(), child.num_visits, self.num_visits);
            if ucb > best_ucb {
                best_ucb = ucb;
                best_child = *child_index;
            }
        }
        assert!(best_ucb != -f32::INFINITY); // If this is false, then there is some problem causing a negative infinity ucb score
        best_child
    }
    pub fn expand<E: PositionEvaluate>(&mut self, ctx: *mut MctsSearch<E>, eval: &E, mut state: TerraceGameState, transmutations: &[TerraceGameState]) -> (f32, f32, f32) {
        assert!(self.expandable_moves.len() > 0);
        let search = unsafe {&mut *ctx};
        let (mov, policy) = self.expandable_moves.pop().unwrap(); // We have already sorted it to maximize speed
        if mov == Move::NONE {
            panic!("Attempted to make null move: {}, {}, {}", self.expandable_moves.len(), self.children.len(), self.is_terminal);
        }
        // The following code is for in case an invalid move is made, so that we can have a visual representation
        /*if !state.is_move_valid(mov) {
            println!("Attempted to make move: {:?}->{:?}, evaluations done: {}", mov.from.xy(), mov.to.xy(), search.nodes.len());
            crate::gui::run(state);
        }*/
        state.make_move(mov);
        let mut is_transmutation = false;
        for s in transmutations {
            if state.is_transmutation(s) {
                is_transmutation = true;
                break;
            }
        }
        let(value, policy, is_terminal, expandable_moves) = if is_transmutation {
            ((0.0, 0.5, 0.0), 0.0, true, vec![])
        } else if state.result() == GameResult::Ongoing {
            let eval = eval.evaluate_on_position(&state);
            let value_total = eval.0[0] + eval.0[1] + eval.0[2];
            let eval = ([eval.0[0] / value_total, eval.0[1] / value_total, eval.0[2] / value_total], eval.1);
            let mut expandable_vec = Vec::new();
            for (i, mov) in rules::ALL_POSSIBLE_MOVES.iter().enumerate() {
                if state.is_move_valid(*mov) {
                    expandable_vec.push((*mov, eval.1[i]));
                }
            }
            if expandable_vec.len() == 0 {
                expandable_vec.push((Move::SKIP, 1.0));
            }
            expandable_vec.sort_unstable_by(|&(_, a), &(_, b)| {
                a.partial_cmp(&b).unwrap()
            });
            ((eval.0[0], eval.0[1], eval.0[2]), policy, false, expandable_vec)
        } else {
            let value = if state.result().into_absolute() == AbsoluteGameResult::Draw {
                (0.0, 1.0, 0.0)
            } else {
                (0.0, 0.0, 1.0) // Its not possible to win when its not your turn
            };
            let expandable_vec = Vec::new();
            (value, 0.0, true, expandable_vec)
        };
        self.children.push(search.nodes.len());
        search.nodes.push(MctsNode {
            mov,
            num_visits: 1,
            total_value: value,
            children: Vec::new(),
            expandable_moves,
            policy,
            is_terminal
        });
        value
    }
}
pub struct MctsSearch<E: PositionEvaluate> {
    /// The position of the game at the time of searching
    root_position: TerraceGameState,
    config: MctsSearchConfig,
    /// The list of all expanded nodes, with the index being used to identify them
    nodes: Vec<MctsNode>,
    /// The network or whatever is evaluating positions
    eval: *const E,
}
impl<E: PositionEvaluate> MctsSearch<E> {
    pub fn new(root_position: TerraceGameState, config: MctsSearchConfig, eval: *const E) -> Self {
        Self {
            root_position,
            config,
            nodes: Vec::with_capacity(config.initial_list_size),
            eval,
        }
    }
    pub fn search(&mut self) -> MctsSearchResult {
        let mut indices_list = Vec::new();
        let mut transmutations_list = Vec::new();
        let start_time = Instant::now();
        let mut evaluations = 0;
        if self.nodes.len() == 0 {
            if self.root_position.result() != GameResult::Ongoing {
                panic!("Searched already complete position!");
            }
            let eval = unsafe {&*self.eval}.evaluate_on_position(&self.root_position);
            let mut expandable_vec = Vec::new();
            for (i, mov) in rules::ALL_POSSIBLE_MOVES.iter().enumerate() {
                if self.root_position.is_move_valid(*mov) {
                    expandable_vec.push((*mov, eval.1[i]));
                }
            }
            if expandable_vec.len() == 0 {
                expandable_vec.push((Move::SKIP, 1.0));
            }
            expandable_vec.sort_unstable_by(|&(_, a), &(_, b)| {
                a.partial_cmp(&b).unwrap()
            });
            let value = (eval.0[0], eval.0[1], eval.0[2]);
            let policy = 1.0;
            self.nodes.push(MctsNode {
                mov: Move::NONE,
                num_visits: 1,
                total_value: value,
                children: Vec::new(),
                expandable_moves: expandable_vec,
                policy, // This doesn't really matter
                is_terminal: false
            });
            evaluations += 1;
        }
        loop {
            // Detect stop condition
            match self.config.stop_condition {
                MctsStopCondition::TotalEvaluations(num) => {
                    if self.nodes[0].num_visits >= num {
                        break;
                    }
                }
                MctsStopCondition::MaxTime(time) => {
                    if (Instant::now() - start_time) >= time {
                        break;
                    }
                }
                MctsStopCondition::Evaluations(num) => {
                    if evaluations >= num {
                        break;
                    }
                }
            }
            // Selection
            let ctx = self as *mut Self;
            let mut node_index = 0;
            let mut node = &mut self.nodes[0];
            let mut state = self.root_position;
            transmutations_list.clear();
            transmutations_list.push(state);
            indices_list.clear();
            indices_list.push(0);
            loop {
                if node.expandable_moves.len() != 0 {
                    break;
                }
                if node.is_terminal {
                    break;
                }
                assert!(node.children.len() > 0);
                node_index = node.select_child(ctx);
                assert!(node_index != 0);
                node = &mut self.nodes[node_index];
                let mov = node.mov;
                state.make_move(mov);
                indices_list.push(node_index);
                transmutations_list.push(state);
            }
            // Expansion, simulation
            let value = if node.is_terminal {
                indices_list.pop();
                node.total_value
            } else {
                node.expand(ctx, unsafe {&*self.eval}, state, &transmutations_list)
            };
            evaluations += 1; // If we put this in there it will sometimes fail when the entire search space is searchable
            // Backpropagation
            for (i, index) in indices_list.iter().rev().enumerate() {
                let node = &mut self.nodes[*index];
                node.num_visits += 1;
                if i % 2 == 1 {
                    node.total_value.0 += value.0;
                    node.total_value.1 += value.1;
                    node.total_value.2 += value.2;
                } else { // For example, the last node before expanding. This should have the opposite of the value
                    node.total_value.0 += value.2;
                    node.total_value.1 += value.1;
                    node.total_value.2 += value.0;
                }
            }
        }
        let mut best_move = Move::NONE;
        let mut best_num_visits = -f32::INFINITY;
        let mut best_value = f32::NEG_INFINITY;
        for index in &self.nodes[0].children {
            let node = &self.nodes[*index];
            let num_visits = (1.0 + (random::<f32>() * 2.0 - 1.0) * self.config.policy_deviance) * node.num_visits as f32;
            let value = (1.0 + (random::<f32>() * 2.0 - 1.0) * self.config.policy_deviance) * -evaluation_to_score(node.average_value());
            if !self.config.use_value { // Select based on number of visits
                if num_visits > best_num_visits {
                    best_move = node.mov;
                    best_num_visits = num_visits;
                    best_value = value;
                } else if num_visits == best_num_visits && value > best_value {
                    best_move = node.mov;
                    best_num_visits = num_visits;
                    best_value = value;
                }
            } else { // Select based on value, we choose the lowest because it is evaluated from the POV of the opponent
                if value > best_value {
                    best_move = node.mov;
                    best_num_visits = num_visits;
                    best_value = value;
                }
            }
        }
        let value = self.nodes[0].average_value();
        let time = Instant::now() - start_time;
        MctsSearchResult {
            mov: best_move,
            value,
            evaluations,
            time
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct MctsSearchResult {
    pub mov: Move,
    pub value: (f32, f32, f32),
    pub evaluations: usize,
    pub time: Duration,
}
#[derive(Clone, Copy, Debug)]
pub enum MctsStopCondition {
    TotalEvaluations(usize),
    Evaluations(usize),
    MaxTime(Duration),
}
#[derive(Clone, Copy, Debug)]
pub struct MctsSearchConfig {
    pub stop_condition: MctsStopCondition,
    pub initial_list_size: usize,
    pub use_value: bool,
    pub policy_deviance: f32,
}
pub fn evaluation_to_score(eval: (f32, f32, f32)) -> f32 {
    let (win, draw, loss) = eval;
    (win - loss) / 2.0 + 0.5
}
const UCB_CONST: f32 = 2.0; // Traditionally this is SQRT_2
pub fn ucb(policy: f32, total_score: f32, num_visits: usize, parent_visits: usize) -> f32 {
    total_score / num_visits as f32 + UCB_CONST * policy * ((parent_visits as f32).ln() / num_visits as f32).sqrt()
}