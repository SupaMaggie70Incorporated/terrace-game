use std::f32::INFINITY;

use rand::{random, thread_rng, Rng};

use crate::rules::{AbsoluteGameResult, GameResult, Move, Piece, PieceType, Player, Square, TerraceGameState};


pub fn sample_board(b: &TerraceGameState, x: u8, y: u8) -> Piece {
    if b.player_to_move() == Player::P2 {
        let pc = b.square(Square::from_xy((7 - x,7 - y)));
        if pc.is_any() {
            Piece::new(pc.typ(), pc.player().other())
        } else {
            pc
        }
    } else {
        b.square(Square::from_xy((x, y)))
    }
}
pub fn move_policy(board: &TerraceGameState, mov: Move) -> f32 {
    let mut interesting = false;
    let mut overall = 1.0;
    let (from_x, from_y) = mov.from.xy();
    let (to_x, to_y) = mov.to.xy();
    let pc = board.square(mov.from);
    let capture = board.square(mov.to);
    if capture.is_any() {
        if capture.player() != board.player_to_move() {
            match capture.typ() {
                PieceType::T => overall = f32::INFINITY,
                PieceType::S1 => overall *= 1.1,
                PieceType::S2 => {
                    interesting = true;
                    overall *= 1.5;
                }
                PieceType::S3 => {
                    interesting = true;
                    overall *= 2.5;
                }
                PieceType::S4 => {
                    interesting = true;
                    overall *= 4.0;
                }
            }
        } else {
            match capture.typ() {
                PieceType::T => unreachable!(),
                PieceType::S1 => {
                    interesting = true;
                    overall *= 1.0;
                }
                PieceType::S2 => overall *= 3.0,
                PieceType::S3 => overall *= 0.5,
                PieceType::S4 => overall *= 0.2,
            }
        }
    }

    if interesting {overall * 5.0} else {overall}
}
pub fn eval_board(board: &TerraceGameState, legal_move_policies: &mut Vec<(Move, f32)>) -> f32 {
    if board.result() != GameResult::Ongoing {
        let mut res = board.result().into_absolute();
        if board.player_to_move() == Player::P2 {res = res.other()};
        return match res {
            crate::rules::AbsoluteGameResult::P1Win => INFINITY,
            crate::rules::AbsoluteGameResult::Draw => 0.0,
            AbsoluteGameResult::P2Win => -INFINITY
        }
    }
    /*for x in 0..8 {
        for y in 0..8 {
            let from = Square::from_xy((x, y));
            if board.square(from).is_player(Player::P1) {
                for x2 in 0..8 {
                    for y2 in 0..8 {
                        let mov = Move::new(from, Square::from_xy((x2, y2)));
                        if board.is_move_valid(mov) {
                            legal_move_policies.push((mov, move_policy(board, mov)));
                        }
                    }
                }
            }
        }
    }*/
    let mut piece_score = 0.0;
    let mut defensive_score = 0.0;
    let mut offensive_score = 0.0;
    for x in 0..8 {
        for y in 0..8 {
            let p = sample_board(&board, x, y);
            if !p.is_any() {continue};
            let (p_x, p_y) = if p.player() == Player::P2 {(7 - x, 7 - y)} else {(x, y)};
            let dist_from_bottom_left = p_x.max(p_y);
            let dist_from_top_right = 7 - p_x.min(p_y);
            let mut piece_value: f32 = 0.0;
            let mut offensive_value: f32 = 0.0;
            let mut defensive_value: f32 = 0.0;
            match p.typ() {
                PieceType::T => {
                    // Penalize being far up the board
                    let y_penalty = match p_y {
                        0 => 0.0,
                        1 => -0.5,
                        2 => -1.5,
                        3 => -2.5,
                        4 => -4.5,
                        5 => -7.0,
                        6 => -10.0,
                        7 => -15.0,
                        _ => unreachable!()
                    };
                    // Reward running towards the higher corner, where you are invincible
                    let x_penalty = match p_x {
                        0 => 0.0,
                        1 => 0.1,
                        2 => 0.2,
                        3 => 0.4,
                        4 => 0.6,
                        5 => 1.0,
                        6 => 1.4,
                        7 => 2.0,
                        _ => unreachable!()
                    };
                    piece_value = y_penalty + x_penalty;
                }
                PieceType::S1 => {
                    piece_value = match p_y {
                        0 => 0.25,
                        1 => 0.2,
                        2 => 0.15,
                        3 => 0.1,
                        _ => 0.0
                    };
                    defensive_value = match dist_from_bottom_left {
                        0 => 0.5,
                        1 => 0.3,
                        2 => 0.3,
                        3 => 0.2,
                        _ => 0.0
                    };
                    if dist_from_bottom_left >= 4 && dist_from_bottom_left >= 4 {
                        piece_value -= dist_from_bottom_left.min(dist_from_top_right) as f32 * 0.025;
                    }
                }
                PieceType::S2 => {
                    piece_value = match p_y {
                        0 => 0.5,
                        1 => 0.4,
                        2 => 0.3,
                        _ => 0.2
                    };
                    defensive_value = match dist_from_bottom_left {
                        0 => 1.0,
                        1 => 0.5,
                        2 => 0.5,
                        3 => 0.35,
                        _ => 0.0
                    };
                    if dist_from_bottom_left >= 4 && dist_from_bottom_left >= 4 {
                        piece_value -= dist_from_bottom_left.min(dist_from_top_right) as f32 * 0.05;
                    }
                }
                PieceType::S3 => {
                    piece_value = 4.0;
                    defensive_value = match dist_from_bottom_left {
                        0 => 2.5,
                        1 => 1.5,
                        2 => 2.0,
                        3 => 1.0,
                        _ => 0.0
                    };
                    offensive_value = match dist_from_top_right {
                        0 => unreachable!(),
                        1 => 5.0,
                        2 => 3.0,
                        3 => 2.0,
                        _ => 0.0
                    };
                    if dist_from_bottom_left >= 4 && dist_from_bottom_left >= 4 {
                        piece_value -= dist_from_bottom_left.min(dist_from_top_right) as f32 * 0.4;
                    }
                }
                PieceType::S4 => {
                    piece_value = 10.0;
                    defensive_value = match dist_from_bottom_left {
                        0 => -1.0,
                        1 => 3.0,
                        2 => 3.0,
                        3 => 4.0,
                        _ => 0.0
                    };
                    offensive_value = match dist_from_top_right {
                        0 => unreachable!(),
                        1 => 10.0,
                        2 => 5.0,
                        3 => 3.0,
                        _ => 0.0
                    };
                    if dist_from_bottom_left >= 4 && dist_from_bottom_left >= 4 {
                        piece_value -= dist_from_bottom_left.min(dist_from_top_right) as f32 * 0.5;
                    }
                }
            };
            let size = p.size();
            if p.player() == Player::P1 {
                piece_score += piece_value;
                defensive_score += defensive_value;
                offensive_score += offensive_value;
                for offset in [(-1, -1), (-1, 1), (1, 1), (1, -1)] {
                    let (x2, y2) = (x as i8 + offset.0, y as i8 + offset.1);
                    if x2 >= 0 && x2 < 8 && y2 >= 0 && y2 < 8 {
                        let sq2 = Square::from_xy((x2 as u8, y2 as u8));
                        
                    }
                }
            } else {
                piece_score -= piece_value;
                defensive_score -= offensive_value;
                offensive_score -= defensive_value;
            }
        }
    }
    defensive_score.min(0.0) + offensive_score.max(0.0) + piece_score
}
pub fn search_blanket_depth1(board: &TerraceGameState) -> (Move, f32) {
    if board.result() != GameResult::Ongoing {
        let mut res = board.result().into_absolute();
        if board.player_to_move() == Player::P2 {res = res.other()};
        return match res {
            AbsoluteGameResult::P1Win => (Move::NONE, INFINITY),
            AbsoluteGameResult::Draw => (Move::NONE, 0.0),
            AbsoluteGameResult::P2Win => (Move::NONE, -INFINITY)
        }
    }
    let mut legal_moves = Vec::new();
    board.generate_moves(&mut legal_moves);
    let mut best = (0, f32::INFINITY);
    let mut unused = Vec::new();
    for (i, mov) in legal_moves.iter().enumerate() {
        let mut b = *board;
        b.make_move(*mov);
        unused.clear();
        let value = eval_board(&b, &mut unused);
        if value < best.1 {
            best = (i, value)
        }
    }
    (legal_moves[best.0], -best.1)
}
pub fn search_blanket_depth2(board: &TerraceGameState) -> (Move, f32) {
    let mut legal_moves = Vec::new();
    board.generate_moves(&mut legal_moves);
    let mut best = (0, f32::INFINITY);
    for (i, mov) in legal_moves.iter().enumerate() {
        let mut b = *board;
        b.make_move(*mov);
        let (_, value) = search_blanket_depth1(&b);
        if value < best.1 {
            best = (i, value)
        }
    }
    (legal_moves[best.0], -best.1)
}
pub fn search_blanket_depth(board: &TerraceGameState, d: usize, legal_move_policies: &mut Vec<(Move, f32)>) -> (Move, f32) {
    if board.result() != GameResult::Ongoing {
        let mut res = board.result().into_absolute();
        if board.player_to_move() == Player::P2 {res = res.other()};
        return match res {
            AbsoluteGameResult::P1Win => (Move::NONE, INFINITY),
            AbsoluteGameResult::Draw => (Move::NONE, 0.0),
            AbsoluteGameResult::P2Win => (Move::NONE, -INFINITY)
        }
    }
    if d == 0 {
        legal_move_policies.clear();
        return (Move::NONE, eval_board(board, legal_move_policies));
    }
    let mut legal_moves = Vec::new();
    board.generate_moves(&mut legal_moves);
    let mut best = (0, f32::INFINITY);
    for (i, mov) in legal_moves.iter().enumerate() {
        let mut b = *board;
        b.make_move(*mov);
        let (_, value) = search_blanket_depth(&b, d - 1, legal_move_policies);
        if value < best.1 {
            best = (i, value)
        }
    }
    (legal_moves[best.0], -best.1)
}
pub struct FullySearchedPositionInfo {
    children_index: usize,
    num_children: usize,
    selected_child_by_value: Option<usize>,
    selected_child_by_policy: usize,
    total_children_policy: f32,
    value: f32,
}
pub struct SearchPositionInfo {
    /// Contains the children first index, the number of children, if the children are fully evaluated, the total policy of the children, and the value of the node itself
    children_value: Option<FullySearchedPositionInfo>,
    /// This should be equal to the highest policy of the children, such that when all interesting continuations of a move have been examined, it is no longer searched
    policy: f32,
    mov: Move,
}
impl SearchPositionInfo {
    pub fn search(&mut self, board: &mut TerraceGameState, search_ptr: *mut SearchInfo) {
        let mut search = unsafe {&mut *search_ptr};
        if search.total_evaluations >= search.max_evaluations {
            return;
        }
        if let Some(full_info) = &self.children_value {
            if !full_info.selected_child_by_value.is_some() {
                let mut max_policy = f32::NEG_INFINITY;
                for i in 0..full_info.num_children {
                    let child = &mut search.table[full_info.children_index + i];
                    if child.children_value.is_none() {
                        let undo = board.make_move(child.mov);
                        child.search(board, search_ptr);
                        board.undo_move(undo);
                    }
                }
                self.policy = max_policy;
            } else {
                let rand = thread_rng().gen_range(0.0..full_info.total_children_policy);
                let mut total = 0.0;
                for i in 0..full_info.num_children {
                    let child = &mut search.table[full_info.children_index + i];
                    total += child.policy;
                    if total >= rand {
                        let undo = board.make_move(child.mov);
                        child.search(board, search_ptr);
                        board.undo_move(undo);
                        break;
                    }
                }
            }
        } else {
            let children_index = search.policy_vec.len();
            search.policy_vec.clear();
            let value = eval_board(board, &mut search.policy_vec);
            search.total_evaluations += 1;
            let num_children = search.policy_vec.len();
            let mut total_policy = 0.0;
            let mut best = (f32::NEG_INFINITY, 0);
            for (mov, policy) in &search.policy_vec {
                if *policy > best.0 {
                    best = (*policy, search.policy_vec.len())
                }
                total_policy += *policy;
                search.table.push(SearchPositionInfo {
                    children_value: None,
                    policy: *policy,
                    mov: *mov
                });
            }
            self.children_value = Some(FullySearchedPositionInfo {
                children_index,
                num_children,
                selected_child_by_value: None,
                selected_child_by_policy: best.1,
                total_children_policy: total_policy,
                value
            });
            self.policy = best.0;
        }
    }
}
pub struct SearchInfo {
    root: TerraceGameState,
    table: Vec<SearchPositionInfo>,
    policy_vec: Vec<(Move, f32)>,
    total_evaluations: usize,
    max_evaluations: usize,
}
impl SearchInfo {
    pub fn new(root: TerraceGameState, initial_search_capacity: usize) -> Self {
        let mut s = Self {
            root,
            table: Vec::with_capacity(initial_search_capacity),
            policy_vec: Vec::with_capacity(256),
            total_evaluations: 0,
            max_evaluations: 0
        };
        s.table.push(SearchPositionInfo {
            children_value: None,
            policy: 1.0,
            mov: Move::new(Square::from_xy((0,0)), Square::from_xy((0,0)))
        });
        s.go(1);
        s
    }
    pub fn go(&mut self, num_searches: usize) {
        self.max_evaluations = num_searches;
        self.total_evaluations = 0;
        let mut board = self.root;
        let self_ptr = self as *mut Self;
        while self.total_evaluations < self.max_evaluations {
            self.table[0].search(&mut board, self_ptr);
        }
    }
    pub fn evaluation(&mut self) -> f32 {
        self.table[0].children_value.as_ref().unwrap().value
    }
    pub fn best_move(&mut self) -> Move {
        let full = self.table[0].children_value.as_ref().unwrap();
        let mut best = (f32::NEG_INFINITY, Move::new(Square::from_xy((0,0)), Square::from_xy((0,0))));
        if let Some(best) = full.selected_child_by_value {
            self.table[best].mov
        } else {
            self.table[full.selected_child_by_policy].mov
        }
    }
}