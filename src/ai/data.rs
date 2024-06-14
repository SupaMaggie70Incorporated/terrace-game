use std::{fs, fs::File, io::{BufReader, BufWriter, Read, Write}, path::{Path, PathBuf}, sync::Mutex, thread, time::Duration};
use std::sync::mpsc::{channel, Receiver, Sender};

use burn::{data, tensor::backend::Backend};
use burn::prelude::{Bool, Data, Int, Shape, Tensor};
use indicatif::{ProgressBar, ProgressStyle};
use rand::{seq::SliceRandom, Rng};

use crate::{mcts::{MctsSearch, MctsSearchConfig}, rules::{AbsoluteGameResult, GameResult, Move, Piece, PieceType, Player, Square, TerraceGameState}, rules};
use crate::ai::net::{INPUT_CHANNELS, POLICY_OUTPUT_SIZE};

use super::{eval::PositionEvaluate, net::{Network, NetworkInput, NetworkTarget}};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PositionInfo {
    pub bytes: [u8; Self::SIZE],
}
impl PositionInfo {
    pub const SIZE: usize = 34;
    fn piece_to_u8(p: Piece) -> u8 {
        if !p.is_any() {return 0}
        let a = match p.typ() {
            PieceType::T => 0,
            PieceType::S1 => 1,
            PieceType::S2 => 2,
            PieceType::S3 => 3,
            PieceType::S4 => 4
        } + if p.is_player(Player::P2) {9} else {1};
        a
    }
    fn piece_from_u8(p: u8) -> Piece {
        if p == 0 {return Piece::NONE};
        let p = p - 1;
        let player = if (p & 8) == 0 {Player::P1} else {Player::P2};
        let typ = match p & 7 {
            0 => PieceType::T,
            1 => PieceType::S1,
            2 => PieceType::S2,
            3 => PieceType::S3,
            4 => PieceType::S4,
            _ => panic!("Invalid piece!")
        };
        Piece::new(typ, player)
    }
    pub fn new(state: TerraceGameState, result: AbsoluteGameResult, mov: Move) -> Self {
        let flip = state.player_to_move() == Player::P2;
        let result = match if flip {result.other()} else {result} {
            AbsoluteGameResult::P1Win => 0,
            AbsoluteGameResult::Draw => 1,
            AbsoluteGameResult::P2Win => 2,
        };
        let mut bytes = [0; Self::SIZE];
        for x in 0..8 {
            for y_base in 0..4 {
                // Where y_base is 0-3
                // square at (x,2 * y_base) is from (x + 8 * y_base)
                // y is 0-6
                // square at (x, y) is from (x + 4 * y)
                bytes[x + y_base * 8] = if flip {
                    let x1 = 7 - x as u8;
                    let y1 = 7 - y_base as u8 * 2;
                    let y2 = 6 - y_base as u8 * 2;
                    Self::piece_to_u8(state.square(Square::from_xy((x1, y1)))) | (Self::piece_to_u8(state.square(Square::from_xy((x1, y2)))) << 4)
                } else {
                    let x1 = x as u8;
                    let y1 = y_base as u8 * 2;
                    let y2 = y_base as u8 * 2 + 1;
                    Self::piece_to_u8(state.square(Square::from_xy((x1, y1)))) | (Self::piece_to_u8(state.square(Square::from_xy((x1, y2)))) << 4)
                }
            }
        }
        bytes[32] = result + (mov.from.index() << 2);
        bytes[33] = mov.to.index();
        Self {
            bytes
        }
    }
    pub fn to_state(&self) -> (TerraceGameState, AbsoluteGameResult, Move) {
        let mut state = TerraceGameState::setup_new();
        let result = match self.bytes[32] & 3 {
            0 => AbsoluteGameResult::P1Win,
            1 => AbsoluteGameResult::Draw,
            2 => AbsoluteGameResult::P2Win,
            _ => panic!("Invalid state passed")
        };
        let from = Square::from_index(self.bytes[32] >> 2);
        let to = Square::from_index(self.bytes[33]);
        for x in 0..8 {
            for y_base in 0..4 {
                let sq1 = Square::from_xy((x, y_base * 2));
                let sq2 = Square::from_xy((x, y_base * 2 + 1));
                let v = self.bytes[x as usize + y_base as usize * 8];
                let p1 = Self::piece_from_u8(v & 15);
                let p2 = Self::piece_from_u8(v >> 4);
                *state.square_mut(sq1) = p1;
                *state.square_mut(sq2) = p2;
            }
        }
        (state, result, Move {from, to})
    }
    pub fn get_piece_at(&self, x: usize, y: usize) -> Piece {
        let byte = self.bytes[x + (y / 2) * 8];
        if y % 2 == 1 {
            Self::piece_from_u8(byte >> 4)
        } else {
            Self::piece_from_u8(byte & 15)
        }
    }
    pub fn get_game_result(&self) -> AbsoluteGameResult {
        match self.bytes[32] & 3 {
            0 => AbsoluteGameResult::P1Win,
            1 => AbsoluteGameResult::Draw,
            2 => AbsoluteGameResult::P2Win,
            _ => panic!("Invalid game result in training data")
        }
    }
    pub fn get_move(&self) -> Move {
        Move {
            from: Square::from_index(self.bytes[32] >> 2),
            to: Square::from_index(self.bytes[33])
        }
    }
}
pub fn load_data_from_file<P: AsRef<Path>, B: Backend>(path: P, dev: &B::Device) -> (NetworkInput<B>, NetworkTarget<B>) {
    let file = File::open(path).unwrap();
    let num_datapoints = file.metadata().unwrap().len() / PositionInfo::SIZE as u64;
    let mut reader = BufReader::new(file);
    let mut data = Vec::new();
    for _ in 0..num_datapoints {
        let mut datapoint = PositionInfo {bytes: [0; PositionInfo::SIZE]};
        reader.read_exact(&mut datapoint.bytes).unwrap();
        data.push(datapoint);
    }
    data.as_slice().to_training_values(dev)
}
struct MultithreadedDataCollectionState {
    required: Option<usize>,
    current: usize,
    completed: usize,
    stop: bool,
}
pub fn generate_random_data<P: AsRef<Path>>(path: P, datapoint_count: usize, max_datapoints_per_game: usize, first_valid_move: usize) -> Result<(), std::io::Error> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let mut mov_vec = Vec::new();
    let mut states_vec = Vec::new();
    let mut num_written = 0;
    while num_written < datapoint_count {
        states_vec.clear();
        let mut state = TerraceGameState::setup_new();
        while state.result() == GameResult::Ongoing {
            mov_vec.clear();
            state.generate_moves(&mut mov_vec);
            if mov_vec.len() == 0 {
                state.skip_turn();
                state.generate_moves(&mut mov_vec);
            }
            let mov = mov_vec[rand::thread_rng().gen_range(0..mov_vec.len())];
            if state.move_number() as usize >= first_valid_move {
                states_vec.push((state, mov));
            }
            state.make_move(mov);
        }
        states_vec.shuffle(&mut rand::thread_rng());
        let result = state.result().into_absolute();
        for i in 0..states_vec.len().min(max_datapoints_per_game) {
            if num_written >= datapoint_count {
                break;
            }
            writer.write_all(&PositionInfo::new(states_vec[i].0, result, states_vec[i].1).bytes)?;
            num_written += 1;
        }
    }
    writer.flush()?;
    Ok(())
}
fn random_data_writer_thread(mut folder: PathBuf, state: &Mutex<MultithreadedDataCollectionState>, datapoints_per_file: usize, max_datapoints_per_game: usize, first_valid_move: usize) {
    loop {
        let mut lock = state.lock().unwrap();
        if Some(lock.current) == lock.required || lock.stop {
            break;
        }
        let i = lock.current;
        lock.current += 1;
        drop(lock);
        folder.push(format!("{}.bin", i));
        generate_random_data(&folder, datapoints_per_file, max_datapoints_per_game, first_valid_move).unwrap();
        folder.pop();
        let mut lock = state.lock().unwrap();
        lock.completed += 1;
        drop(lock);
    }
}
pub fn generate_random_data_multithreaded<P: AsRef<Path>>(dir: P, thread_count: usize, num_files: Option<usize>, show_progress: bool, datapoints_per_file: usize, max_datapoints_per_game: usize, first_valid_move: usize) {
    let state = Mutex::new(MultithreadedDataCollectionState {
        required: num_files,
        current: 0,
        completed: 0,
        stop: false
    });
    let mut threads = Vec::new();
    let state_ref = &state as *const _;
    for i in 0..thread_count {
        let folder = dir.as_ref().to_owned();
        let state_ref = unsafe {&*state_ref};
        threads.push(thread::spawn(move || {
            random_data_writer_thread(folder, state_ref, datapoints_per_file, max_datapoints_per_game, first_valid_move);
        }));
    }
    if show_progress {
        let bar = if let Some(files) = num_files {
            ProgressBar::new(files as u64).with_style(ProgressStyle::with_template("[{elapsed_precise}] {bar:100.cyan/blue} {pos:>7}/{len:7} {msg}").unwrap())
        } else {
            ProgressBar::new(u64::MAX).with_style(ProgressStyle::with_template("[{elapsed_precise}] {pos:>7} {msg}").unwrap())
        };
        loop {
            let lock = state.lock().unwrap();
            bar.set_position(lock.completed as u64);
            if let Some(files) = num_files {
                if files == lock.completed {
                    break;
                }
            }
            drop(lock);
            thread::sleep(Duration::from_millis(100));
        }
        bar.finish();
    }
    for thread in threads {
        thread.join().unwrap();
    }
}
pub fn generate_data<P: AsRef<Path>, E: PositionEvaluate>(path: P, eval: &E, mcts_config: MctsSearchConfig, datapoint_count: usize, max_datapoints_per_game: usize, num_rand_moves: usize, show_progress: bool) -> Result<(), std::io::Error> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let mut mov_vec = Vec::new();
    let mut states_vec = Vec::new();
    let mut num_written = 0;
    let bar = if show_progress {
        ProgressBar::new(datapoint_count as u64).with_style(ProgressStyle::with_template("[{elapsed_precise}] {bar:100.cyan/blue} {pos:>7}/{len:7} {msg}").unwrap())
    } else {
        ProgressBar::hidden()
        //ProgressBar::new(u64::MAX).with_style(ProgressStyle::with_template("[{elapsed_precise}] {pos:>7} {msg}").unwrap())
    };
    bar.set_position(0);
    while num_written < datapoint_count {
        states_vec.clear();
        let mut state = TerraceGameState::setup_new();
        for _ in 0..num_rand_moves {
            if state.result() != GameResult::Ongoing {
                break;
            }
            mov_vec.clear();
            state.generate_moves(&mut mov_vec);
            if mov_vec.len() == 0 {
                state.skip_turn();
                state.generate_moves(&mut mov_vec);
            }
            let mov = mov_vec[rand::thread_rng().gen_range(0..mov_vec.len())];
            state.make_move(mov);
        }
        while state.result() == GameResult::Ongoing {
            // We keep generating moves so we can handle a position with no legal moves
            mov_vec.clear();
            state.generate_moves(&mut mov_vec);
            let mov = if mov_vec.len() == 0 {
                Move::SKIP
            } else if mov_vec.len() == 1 {
                mov_vec[0]
            } else {
                let mut mcts = MctsSearch::new(state, mcts_config, eval as *const _);
                let mov = mcts.search().mov;
                mov
            };
            states_vec.push((state, mov));
            state.make_move(mov);
        }
        states_vec.shuffle(&mut rand::thread_rng());
        let result = state.result().into_absolute();
        for i in 0..states_vec.len().min(max_datapoints_per_game) {
            if num_written >= datapoint_count {
                break;
            }
            writer.write_all(&PositionInfo::new(states_vec[i].0, result, states_vec[i].1).bytes)?;
            num_written += 1;
        }
        bar.set_position(num_written as u64);
    }
    bar.finish();
    writer.flush()?;
    Ok(())
}
fn data_writer_thread<E: PositionEvaluate>(mut folder: PathBuf, state: &Mutex<MultithreadedDataCollectionState>, eval: E, mcts_config: MctsSearchConfig, datapoints_per_file: usize, max_datapoints_per_game: usize, num_rand_moves: usize) {
    loop {
        let mut lock = state.lock().unwrap();
        if Some(lock.current) == lock.required || lock.stop {
            break;
        }
        let i = lock.current;
        lock.current += 1;
        drop(lock);
        folder.push(format!("{i}.bin"));
        generate_data(&folder, &eval, mcts_config, datapoints_per_file, max_datapoints_per_game, num_rand_moves, false).unwrap();
        folder.pop();
        let mut lock = state.lock().unwrap();
        lock.completed += 1;
        drop(lock);
    }
}
pub fn generate_data_multithreaded<P: AsRef<Path>, E: PositionEvaluate + 'static>(dir: P,thread_count: usize, num_files: Option<usize>, show_progress: bool, eval: E, mcts_config: MctsSearchConfig, datapoint_count: usize, max_datapoints_per_game: usize, num_rand_moves: usize) {
    let state = Mutex::new(MultithreadedDataCollectionState {
        required: num_files,
        current: 0,
        completed: 0,
        stop: false
    });
    let _ = std::fs::create_dir_all(dir.as_ref());
    let mut threads = Vec::new();
    let state_ref = &state as *const _;
    let eval_ref = &eval as *const _;
    for i in 0..thread_count {
        let folder = dir.as_ref().to_owned();
        let state_ref = unsafe {&*state_ref};
        let eval = eval.clone();
        threads.push(thread::spawn(move || {
            data_writer_thread(folder, state_ref, eval, mcts_config, datapoint_count, max_datapoints_per_game, num_rand_moves);
        }));
    }
    if show_progress {
        let bar = if let Some(files) = num_files {
            ProgressBar::new(files as u64).with_style(ProgressStyle::with_template("[{elapsed_precise}] {bar:100.cyan/blue} {pos:>7}/{len:7} {msg}").unwrap())
        } else {
            ProgressBar::new(u64::MAX).with_style(ProgressStyle::with_template("[{elapsed_precise}] {pos:>7} {msg}").unwrap())
        };
        loop {
            let lock = state.lock().unwrap();
            bar.set_position(lock.completed as u64);
            if let Some(files) = num_files {
                if files == lock.completed {
                    break;
                }
            }
            drop(lock);
            thread::sleep(Duration::from_millis(100));
        }
        bar.finish();
    }
    for thread in threads {
        thread.join().unwrap();
    }
}

pub struct DataLoader<B: Backend> {
    next_sender: Sender<bool>,
    data_receiver: Receiver<Option<(NetworkInput<B>, NetworkTarget<B>)>>
}
impl<B: Backend> DataLoader<B> {
    const MAX_NUM_DATAPOINTS: usize = 100_000;
    fn loader_handler(dev: &B::Device, dir: PathBuf, sender: Sender<Option<(NetworkInput<B>, NetworkTarget<B>)>>, receiver: Receiver<bool>, epoch_count: usize) {
        for _ in 0..epoch_count {
            let mut files = fs::read_dir(&dir).unwrap();
            let mut data = Vec::new();
            loop {
                if let Some(Ok(entry)) = files.next() {
                    let file = File::open(entry.path()).unwrap();
                    let num_datapoints = file.metadata().unwrap().len() as usize / PositionInfo::SIZE;
                    let mut reader = BufReader::new(file);
                    let num_sets = num_datapoints / Self::MAX_NUM_DATAPOINTS;
                    for set_index in 0..num_sets + 1 { // The +1 is so that the final set isn't discared, particularly in datasets smaller than the MAX_NUM_DATAPOINTS this is important
                        data.clear();
                        let data_len = if set_index == num_sets {
                            num_datapoints - num_sets * Self::MAX_NUM_DATAPOINTS // The remainder of the sets
                        } else {
                            Self::MAX_NUM_DATAPOINTS
                        };
                        for _ in 0..data_len {
                            let mut datapoint = PositionInfo { bytes: [0; PositionInfo::SIZE] };
                            reader.read_exact(&mut datapoint.bytes).unwrap();
                            data.push(datapoint);
                        }
                        let data_to_send = data.as_slice().to_training_values(dev);
                        if sender.send(Some(data_to_send)).is_err() { return }
                        if let Ok(v) = receiver.recv() {
                            if !v {
                                return;
                            }
                        } else {
                            return;
                        }
                    }
                } else {
                    break;
                }
            }
        }
        if sender.send(None).is_err() {return};
        return;
    }
    pub fn new<P: AsRef<Path>>(dev: &B::Device, dir: P, epoch_count: usize) -> (Self, usize) {
        let dir = dir.as_ref().to_owned();
        let file_count = fs::read_dir(&dir).unwrap().count();
        let datapoints_per_file = fs::read_dir(&dir).unwrap().next().unwrap().unwrap().metadata().unwrap().len() as usize / PositionInfo::SIZE;
        let sets_per_file = datapoints_per_file / Self::MAX_NUM_DATAPOINTS + if datapoints_per_file % Self::MAX_NUM_DATAPOINTS != 0 {1} else {0};
        let (next_sender, next_receiver) = channel();
        let (data_sender, data_receiver) = channel();
        let other_dev = dev.clone();
        thread::spawn(move || {
            Self::loader_handler(&other_dev, dir, data_sender, next_receiver, epoch_count);
        });
        (Self {
            next_sender,
            data_receiver
        }, file_count * sets_per_file * epoch_count)
    }
    pub fn get_next(&self) -> Option<(NetworkInput<B>, NetworkTarget<B>)> {
        let value = self.data_receiver.recv();
        if let Ok(Some(v)) = value {
            let _ = self.next_sender.send(true);
            Some(v)
        } else {
            None
        }
    }
}
impl<B: Backend> Drop for DataLoader<B> {
    fn drop(&mut self) {
        let _ = self.next_sender.send(false);
    }
}
impl<B: Backend> Iterator for DataLoader<B> {
    type Item = (NetworkInput<B>, NetworkTarget<B>);
    fn next(&mut self) -> Option<Self::Item> {
        self.get_next()
    }
}

pub trait TrainableData {
    fn to_training_values<B: Backend>(&self, dev: &B::Device) -> (NetworkInput<B>, NetworkTarget<B>);
}
impl TrainableData for &[crate::ai::data::PositionInfo] {
    fn to_training_values<B: Backend>(&self, dev: &B::Device) -> (NetworkInput<B>, NetworkTarget<B>) {
        let mut input_state_vec = Vec::new();
        let mut input_legal_moves_vec = Vec::new();
        let mut target_value_vec = Vec::new();
        let mut target_policy_vec = Vec::new();
        let mut len = self.len();
        for &item in self.iter() {
            let (state, result, mov) = item.to_state();
            if let Some(m_index) = rules::ALL_POSSIBLE_MOVES.iter().position(|m| *m == mov) { // We could change this later, but for now this seems to work
                for x in 0..8 {
                    for y in 0..8 {
                        let pc = state.square(Square::from_xy((x as u8, y as u8)));
                        input_state_vec.push(!pc.is_any());
                        input_state_vec.push(pc == Piece::new(crate::rules::PieceType::T, Player::P1));
                        input_state_vec.push(pc == Piece::new(crate::rules::PieceType::T, Player::P2));
                        for size in 0..4 {
                            input_state_vec.push(pc == Piece::new(crate::rules::PieceType::from_size(size), Player::P1));
                            input_state_vec.push(pc == Piece::new(crate::rules::PieceType::from_size(size), Player::P2));
                        }
                        for i in 0..8 {
                            input_state_vec.push(i == rules::TOPOGRAPHICAL_BOARD_MAP[x][y]);
                        }
                    }
                }
                for mov in &*rules::ALL_POSSIBLE_MOVES {
                    input_legal_moves_vec.push(!state.is_move_valid(*mov));
                }
                target_value_vec.push(match result {
                    AbsoluteGameResult::P1Win => 0,
                    AbsoluteGameResult::Draw => 1,
                    AbsoluteGameResult::P2Win => 2
                });
                target_policy_vec.push(m_index as i32);
            } else {
                if mov == rules::Move::SKIP {
                    len -= 1;
                } else {
                    panic!("Illegal move in data: {:?}->{:?}", mov.from.xy(), mov.to.xy());
                }
            }
        }
        let input_state_data = Data::new(input_state_vec, Shape::new([len, INPUT_CHANNELS, 8, 8]));
        let input_legal_move_data = Data::new(input_legal_moves_vec, Shape::new([len, POLICY_OUTPUT_SIZE]));
        let target_value_data = Data::new(target_value_vec, Shape::new([len]));
        let target_policy_data = Data::new(target_policy_vec, Shape::new([len]));
        let input = NetworkInput {
            state: Tensor::<B, 4, Bool>::from_data(input_state_data, dev),
            illegal_moves: Tensor::<B, 2, Bool>::from_data(input_legal_move_data, dev)
        };
        let target = NetworkTarget {
            value: Tensor::<B, 1, Int>::from_data(target_value_data.convert(), dev),
            policy: Tensor::<B, 1, Int>::from_data(target_policy_data.convert(), dev)
        };
        (input, target)
    }
}