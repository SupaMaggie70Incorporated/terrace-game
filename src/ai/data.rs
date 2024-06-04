use std::{fs::File, io::{BufReader, BufWriter, Read, Write}, path::{Path, PathBuf}, sync::Mutex, thread, time::Duration};

use burn::tensor::backend::Backend;
use indicatif::{ProgressBar, ProgressStyle};
use rand::{seq::SliceRandom, Rng};

use crate::rules::{AbsoluteGameResult, GameResult, Piece, PieceType, Player, Square, TerraceGameState};

use super::net::{NetworkInput, NetworkTarget, TrainableData};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PositionInfo {
    pub bytes: [u8; Self::SIZE],
}
impl PositionInfo {
    pub const SIZE: usize = 33;
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
    pub fn new(state: TerraceGameState, result: AbsoluteGameResult) -> Self {
        let flip = state.player_to_move() == Player::P2;
        let result = match if flip {result.other()} else {result} {
            AbsoluteGameResult::P1Win => 0,
            AbsoluteGameResult::Draw => 1,
            AbsoluteGameResult::P2Win => 2,
        };
        let mut bytes = [0; 33];
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
        bytes[32] = result;
        Self {
            bytes
        }
    }
    pub fn to_state(&self) -> (TerraceGameState, AbsoluteGameResult) {
        let mut state = TerraceGameState::setup_new();
        let result = match self.bytes[32] {
            0 => AbsoluteGameResult::P1Win,
            1 => AbsoluteGameResult::Draw,
            2 => AbsoluteGameResult::P2Win,
            _ => panic!("Invalid state passed")
        };
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
        (state, result)
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
        match self.bytes[32] {
            0 => AbsoluteGameResult::P1Win,
            1 => AbsoluteGameResult::Draw,
            2 => AbsoluteGameResult::P2Win,
            _ => panic!("Invalid game result in training data")
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
pub fn generate_random_data<P: AsRef<Path>>(path: P, datapoint_count: usize, max_datapoints_per_game: usize, first_valid_move: usize) -> Result<(), std::io::Error> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let mut mov_vec = Vec::new();
    let mut states_vec = Vec::new();
    let mut num_written = 0;
    while num_written < datapoint_count {
        states_vec.clear();
        let mut state = TerraceGameState::setup_new();
        states_vec.push(state);
        while state.result() == GameResult::Ongoing {
            mov_vec.clear();
            state.generate_moves(&mut mov_vec);
            if mov_vec.len() == 0 {
                state.skip_turn();
                state.generate_moves(&mut mov_vec);
                #[cfg(feature = "gui")]
                if mov_vec.len() == 0 {
                    crate::gui::run(state).unwrap();
                }
            }
            let mov = mov_vec[rand::thread_rng().gen_range(0..mov_vec.len())];
            state.make_move(mov);
            if state.move_number() as usize - 1 >= first_valid_move {
                states_vec.push(state);
            }
        }
        states_vec.shuffle(&mut rand::thread_rng());
        let result = state.result().into_absolute();
        for i in 0..states_vec.len().min(max_datapoints_per_game) {
            if num_written >= datapoint_count {
                break;
            }
            writer.write_all(&PositionInfo::new(states_vec[i], result).bytes)?;
            num_written += 1;
        }
    }
    writer.flush()?;
    Ok(())
}
struct MultithreadedDataCollectionState {
    required: Option<usize>,
    current: usize,
    completed: usize,
    stop: bool,
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