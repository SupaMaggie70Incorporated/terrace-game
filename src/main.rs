use std::env::args;

use rand::Rng;
use rules::{AbsoluteGameResult, GameResult, TerraceGameState};

mod rules;
#[cfg(feature = "gui")]
mod gui;
#[cfg(feature = "ai")]
mod ai;

fn main() {
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
            "other" => {
                loop {
                    let mut game_state = TerraceGameState::setup_new();
                    let mut mov_vec = Vec::new();
                    for i in 0..1000 {
                        if game_state.result() != GameResult::Ongoing {
                            //println!("Game ended on move {i}: {:?}", game_state.result());
                            break;
                        }
                        mov_vec.clear();
                        game_state.generate_moves(&mut mov_vec);
                        let mov = mov_vec[rand::thread_rng().gen_range(0..mov_vec.len())];
                        game_state.make_move(mov);
                    }
                }
            }
            _ => println!("Unrecognized mode!")
        };
    }
}
