use std::io::Read;

use rand::Rng;
use slint::{Color, Image, Model, ModelExt, ModelRc, SharedPixelBuffer, VecModel};
use image::EncodableLayout;

use crate::rules::{self, GameResult, Move, Square, TerraceGameState};

slint::include_modules!();



fn load_image(bytes: &[u8]) -> Image {
    let image = image::load_from_memory(bytes).unwrap().into_rgba8();
    let buffer =
        SharedPixelBuffer::clone_from_slice(image.as_bytes(), image.width(), image.height());
    slint::Image::from_rgba8(buffer)
}
fn create_board() -> ModelRc<ModelRc<SquareInfo>> {
    let mut outer_vec = Vec::new();
    for x in 0..8 {
        let mut inner_vec = Vec::new();
        for y in 0..8 {
            inner_vec.push(SquareInfo {
                is_t: false,
                size: 0,
                p2: false,
                selected: false,
                highlighted: false,
            })
        }
        let model = ModelRc::new(VecModel::from(inner_vec));
        outer_vec.push(model);
    }
    ModelRc::new(VecModel::from(outer_vec))
}
fn copy_board(b: &rules::TerraceGameState, model: ModelRc::<ModelRc<SquareInfo>>) {
    for x in 0..8 {
        for y in 0..8 {
            let p = b.square(Square::from_xy((x, y)));
            let mut data = if p.is_any() {
                SquareInfo {
                    is_t: p.is_t(),
                    size: p.size() as i32 + 1,
                    p2: p.is_player(rules::Player::P2),
                    selected: false,
                    highlighted: false,
                }
            } else {
                SquareInfo {
                    is_t: false,
                    size: 0,
                    p2: false,
                    selected: false,
                    highlighted: false,
                }
            };
            model.row_data_tracked(x as usize).unwrap().set_row_data(y as usize, data);
        }
    }
}
fn get_square(model: ModelRc::<ModelRc<SquareInfo>>, x: u8, y: u8) -> SquareInfo {
    model.row_data(x as usize).unwrap().row_data(y as usize).unwrap()
}
fn set_square(model: ModelRc::<ModelRc<SquareInfo>>, x: u8, y: u8, info: SquareInfo) {
    model.row_data_tracked(x as usize).unwrap().set_row_data(y as usize, info);

}

struct AppState {
    game_state: TerraceGameState,
    state_history: Vec<TerraceGameState>,
    state_index: usize,
    selected_square: Option<Square>,
    ui: AppWindow,
}

pub fn run(game_state: TerraceGameState) -> Result<(), slint::PlatformError> {
    let ui = AppWindow::new()?;

    let mut app_state = AppState {
        game_state: game_state,
        state_history: vec![game_state],
        state_index: 0,
        selected_square: None,
        ui
    };
    let app_state_ptr = &mut app_state as *mut AppState;

    let style = app_state.ui.global::<Style>();
    let logic = app_state.ui.global::<Logic>();
    let ui_game_state = app_state.ui.global::<GameState>();
    ui_game_state.set_board(create_board());
    copy_board(&app_state.game_state, ui_game_state.get_board());
    logic.on_board_height(move |x, y| {
        rules::TOPOGRAPHICAL_BOARD_MAP[x as usize][y as usize] as i32
    });
    style.on_height_to_color(move |height| {
        assert!(height < 8);
        let value = match height {
            0 => 0x20,
            1 => 0x28,
            2 => 0x30,
            3 => 0x38,
            4 => 0x40,
            5 => 0x48,
            6 => 0x50,
            7 => 0x58,
            _ => unreachable!()
        };
        slint::Brush::SolidColor(Color::from_rgb_u8(value, 0x10, 0x10))
    });
    style.set_p1_piece_color(slint::Brush::SolidColor(Color::from_rgb_u8(0, 0x80, 0)));
    style.set_p2_piece_color(slint::Brush::SolidColor(Color::from_rgb_u8(0, 0, 0x80)));
    style.set_highlight_color(slint::Brush::SolidColor(Color::from_argb_u8(0xff, 0x90, 0x90, 0x90)));
    style.set_selection_color(slint::Brush::SolidColor(Color::from_argb_u8(0x80, 0x00, 0x80, 0x80)));

    logic.on_left_click(move |x, y| {
        let app_state = unsafe {&mut *app_state_ptr};
        if app_state.game_state.result() != GameResult::Ongoing {
            println!("Click disregarded as game over: {:?}", app_state.game_state.result());
        } else if app_state.state_index != app_state.state_history.len() - 1 {
            app_state.selected_square = None;
            app_state.state_index += 1;
            copy_board(&app_state.state_history[app_state.state_index], app_state.ui.global::<GameState>().get_board());
        } else if let Some(from) = app_state.selected_square {
            let to = Square::from((x as u8, y as u8));
            let mov = Move::new(from, to);
            if app_state.game_state.is_move_valid(mov) {
                app_state.game_state.make_move(mov);
                app_state.state_history.push(app_state.game_state);
                app_state.state_index += 1;
            } else {
                println!("Illegal move");
            }
            copy_board(&app_state.game_state, app_state.ui.global::<GameState>().get_board());
            app_state.selected_square = None;
        } else {
            let sq = Square::from((x as u8, y as u8));
            if !app_state.game_state.square(sq).is_player(app_state.game_state.player_to_move()) {
                return;
            }
            app_state.selected_square = Some(sq);
            let state = app_state.ui.global::<GameState>().get_board();
            let mut info = get_square(state.clone(), x as u8, y as u8);
            info.selected = true;
            set_square(state.clone(), x as u8, y as u8, info);
            for x in 0..8 {
                for y in 0..8 {
                    let to = Square::from_xy((x, y));
                    if app_state.game_state.is_move_valid(Move::new(sq, to)) {
                        let mut info = get_square(state.clone(), x as u8, y as u8);
                        info.highlighted = true;
                        set_square(state.clone(), x as u8, y as u8, info);
                    }
                }
            }
        }
    });
    logic.on_right_click(move |x, y| {
        let app_state = unsafe {&mut *app_state_ptr};
        if app_state.state_index != 0 {
            app_state.state_index -= 1;
            copy_board(&app_state.state_history[app_state.state_index], app_state.ui.global::<GameState>().get_board());
        }
    });

    app_state.ui.run()
}