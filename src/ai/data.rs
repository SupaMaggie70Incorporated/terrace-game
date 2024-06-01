use crate::rules::{AbsoluteGameResult, GameResult, Piece, PieceType, Player, Square, TerraceGameState};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PositionInfo {
    bytes: [u8; 33],
}
impl PositionInfo {
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
}