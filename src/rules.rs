use lazy_static::lazy_static;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Player {
    P1,
    P2
}
impl Player {
    pub const fn other(self) -> Self {
        match self {
            Self::P1 => Self::P2,
            Self::P2 => Self::P1
        }
    }
}
impl Into<bool> for Player {
    fn into(self) -> bool {
        self == Self::P2
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieceType {
    T,
    S1,
    S2,
    S3,
    S4
}
impl PieceType {
    pub const fn size(self) -> u8 {
        match self {
            Self::T => 0,
            Self::S1 => 0,
            Self::S2 => 1,
            Self::S3 => 2,
            Self::S4 => 3,
        }
    }
    pub const fn from_size(size: u8) -> Self {
        assert!(size < 4);
        match size {
            0 => Self::S1,
            1 => Self::S2,
            2 => Self::S3,
            3 => Self::S4,
            _ => unreachable!()
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Piece {
    is_any: bool,
    typ: PieceType,
    player: Player
}
impl Piece {
    pub const NONE: Self = Self {
        is_any: false,
        typ: PieceType::T,
        player: Player::P1
    };
    pub const fn new(typ: PieceType, player: Player) -> Self {
        Self {
            is_any: true,
            typ,
            player
        }
    }
    pub const fn is_any(self) -> bool {
        self.is_any
    }
    pub const fn player(self) -> Player {
        self.player
    }
    pub const fn is_player(self, player: Player) -> bool {
        self.is_any && (self.player as u8) == (player as u8) // This is because eq isn't a const trait, and you can only compare primitives like integer types
    }
    pub const fn size(self) -> u8 {
        assert!(self.is_any);
        self.typ.size()
    }
    pub const fn typ(self) -> PieceType {
        assert!(self.is_any);
        self.typ
    }
    pub const fn is_t(self) -> bool {
        self.is_any && self.typ as u8 == PieceType::T as u8
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Square {
    inner: u8
}
impl From<(u8, u8)> for Square {
    fn from(value: (u8, u8)) -> Self {
        Self::from_xy(value)
    }
}
impl Square {
    pub const fn from_xy(value: (u8, u8)) -> Self {
        assert!(value.0 < 8 && value.1 < 8);
        Self {
            inner: value.0 + (value.1 << 3)
        }
    }
    pub const fn from_index(index: u8) -> Self {
        Self {
            inner: index
        }
    }
    pub const fn height(self) -> u8 {
        let (x, y) = self.xy();
        TOPOGRAPHICAL_BOARD_MAP[x as usize][y as usize]
    }
    pub const fn xy(self) -> (u8, u8) {
        (self.inner & 7, self.inner >> 3)
    }
    pub const fn x(self) -> u8 {
        self.inner & 7
    }
    pub const fn y(self) -> u8 {
        self.inner >> 4
    }
    pub const fn add(self, x_inc: i8, y_inc: i8) -> Self {
        let (x,y) = self.xy();
        let (x, y) = (x as i8 + x_inc, y as i8 + y_inc);
        assert!(x >= 0 && y >= 0 && x < 8 && y < 8);
        Self::from_xy((x as u8, y as u8))
    }
    pub const fn is_diagonal(self, other: Self) -> bool {
        let (x1, y1) = self.xy();
        let (x2, y2) = other.xy();
        let x = x1 as i8 - x2 as i8;
        let y = y1 as i8 - y2 as i8;
        x.abs() == 1 && y.abs() == 1
    }
    pub const fn is_adjacent(self, other: Self) -> bool {
        let (x1, y1) = self.xy();
        let (x2, y2) = other.xy();
        let x = x1 as i8 - x2 as i8;
        let y = y1 as i8 - y2 as i8;
        x.abs() + y.abs() == 1
    }
    pub const fn is_higher(self, other: Self) -> bool {
        self.height() > other.height()
    }
    pub const fn is_lower(self, other: Self) -> bool {
        self.height() < other.height()
    }
    pub const fn is_diagonal_and_higher(self, other: Self) -> bool {
        self.is_diagonal(other) && self.is_higher(other)
    }
    pub const fn is_diagonal_and_lower(self, other: Self) -> bool {
        self.is_diagonal(other) && self.is_lower(other)
    }
    pub const fn same_quadrant(self, other: Self) -> bool {
        let (x1, y1) = self.xy();
        let (x2, y2) = other.xy();
        return ((x1 < 4) == (x2 < 4)) && ((y1 < 4) == (y2 < 4))
    }
    pub const fn index(self) -> u8 {
        assert!(self.inner < 64);
        self.inner
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Move {
    pub from: Square,
    pub to: Square
}
impl Move {
    pub const NONE: Self = Move {
        from: Square {inner: 255},
        to: Square {inner: 255}
    };
    pub const SKIP: Self = Move {
        from: Square {inner: 0},
        to: Square {inner: 0}
    };
    pub const fn new(from: Square, to: Square) -> Self {
        Self {
            from,
            to
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameResult {
    Ongoing,
    P1WinCapture,
    P1WinInvasion,
    P1WinResignation,
    P2WinCapture,
    P2WinInvasion,
    P2WinResignation,
    DrawAgreement,
    Draw50Moves
}
impl GameResult {
    pub fn into_absolute(self) -> AbsoluteGameResult {
        match self {
            Self::P1WinCapture => AbsoluteGameResult::P1Win,
            Self::P1WinInvasion => AbsoluteGameResult::P1Win,
            Self::P1WinResignation => AbsoluteGameResult::P1Win,

            Self::P2WinCapture => AbsoluteGameResult::P2Win,
            Self::P2WinInvasion => AbsoluteGameResult::P2Win,
            Self::P2WinResignation => AbsoluteGameResult::P2Win,

            Self::Draw50Moves => AbsoluteGameResult::Draw,
            Self::DrawAgreement => AbsoluteGameResult::Draw,

            Self::Ongoing => panic!("GameResult::Ongoing cannot be converted to AbsoluteGameResult")
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbsoluteGameResult {
    P1Win,
    P2Win,
    Draw
}
impl AbsoluteGameResult {
    pub fn other(self) -> Self {
        match self {
            Self::P1Win => Self::P2Win,
            Self::P2Win => Self::P1Win,
            Self::Draw => Self::Draw
        }
    }
}

const fn generate_topographical_map() -> [[u8; 8]; 8] {
    let mut map = [[0; 8]; 8];
    let mut x = 0;
    let mut y = 0;
    while x < 8 {
        y = 0;
        while y < 8 {
            // 3 3 3 3 4 4 4 4
            // 2 2 2 3 4 5 5 5
            // 1 1 2 3 4 5 6 6
            // 0 1 2 3 4 5 6 7
            map[x][y] = if x < 4 && y < 4 {
                if x >= y {
                    x as u8
                } else {
                    y as u8
                }
            } else if x < 4 && y >= 4 {
                let temp_x = 7 - x;
                let temp_y = y;
                if temp_x < temp_y {
                    temp_x as u8
                } else {
                    temp_y as u8
                }
            } else if x >= 4 && y < 4 {
                let temp_x = x;
                let temp_y = 7 - y;
                if temp_x < temp_y {
                    temp_x as u8
                } else {
                    temp_y as u8
                }
            } else {
                if x >= y {
                    7 - y as u8
                } else {
                    7 - x as u8
                }
            };
            y += 1;
        }
        x += 1;
    }
    map
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UndoableMove {
    inner: Move,
    capture: Piece,
    prev_moves_since_capture: u16,
}
pub const TOPOGRAPHICAL_BOARD_MAP: [[u8; 8]; 8] = generate_topographical_map();
lazy_static! {
    pub static ref POSSIBLE_MOVE_MAP: [[(u8,[Move;12]);8];8] = get_possible_moves_map();
    pub static ref ALL_POSSIBLE_MOVES: Vec<Move> = get_all_possible_moves();
}
pub const NUM_POSSIBLE_MOVES: usize = 568;
pub fn get_possible_moves_map() -> [[(u8,[Move;12]);8];8] {
    let mut map = [[(0, [Move::NONE; 12]); 8]; 8];
    for x in 0..8 {
        for y in 0..8 {
            map[x][y] = get_possible_moves(Square::from_xy((x as u8, y as u8)));
        }
    }
    map
}
pub fn get_all_possible_moves() -> Vec<Move> {
    let mut vec = Vec::new();
    for x in 0..8 {
        for y in 0..8 {
            let entry = POSSIBLE_MOVE_MAP[x][y];
            for mov in &entry.1[0..entry.0 as usize] {
                vec.push(*mov);
            }
        }
    }
    vec
}
pub fn get_possible_moves(sq: Square) -> (u8, [Move;12]) {
    let (sq_x, sq_y) = sq.xy();
    let mut b = TerraceGameState::setup_new();
    b.board = [[Piece::NONE; 8]; 8];
    b.board[sq_x as usize][sq_y as usize] = Piece::new(PieceType::S1, Player::P1);

    let mut move_index = 0;
    let mut moves = [Move::NONE;12];
    for x in 0..8 {
        for y in 0..8 {
            let mov = Move::new(sq, Square::from_xy((x, y)));
            if sq == Square::from_xy((x, y)) {
                continue;
            }
            if b.is_move_valid(mov) {
                moves[move_index] = mov;
                move_index += 1;
            }
        }
    }
    if sq_x > 0 {
        if sq_y > 0 {
            b.board[sq_x as usize - 1][sq_y as usize - 1] = Piece::new(PieceType::S1, Player::P2);
            let mov = Move::new(sq, Square::from_xy((sq_x - 1, sq_y - 1)));
            if b.is_move_valid(mov) {
                moves[move_index] = mov;
                move_index += 1;
            }
        }
        if sq_y < 7 {
            b.board[sq_x as usize - 1][sq_y as usize + 1] = Piece::new(PieceType::S1, Player::P2);
            let mov = Move::new(sq, Square::from_xy((sq_x - 1, sq_y + 1)));
            if b.is_move_valid(mov) {
                moves[move_index] = mov;
                move_index += 1;
            }
        }
    }
    if sq_x < 7 {
        if sq_y > 0 {
            b.board[sq_x as usize + 1][sq_y as usize - 1] = Piece::new(PieceType::S1, Player::P2);
            let mov = Move::new(sq, Square::from_xy((sq_x + 1, sq_y - 1)));
            if b.is_move_valid(mov) {
                moves[move_index] = mov;
                move_index += 1;
            }
        }
        if sq_y < 7 {
            b.board[sq_x as usize + 1][sq_y as usize + 1] = Piece::new(PieceType::S1, Player::P2);
            let mov = Move::new(sq, Square::from_xy((sq_x + 1, sq_y + 1)));
            if b.is_move_valid(mov) {
                moves[move_index] = mov;
                move_index += 1;
            }
        }
    }
    (move_index as u8, moves)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TerraceGameState {
    player_to_move: Player,
    board: [[Piece; 8]; 8],
    moves_since_capture: u16,
    move_number: u16,
    result: GameResult,
}
impl TerraceGameState {
    pub const NO_CAPTURE_MOVES_TO_DRAW: u16 = 40;
    pub const fn setup_new() -> Self {
        let mut board = [[Piece::NONE;8];8];
        let mut x = 0;
        while x < 8 {
            board[x][0] = Piece::new(PieceType::from_size(x as u8 / 2), Player::P1);
            board[x][1] = Piece::new(PieceType::from_size((7 - x) as u8 / 2), Player::P1);
            board[x][6] = Piece::new(PieceType::from_size(x as u8 / 2), Player::P2);
            board[x][7] = Piece::new(PieceType::from_size((7 - x) as u8 / 2), Player::P2);

            x += 1
        }
        board[0][0] = Piece::new(PieceType::T, Player::P1);
        board[7][7] = Piece::new(PieceType::T, Player::P2);
        Self {
            player_to_move: Player::P1,
            board,
            moves_since_capture: 0,
            move_number: 0,
            result: GameResult::Ongoing,
        }
    }
    pub const fn player_to_move(&self) -> Player {
        self.player_to_move
    }
    pub const fn square(&self, sq: Square) -> Piece {
        let (x, y) = sq.xy();
        self.board[x as usize][y as usize]
    }
    pub fn square_mut(&mut self, sq: Square) -> &mut Piece {
        let (x, y) = sq.xy();
        &mut self.board[x as usize][y as usize]
    }
    pub fn skip_turn(&mut self) {
        self.player_to_move = self.player_to_move.other();
    }
    pub fn make_move(&mut self, mov: Move) -> UndoableMove {
        if mov == Move::SKIP {
            self.skip_turn();
            return UndoableMove {inner: Move::SKIP, capture: Piece::NONE, prev_moves_since_capture: 0};
        }
        assert!(self.result == GameResult::Ongoing);
        if !self.is_move_valid(mov) {
            println!("Move is invalid: {:?}, {:?}", mov.from.xy(), mov.to.xy());
            //crate::gui::run(*self).unwrap();
        }
        assert!(self.is_move_valid(mov));
        let prev_moves_since_capture = self.moves_since_capture;
        if self.square(mov.to).is_any {
            self.moves_since_capture = 0;
        } else {
            self.moves_since_capture += 1;
        }
        let capture = self.square(mov.to);
        *self.square_mut(mov.to) = self.square(mov.from);
        *self.square_mut(mov.from) = Piece::NONE;
        self.move_number += 1;
        self.player_to_move = self.player_to_move.other();
        if capture.is_t() && self.player_to_move.other() == Player::P1 {
            self.result = GameResult::P1WinCapture;
        }
        else if capture.is_t() && self.player_to_move.other() == Player::P2 {
            self.result = GameResult::P2WinCapture;
        }
        else if mov.to == Square::from_xy((7, 7)) && self.player_to_move.other() == Player::P1 {
            self.result = GameResult::P1WinInvasion;
        }
        if mov.to == Square::from_xy((0, 0)) && self.player_to_move.other() == Player::P2 {
            self.result = GameResult::P2WinInvasion;
        }
        else if self.moves_since_capture >= Self::NO_CAPTURE_MOVES_TO_DRAW {
            self.result = GameResult::Draw50Moves;
        }
        UndoableMove {
            inner: mov,
            capture,
            prev_moves_since_capture
        }
    }
    pub fn resignation(&mut self, player: Player) {
        assert!(self.result == GameResult::Ongoing);
        self.result = match player {
            Player::P1 => GameResult::P2WinResignation,
            Player::P2 => GameResult::P1WinResignation
        }
    }
    pub fn draw_agreement(&mut self) {
        assert!(self.result == GameResult::Ongoing);
        self.result = GameResult::DrawAgreement;
    }
    pub const fn result(&self) -> GameResult {
        self.result
    }
    pub fn is_style_win(&self) -> bool {
        (self.result == GameResult::P1WinInvasion && self.board[7][7] == Piece::new(PieceType::T, Player::P1))||
        (self.result == GameResult::P2WinInvasion && self.board[0][0] == Piece::new(PieceType::T, Player::P2))
    }
    pub fn is_move_valid(&self, mov: Move) -> bool {
        assert!(self.result == GameResult::Ongoing);
        assert!(mov != Move::NONE);
        if mov == Move::SKIP {
            assert!(!self.has_legal_moves());
            return true;
        }
        let piece = self.square(mov.from);
        if !piece.is_player(self.player_to_move) {
            return false;
        }
        let size = piece.size();
        let capture = self.square(mov.to);
        let height_from = mov.from.height();
        let height_to = mov.to.height();
        let (x1, y1) = mov.from.xy();
        let (x2, y2) = mov.to.xy();
        if height_from == height_to && mov.from != mov.to {
            if capture.is_any {
                return false;
            }
            if !mov.from.same_quadrant(mov.to) {return false}; // The only case for this is when crossing the center diagonally, which is against the rules(as far as I can tell)
            let min_x = x1.min(x2);
            let max_x = x1.max(x2);
            let min_y = y1.min(y2);
            let max_y = y1.max(y2);
            if y1 < 4 {
                for x in min_x + 1..max_x {
                    if self.square(Square::from_xy((x, max_y))).is_any {
                        return false;
                    }
                }
            } else {
                for x in min_x + 1..max_x {
                    if self.square(Square::from_xy((x, min_y))).is_any {
                        return false;
                    }
                }
            }
            if x1 < 4 {
                // This could be cleaned up but I don't have the brainpower right now
                if y1 < 4 && x1 != x2 && y1 != y2 && self.square(Square::from_xy((max_x, max_y))).is_any {
                    return false;
                }
                if y1 >= 4 && x1 != x2 && y1 != y2 && self.square(Square::from_xy((max_x, min_y))).is_any {
                    return false;
                }
                for y in min_y + 1..max_y {
                    if self.square(Square::from_xy((max_x, y))).is_any {
                        return false;
                    }
                }
            } else {
                if y1 < 4 && x1 != x2 && y1 != y2 && self.square(Square::from_xy((min_x, max_y))).is_any {
                    return false;
                }
                if y1 >= 4 && x1 != x2 && y1 != y2 && self.square(Square::from_xy((min_x, min_y))).is_any {
                    return false;
                }
                /*if (y1 < 4 && x1 != x2 && self.square(Square::from_xy((min_x, max_y))).is_any) ||
                (y1 >= 4 && x1 != x2 && self.square(Square::from_xy((min_x, min_y))).is_any) {
                    return false;
                }*/
                for y in min_y + 1..max_y {
                    if self.square(Square::from_xy((min_x, y))).is_any {
                        return false;
                    }
                }
            }
            return true;
        } else if mov.from.is_diagonal(mov.to) {
            if height_from > height_to { // Downhill
                return capture.is_any && capture.size() <= size
                    && (capture.typ != PieceType::T || capture.player != piece.player); // Prevent capturing your own T
            } else if height_from < height_to {
                return !capture.is_any;
            } else if mov.from.same_quadrant(mov.to) {
                return !capture.is_any;
            } else { // This should be impossible to reach
                unreachable!();
                //return false;
            }
        } else if mov.from.is_adjacent(mov.to) {
            return !capture.is_any; // When moving on the same terrace, directly up, or directly down terraces, it only matters if there is a piece there
        } else { // Not a valid type of move
            return false;
        }
    }
    /// This badly needs to be optimized but it works for now
    pub fn generate_moves(&self, vec: &mut Vec<Move>) {
        assert!(vec.len() == 0);
        assert!(self.result == GameResult::Ongoing);
        for from_x in 0..8 {
            for from_y in 0..8 {
                if self.square(Square::from_xy((from_x as u8, from_y as u8))).is_player(self.player_to_move) {
                    let entry = (*POSSIBLE_MOVE_MAP)[from_x][from_y];
                    for &mov in &entry.1[0..entry.0 as usize] {
                        if self.is_move_valid(mov) {
                            vec.push(mov);
                        }
                    }
                }
            }
        }
    }
    pub fn moves_since_capture(&self) -> u16 {
        self.moves_since_capture
    }
    pub fn move_number(&self) -> u16 {
        self.move_number
    }
    pub fn undo_move(&mut self, mov: UndoableMove) {
        if mov.inner == Move::SKIP {
            self.player_to_move = self.player_to_move.other();
            return;
        }
        self.result = GameResult::Ongoing;
        *self.square_mut(mov.inner.from) = self.square(mov.inner.to);
        *self.square_mut(mov.inner.to) = mov.capture;
        self.player_to_move = self.player_to_move.other();
        self.move_number -= 1;
        self.moves_since_capture = mov.prev_moves_since_capture;
    }
    pub fn has_legal_moves(&self) -> bool {
        for x in 0..8 {
            for y in 0..8 {
                if self.square(Square::from_xy((x as u8, y as u8))).is_player(self.player_to_move) {
                    let entry = POSSIBLE_MOVE_MAP[x][y];
                    for &mov in &entry.1[0..entry.0 as usize] {
                        if self.is_move_valid(mov) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
    pub fn is_transmutation(&self, other: &Self) -> bool {
        self.board == other.board && self.player_to_move == other.player_to_move
    }
}