# play.py -- Human vs AI harness using ai.py (book-enabled)
# Usage: python play.py --ai-side black --depth 2

import argparse
import chess
from ai import minimax_sse, evaluate_board, build_book_map, build_openings_npz
from copy import deepcopy

# Piece wrapper used by ai.py expectations
class Piece:
    def _init_(self, name, colour):
        self.name = name
        self.colour = colour
    def _repr_(self):
        return f"{self.colour}{self.name}"

class Game:
    def _init_(self, board=None):
        self.board_obj = board or chess.Board()
        self.board = [[None]*8 for _ in range(8)]
        self._sync_board()
        self.state = None
    def _sync_board(self):
        for r in range(8):
            for c in range(8):
                sq = chess.square(c, 7-r)
                p = self.board_obj.piece_at(sq)
                if p is None:
                    self.board[r][c] = None
                else:
                    sym = p.symbol()
                    colour = 'w' if sym.isupper() else 'b'
                    mapn = {'p':'P','n':'N','b':'B','r':'R','q':'Q','k':'K'}
                    self.board[r][c] = Piece(mapn[sym.lower()], colour)
    def light_copy(self):
        new = Game(self.board_obj.copy())
        new.state = self.state
        return new
    def get_moves(self, r, c):
        from_sq = chess.square(c, 7-r)
        mvlist = []
        for mv in self.board_obj.legal_moves:
            if mv.from_square == from_sq:
                to_sq = mv.to_square
                file = chess.square_file(to_sq); rank = chess.square_rank(to_sq)
                r2 = 7 - rank; c2 = file
                mvlist.append((r2, c2))
        return mvlist
    def make_move(self, src, dst, promotion=None):
        from_sq = chess.square(src[1], 7-src[0]); to_sq = chess.square(dst[1], 7-dst[0])
        if promotion:
            prom_map = {'Q': chess.QUEEN, 'R': chess.ROOK, 'B': chess.BISHOP, 'N': chess.KNIGHT}
            mv = chess.Move(from_sq, to_sq, promotion=prom_map.get(promotion))
        else:
            mv = chess.Move(from_sq, to_sq)
        if mv not in self.board_obj.legal_moves:
            # try SAN parse fallback
            try:
                mv = self.board_obj.parse_san(self.board_obj.san(mv))
            except Exception:
                pass
        self.board_obj.push(mv)
        self._sync_board()
        if self.board_obj.is_game_over():
            self.state = self.board_obj.result()
    def san(self, src_dst_promo):
        try:
            if isinstance(src_dst_promo, tuple) and len(src_dst_promo) >= 2:
                src, dst = src_dst_promo[0], src_dst_promo[1]
                from_sq = chess.square(src[1], 7-src[0]); to_sq = chess.square(dst[1], 7-dst[0])
                mv = chess.Move(from_sq, to_sq)
                return self.board_obj.san(mv)
        except Exception:
            return ''
        return ''

def main(ai_side='black', depth=2, use_book=True):
    game = Game()
    # build / load book
    book_map = None
    if use_book:
        npz = 'openings_dataset.npz'
        if not Path(npz).exists():
            print("openings_dataset.npz missing — building from data/chess-openings sample (if present).")
            try:
                build_openings_npz('data/chess-openings', npz)
            except Exception as e:
                print("Could not build openings npz:", e)
        if Path(npz).exists():
            book_map = build_book_map(npz)
            print("Book loaded with prefixes:", len(book_map))
    print("Starting game. You are", "White" if ai_side=='black' else "Black")
    print(game.board_obj)
    while game.state is None:
        turn = 'white' if game.board_obj.turn == chess.WHITE else 'black'
        if turn == ai_side:
            print(f"AI ({turn}) thinking (depth={depth})...")
            best = minimax_sse(game, depth, float('-inf'), float('inf'), maximizing=(turn=='white'),
                               book_map=book_map, book_bias=1.8, opening_ply_limit=10)
            if not best:
                print("AI failed to find move — resigning")
                break
            src, mv = best
            if isinstance(mv, tuple) and len(mv) == 3:
                dst = mv[1]; promo = mv[2]
            else:
                dst = mv; promo = None
            san = game.san((src, dst, promo))
            print("AI plays:", san)
            game.make_move(src, dst, promotion=promo)
            print(game.board_obj)
        else:
            move_in = input("Your move (UCI e2e4 or SAN): ").strip()
            if move_in.lower() in ('quit','exit','resign'):
                print("Bye.")
                return
            try:
                move = None
                if len(move_in) >= 4 and move_in[0] in 'abcdefgh':
                    # try UCI
                    u = move_in
                    from_file = ord(u[0]) - ord('a'); from_rank = 8 - int(u[1])
                    to_file = ord(u[2]) - ord('a'); to_rank = 8 - int(u[3])
                    src = (from_rank, from_file); dst = (to_rank, to_file)
                    promo = u[4].upper() if len(u) > 4 else None
                    game.make_move(src, dst, promotion=promo)
                else:
                    # SAN
                    mv = game.board_obj.parse_san(move_in)
                    from_sq = chess.square_file(mv.from_square); from_rank = 7 - chess.square_rank(mv.from_square)
                    to_sq = chess.square_file(mv.to_square); to_rank = 7 - chess.square_rank(mv.to_square)
                    game.make_move((from_rank, from_sq), (to_rank, to_sq), promotion=None)
                print(game.board_obj)
            except Exception as e:
                print("Invalid move:", e)
                continue
    print("Game over:", game.state)
    print("Final board:\n", game.board_obj)

if __name__ == "_main_":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai-side", choices=["white","black"], default="black")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--no-book", action="store_true", help="Disable book usage")
    args = parser.parse_args()
    main(ai_side=args.ai_side, depth=args.depth, use_book=not args.no_book)
