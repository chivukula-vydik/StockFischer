# 1_convert_data.py
import numpy as np
import chess
import chess.pgn
import sys
from pathlib import Path
PGN_FILE_PATH = "lichess_15000_games.pgn" 
OUTPUT_NPZ_FILE = "training_data.npz"
MAX_GAMES = 15000 

def board_to_tensor(board):
    planes = np.zeros((12, 8, 8), dtype=np.uint8)
    piece_to_index = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            r, c = chess.square_rank(sq), chess.square_file(sq)
            piece_name = piece.symbol().upper()
            idx = piece_to_index[piece_name] + (6 if piece.color == chess.BLACK else 0)
            planes[idx, r, c] = 1
    return planes

def result_to_value(result):
    if result == "1-0": return 1.0
    if result == "0-1": return -1.0
    return 0.0

def generate_dataset():
    if not Path(PGN_FILE_PATH).exists():
        print(f"Error: PGN file not found at {PGN_FILE_PATH}. Check the filename and path.")
        sys.exit(1)
    X_data, Y_data = [], []; games_processed = 0
    with open(PGN_FILE_PATH, encoding="utf-8") as pgn:
        while games_processed < MAX_GAMES:
            game = chess.pgn.read_game(pgn)
            if game is None: break
            result = game.headers.get("Result"); 
            if result is None or result == '*' or result == 'Time forfeit': continue
            y_result = result_to_value(result); board = game.board()
            for move in game.mainline_moves():
                X_data.append(board_to_tensor(board))
                Y_data.append(y_result) 
                board.push(move)
            games_processed += 1
            if games_processed % 1000 == 0: print(f"Processed {games_processed}/{MAX_GAMES} games...")

    np.savez_compressed(OUTPUT_NPZ_FILE, X=np.array(X_data, dtype=np.uint8), Y=np.array(Y_data, dtype=np.float32))
    print(f"\nSuccessfully created dataset with {len(X_data)} positions. Saved to {OUTPUT_NPZ_FILE}.")

if __name__ == "__main__":
    generate_dataset()
