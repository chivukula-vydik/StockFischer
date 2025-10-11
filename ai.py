# ai.py - StockFischer AI module (evaluation, minimax with SEE, openings/book, trainer)
# Overwrites previous ai.py. Designed to be drop-in and robust.
# See play.py for a simple harness that uses the book + minimax.

import os, json, csv, gzip, subprocess, shutil
from pathlib import Path
import numpy as np

# Optional PyTorch
TORCH_AVAILABLE = True
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    TORCH_AVAILABLE = False
    torch = nn = F = DataLoader = Dataset = DEVICE = None

# ---------------------------
# Piece values & PSTs
# ---------------------------
piece_values = {'P':100, 'N':320, 'B':330, 'R':500, 'Q':900, 'K':20000}

pawn_table = [
     0,0,0,0,0,0,0,0,
    50,50,50,50,50,50,50,50,
    10,10,20,30,30,20,10,10,
     5,5,10,25,25,10,5,5,
     0,0,0,20,20,0,0,0,
     5,-5,-10,0,0,-10,-5,5,
     5,10,10,-20,-20,10,10,5,
     0,0,0,0,0,0,0,0
]

knight_table = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,0,0,0,0,-20,-40,
   -30,0,10,15,15,10,0,-30,
   -30,5,15,20,20,15,5,-30,
   -30,0,15,20,20,15,0,-30,
   -30,5,10,15,15,10,5,-30,
   -40,-20,0,5,5,0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50
]

bishop_table = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,5,0,0,0,0,5,-10,
   -10,10,10,10,10,10,10,-10,
   -10,0,10,10,10,10,0,-10,
   -10,5,5,10,10,5,5,-10,
   -10,0,5,10,10,5,0,-10,
   -10,0,0,0,0,0,0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20
]

rook_table = [
     0,0,0,0,0,0,0,0,
     5,10,10,10,10,10,10,5,
    -5,0,0,0,0,0,0,-5,
    -5,0,0,0,0,0,0,-5,
    -5,0,0,0,0,0,0,-5,
    -5,0,0,0,0,0,0,-5,
    -5,0,0,0,0,0,0,-5,
     0,0,0,5,5,0,0,0
]

queen_table = [
   -20,-10,-10,-5,-5,-10,-10,-20,
   -10,0,0,0,0,0,0,-10,
   -10,0,5,5,5,5,0,-10,
    -5,0,5,5,5,5,0,-5,
     0,0,5,5,5,5,0,-5,
   -10,5,5,5,5,5,0,-10,
   -10,0,5,0,0,0,0,-10,
   -20,-10,-10,-5,-5,-10,-10,-20
]

king_table = [
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20,20,0,0,0,0,20,20,
    20,30,10,0,0,10,30,20
]

# ---------------------------
# SEE (local simplified)
# ---------------------------
def static_exchange_eval_local(game, target, colour, move_cache):
    if not game.board[target[0]][target[1]]:
        return 0
    copyg = game.light_copy()
    gains = [piece_values[copyg.board[target[0]][target[1]].name]]
    side = colour
    steps = 0
    MAX_STEPS = 32
    while steps < MAX_STEPS:
        steps += 1
        attackers = []
        for r in range(8):
            for c in range(8):
                piece = copyg.board[r][c]
                if not piece or piece.colour != side:
                    continue
                moves = move_cache.get((r,c), copyg.get_moves(r,c))
                if target in moves:
                    attackers.append((r,c))
        if not attackers:
            break
        best = min(attackers, key=lambda sq: piece_values[copyg.board[sq[0]][sq[1]].name])
        copyg.make_move(best, target)
        side = 'b' if side == 'w' else 'w'
        if copyg.board[target[0]][target[1]]:
            gains.append(piece_values[copyg.board[target[0]][target[1]].name] - gains[-1])
        else:
            gains.append(-gains[-1])
    for i in range(len(gains)-2, -1, -1):
        gains[i] = max(-gains[i+1], gains[i])
    return gains[0]

# ---------------------------
# Evaluation
# ---------------------------
def evaluate_board(game, move_cache=None):
    if move_cache is None:
        move_cache = {}
    score = 0
    white_pawns, black_pawns = [], []
    for row in range(8):
        for col in range(8):
            piece = game.board[row][col]
            if not piece:
                continue
            val = piece_values[piece.name]
            index = row*8 + col if piece.colour == 'w' else (7-row)*8 + (7-col)
            if piece.name == 'P': val += pawn_table[index]
            elif piece.name == 'N': val += knight_table[index]
            elif piece.name == 'B': val += bishop_table[index]
            elif piece.name == 'R': val += rook_table[index]
            elif piece.name == 'Q': val += queen_table[index]
            elif piece.name == 'K': val += king_table[index]
            if piece.name == 'P':
                (white_pawns if piece.colour=='w' else black_pawns).append((row,col))
            score += val if piece.colour == 'w' else -val
    def pawn_structure(pawns, enemy_pawns, colour):
        bonus = 0
        files = [0]*8
        for r,c in pawns: files[c]+=1
        for f in range(8):
            if files[f]>1: bonus -= 20*(files[f]-1)
            if f==0 and files[f+1]==0: bonus -= 15
            elif f==7 and files[f-1]==0: bonus -= 15
            elif 0<f<7 and files[f-1]==0 and files[f+1]==0: bonus -= 15
        for r,c in pawns:
            blocked = any(ec in [c-1,c,c+1] and ((colour=='w' and er<r) or (colour=='b' and er>r)) for er,ec in enemy_pawns)
            if not blocked:
                rank = r if colour=='b' else 7-r
                bonus += 10 + rank*5
        return bonus
    score += pawn_structure(white_pawns, black_pawns,'w')
    score -= pawn_structure(black_pawns, white_pawns,'b')
    # mobility
    white_moves = sum(len(move_cache.get((r,c), game.get_moves(r,c))) for r in range(8) for c in range(8)
                      if game.board[r][c] and game.board[r][c].colour=='w')
    black_moves = sum(len(move_cache.get((r,c), game.get_moves(r,c))) for r in range(8) for c in range(8)
                      if game.board[r][c] and game.board[r][c].colour=='b')
    score += 10*(white_moves - black_moves)
    return score

# ---------------------------
# Coordinate helpers
# ---------------------------
# ---------------------------
# Coordinate helpers (improved parser-safe version)
# ---------------------------

def coords_to_uci(src, dst, promotion=None):
    """Convert internal (row, col) tuples to UCI notation like 'e2e4'."""
    file_from = chr(ord('a') + src[1])
    rank_from = str(8 - src[0])
    file_to = chr(ord('a') + dst[1])
    rank_to = str(8 - dst[0])
    uci = f"{file_from}{rank_from}{file_to}{rank_to}"
    if promotion:
        uci += promotion.lower()
    return uci


def coords_to_desttoken(dst, promotion=None):
    """Convert a destination (row, col) to a short token like 'e4' for book matching."""
    file_to = chr(ord('a') + dst[1])
    rank_to = str(8 - dst[0])
    token = f"{file_to}{rank_to}"
    if promotion:
        token += promotion.lower()
    return token


def uci_to_coords(uci_str):
    """
    Convert a UCI string like 'e2e4' or 'e7e8q' into ((src_row, src_col), (dst_row, dst_col), promotion)
    Returns None if invalid.
    """
    if not isinstance(uci_str, str) or len(uci_str) < 4:
        return None
    try:
        from_file = ord(uci_str[0]) - ord('a')
        from_rank = 8 - int(uci_str[1])
        to_file = ord(uci_str[2]) - ord('a')
        to_rank = 8 - int(uci_str[3])
        src = (from_rank, from_file)
        dst = (to_rank, to_file)
        promo = uci_str[4].upper() if len(uci_str) > 4 else None
        return src, dst, promo
    except Exception:
        return None


def parse_move_input(move_str, game=None):
   
    import chess
    s = move_str.strip()
    # First, check if it's UCI-style like e2e4 or e7e8q
    if len(s) >= 4 and s[0] in 'abcdefgh' and s[2] in 'abcdefgh':
        parsed = uci_to_coords(s)
        if parsed:
            return parsed
    # Try SAN (like Nf3, exd5, etc.)
    if game and hasattr(game, 'board_obj'):
        try:
            mv = game.board_obj.parse_san(s)
            src = (7 - chess.square_rank(mv.from_square), chess.square_file(mv.from_square))
            dst = (7 - chess.square_rank(mv.to_square), chess.square_file(mv.to_square))
            promo = None
            if mv.promotion:
                promo = chess.piece_symbol(mv.promotion).upper()
            return src, dst, promo
        except Exception:
            pass
    return None

# ---------------------------
# Book helpers
# ---------------------------
def build_book_map(npz_path, max_prefix=8):
    data = np.load(npz_path, allow_pickle=True)
    moves_arr = data['moves']
    book = {}
    for s in moves_arr:
        toks = [t for t in str(s).strip().split() if not t.endswith('.')]
        for i in range(len(toks)):
            if i > max_prefix: break
            prefix = tuple(toks[:i])
            nxt = toks[i]
            book.setdefault(prefix, {})
            book[prefix][nxt] = book[prefix].get(nxt, 0) + 1
    return book

def _derive_simple_played_tokens(game):
    tokens = []
    try:
        board_obj = getattr(game, 'board_obj', None)
        if board_obj is not None:
            import chess
            temp = chess.Board()
            for mv in board_obj.move_stack:
                try:
                    san = temp.san(mv)
                    tokens.append(san if san else mv.uci())
                except Exception:
                    u = mv.uci()
                    tokens.append(u[2:])
                temp.push(mv)
            return tokens
    except Exception:
        pass
    try:
        history = getattr(game, 'history', None)
        if history:
            for mv in history:
                if isinstance(mv, tuple):
                    dst = mv[1]; tokens.append(coords_to_desttoken(dst, mv[2] if len(mv)>2 else None))
            return tokens
    except Exception:
        pass
    return []

def prefer_book_moves(game, legal_moves, book_map, bias=1.5, max_prefix_search=8):
    multipliers = {}
    played = _derive_simple_played_tokens(game)
    nexts = {}
    for L in range(min(len(played), max_prefix_search), -1, -1):
        pref = tuple(played[:L])
        if pref in book_map:
            nexts = book_map[pref]; break
    for mv in legal_moves:
        if isinstance(mv, tuple):
            if len(mv) == 2 and isinstance(mv[0], tuple) and isinstance(mv[1], tuple):
                src, dst = mv[0], mv[1]; promo = None
            elif len(mv) >= 3 and isinstance(mv[0], tuple):
                src, dst, promo = mv[0], mv[1], mv[2]
            else:
                src = None; dst = mv; promo = None
            dest_token = coords_to_desttoken(dst, promotion=promo)
            uci = coords_to_uci(src, dst, promotion=promo) if src is not None else dest_token
        else:
            try:
                uci = mv.uci()
            except Exception:
                uci = str(mv)
            dest_token = uci[2:] if len(uci)>=4 else uci
        mult = bias if (dest_token in nexts or uci in nexts or str(mv) in nexts) else 1.0
        multipliers[mv] = mult
    return multipliers

# ---------------------------
# Minimax + SSE (with optional book_map)
# ---------------------------
def minimax_sse(game, depth, alpha, beta, maximizing, original_depth=None, move_cache=None,
                book_map=None, book_bias=1.5, opening_ply_limit=10, played_token_getter=None):
    if original_depth is None: original_depth = depth
    if move_cache is None: move_cache = {}
    if depth == 0 or game.state is not None:
        return evaluate_board(game, move_cache)
    best_move = None
    def order_moves(src, moves_list):
        if book_map is None:
            return moves_list
        if played_token_getter:
            played = played_token_getter(game)
        else:
            played = _derive_simple_played_tokens(game)
        if len(played) >= opening_ply_limit:
            return moves_list
        nexts = {}
        for L in range(min(len(played), 8), -1, -1):
            pref = tuple(played[:L])
            if pref in book_map:
                nexts = book_map[pref]; break
        annotated = []
        for mv in moves_list:
            promo = mv[2] if isinstance(mv, tuple) and len(mv)>2 else None
            dst_token = coords_to_desttoken(mv if not isinstance(mv, tuple) or not isinstance(mv[0], tuple) else mv[1], promotion=promo)
            uci = coords_to_uci(mv[0], mv[1], promotion=promo) if isinstance(mv, tuple) and isinstance(mv[0], tuple) else str(mv)
            mult = book_bias if (dst_token in nexts or uci in nexts or str(mv) in nexts) else 1.0
            annotated.append((mv, mult))
        annotated.sort(key=lambda x: -x[1])
        return [a[0] for a in annotated]
    if maximizing:
        max_eval = float('-inf')
        for r in range(8):
            for c in range(8):
                piece = game.board[r][c]
                if not piece or piece.colour != 'w': continue
                moves = move_cache.get((r,c), game.get_moves(r,c))
                move_cache[(r,c)] = moves
                moves_ord = order_moves((r,c), moves)
                for mv in moves_ord:
                    if game.board[mv[0]][mv[1]] and static_exchange_eval_local(game, mv, 'w', move_cache) < 0:
                        continue
                    copyg = game.light_copy()
                    if isinstance(mv, tuple) and len(mv)>=3 and mv[2]:
                        copyg.make_move((r,c), mv[1], promotion=mv[2])
                    else:
                        copyg.make_move((r,c), mv)
                    val = minimax_sse(copyg, depth-1, alpha, beta, False, original_depth, move_cache.copy(),
                                      book_map, book_bias, opening_ply_limit, played_token_getter)
                    if val > max_eval:
                        max_eval = val; best_move = ((r,c), mv)
                    alpha = max(alpha, val)
                    if beta <= alpha: break
        return best_move if depth == original_depth else max_eval
    else:
        min_eval = float('inf')
        for r in range(8):
            for c in range(8):
                piece = game.board[r][c]
                if not piece or piece.colour != 'b': continue
                moves = move_cache.get((r,c), game.get_moves(r,c))
                move_cache[(r,c)] = moves
                moves_ord = order_moves((r,c), moves)
                for mv in moves_ord:
                    if game.board[mv[0]][mv[1]] and static_exchange_eval_local(game, mv, 'b', move_cache) < 0:
                        continue
                    copyg = game.light_copy()
                    if isinstance(mv, tuple) and len(mv)>=3 and mv[2]:
                        copyg.make_move((r,c), mv[1], promotion=mv[2])
                    else:
                        copyg.make_move((r,c), mv)
                    val = minimax_sse(copyg, depth-1, alpha, beta, True, original_depth, move_cache.copy(),
                                      book_map, book_bias, opening_ply_limit, played_token_getter)
                    if val < min_eval:
                        min_eval = val; best_move = ((r,c), mv)
                    beta = min(beta, val)
                    if beta <= alpha: break
        return best_move if depth == original_depth else min_eval

# ---------------------------
# Openings dataset builder (JSON/CSV/TSV/TSV.GZ/YAML)
# ---------------------------
def shutil_which(cmd):
    return shutil.which(cmd)

def download_openings_git(repo_url='https://github.com/lichess-org/chess-openings.git', dest='data/chess-openings'):
    destp = Path(dest)
    if destp.exists(): return str(destp)
    if not shutil_which('git'):
        raise RuntimeError("git not found in PATH.")
    subprocess.check_call(['git','clone','--depth','1',repo_url,dest])
    return str(destp)

def build_openings_npz(openings_dir='data/chess-openings', out_npz='openings_dataset.npz', max_openings=None):
    d = Path(openings_dir)
    if not d.exists():
        raise FileNotFoundError(f"{openings_dir} not found.")
    candidates = (list(d.rglob('.json')) + list(d.rglob('.csv')) +
                  list(d.rglob('.tsv')) + list(d.rglob('.tsv.gz')) +
                  list(d.rglob('.yaml')) + list(d.rglob('.yml')))
    if not candidates:
        raise RuntimeError("No JSON/CSV/TSV/YAML files found.")
    chosen = None
    for c in candidates:
        if c.name.lower() in ('openings.json','openings.csv','openings.tsv','openings.tsv.gz','openings.yaml','openings.yml'):
            chosen = c; break
    if chosen is None: chosen = candidates[0]
    moves = []; ecos = []; names = []
    print(f"Parsing openings from {chosen}")
    suffix = chosen.suffix.lower()
    if chosen.name.lower().endswith('.tsv.gz'): suffix = '.tsv.gz'
    if suffix == '.json':
        with open(chosen,'r',encoding='utf-8') as fh:
            data = json.load(fh)
        data_iter = data.values() if isinstance(data, dict) else data
        for entry in data_iter:
            if not isinstance(entry, dict): continue
            mv = entry.get('pgn') or entry.get('moves') or entry.get('uci') or entry.get('san') or entry.get('sequence')
            eco = entry.get('eco') or entry.get('code') or entry.get('ECO')
            name = entry.get('name') or entry.get('label') or entry.get('opening')
            if mv and eco:
                moves.append(mv); ecos.append(eco); names.append(name or '')
            if max_openings and len(moves) >= max_openings: break
    elif suffix == '.csv':
        with open(chosen,'r',encoding='utf-8') as fh:
            rdr = csv.DictReader(fh)
            for row in rdr:
                mv = row.get('pgn') or row.get('moves') or row.get('uci') or row.get('san')
                eco = row.get('eco') or row.get('ECO') or row.get('code')
                name = row.get('name') or row.get('label')
                if mv and eco:
                    moves.append(mv); ecos.append(eco); names.append(name or '')
                if max_openings and len(moves) >= max_openings: break
    elif suffix in ('.tsv', '.tsv.gz'):
        fh = gzip.open(chosen,'rt',encoding='utf-8',errors='replace') if suffix=='.tsv.gz' else open(chosen,'r',encoding='utf-8',errors='replace')
        with fh:
            rdr = csv.DictReader(fh, delimiter='\t')
            mv_keys = ['moves','pgn','uci','san','sequence','move']
            eco_keys = ['eco','code','ECO']
            name_keys = ['name','label','opening','title']
            for row in rdr:
                if not isinstance(row, dict): continue
                mv=None; eco=None; name=None
                for k in mv_keys:
                    if k in row and row[k]:
                        mv=row[k]; break
                for k in eco_keys:
                    if k in row and row[k]:
                        eco=row[k]; break
                for k in name_keys:
                    if k in row and row[k]:
                        name=row[k]; break
                if mv is None:
                    first_col = next(iter(row.keys()), None)
                    mv = row.get(first_col) if first_col else None
                if eco is None:
                    keys = list(row.keys())
                    if len(keys)>1: eco = row.get(keys[1])
                if mv and eco:
                    moves.append(mv); ecos.append(eco); names.append(name or '')
                if max_openings and len(moves) >= max_openings: break
    elif suffix in ('.yaml', '.yml'):
        try:
            import yaml
            with open(chosen,'r',encoding='utf-8') as fh:
                data = yaml.safe_load(fh)
            data_iter = data.values() if isinstance(data, dict) else data
            for entry in data_iter:
                if not isinstance(entry, dict): continue
                mv = entry.get('pgn') or entry.get('moves') or entry.get('uci') or entry.get('san') or entry.get('sequence')
                eco = entry.get('eco') or entry.get('code') or entry.get('ECO')
                name = entry.get('name') or entry.get('label') or entry.get('opening')
                if mv and eco:
                    moves.append(mv); ecos.append(eco); names.append(name or '')
                if max_openings and len(moves) >= max_openings: break
        except Exception:
            raise RuntimeError("YAML found but PyYAML not installed.")
    else:
        raise RuntimeError(f"Unsupported file format: {chosen.suffix}")
    if not moves:
        raise RuntimeError("No openings parsed.")
    np.savez_compressed(out_npz, moves=np.array(moves, dtype=object),
                        eco=np.array(ecos, dtype=object), name=np.array(names, dtype=object))
    print(f"Saved openings dataset to {out_npz} ({len(moves)} entries)")
    return out_npz

# ---------------------------
# Training small classifier
# ---------------------------
def train_openings(npz_path, epochs=3, batch_size=64, lr=1e-3, out_dir='openings_checkpoints'):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available.")
    os.makedirs(out_dir, exist_ok=True)
    data = np.load(npz_path, allow_pickle=True)
    moves, ecos = data['moves'], data['eco']
    unique = list(dict.fromkeys(ecos.tolist()))
    label2idx = {lab:i for i,lab in enumerate(unique)}
    y = np.array([label2idx.get(e, -1) for e in ecos]); mask = y>=0
    moves, y = moves[mask], y[mask]
    if len(y) == 0:
        raise RuntimeError("No data to train.")
    token_counts = {}; tokenized = []; max_len = 8
    for s in moves:
        toks = [t for t in str(s).strip().split() if not t.endswith('.')]
        toks = toks[:max_len]; tokenized.append(toks)
        for t in toks: token_counts[t] = token_counts.get(t,0)+1
    sorted_tokens = sorted(token_counts.items(), key=lambda x:-x[1])
    vocab = {tok:i+1 for i,(tok,_) in enumerate(sorted_tokens)}
    unk_id = len(vocab) + 1; vocab['<unk>'] = unk_id; vocab_size = unk_id + 1
    X = np.zeros((len(tokenized), max_len), dtype=int)
    for i,toks in enumerate(tokenized):
        for j,t in enumerate(toks):
            X[i,j] = vocab.get(t, unk_id)
    class DS(Dataset):
        def _init_(self,X,y): self.X=torch.from_numpy(X).long(); self.y=torch.from_numpy(y).long()
        def _len_(self): return len(self.y)
        def _getitem_(self,idx): return self.X[idx], self.y[idx]
    dl = DataLoader(DS(X,y), batch_size=batch_size, shuffle=True)
    class Net(nn.Module):
        def _init(self, vs): super().init_(); self.emb=nn.Embedding(vs,64,padding_idx=0); self.fc1=nn.Linear(64,128); self.fc2=nn.Linear(128,len(unique))
        def forward(self,x): e=self.emb(x).mean(1); return self.fc2(F.relu(self.fc1(e)))
    model = Net(vocab_size).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(1, epochs+1):
        model.train(); total=0; n=0
        for xb,yb in dl:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); n += 1
        print(f"Epoch {ep}: loss={total/max(1,n):.4f}")
        torch.save({'model': model.state_dict(), 'vocab': vocab, 'labels': unique, 'vocab_size': vocab_size},
                   os.path.join(out_dir, f'ckpt_epoch{ep}.pth'))
    print("Training done; checkpoints in", out_dir)

# ---------------------------
# Load classifier
# ---------------------------
def load_opening_classifier(ckpt_path):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available.")
    sd = torch.load(ckpt_path, map_location='cpu')
    vocab = sd.get('vocab'); labels = sd.get('labels'); state = sd.get('model')
    vocab_size = sd.get('vocab_size', None)
    if vocab is None or labels is None or state is None:
        raise RuntimeError("Checkpoint missing keys.")
    if vocab_size is None:
        vocab_size = max(vocab.values()) + 2
    class Model(nn.Module):
        def _init_(self, vs, emb=64, ncls=len(labels)):
            super()._init_(); self.emb=nn.Embedding(vs,emb,padding_idx=0); self.fc1=nn.Linear(emb,128); self.fc2=nn.Linear(128,ncls)
        def forward(self,x): e=self.emb(x).mean(1); return self.fc2(F.relu(self.fc1(e)))
    m = Model(vocab_size); m.load_state_dict(state); m.eval()
    return m, vocab, labels

# ---------------------------
# Selftest CLI
# ---------------------------
def selftest():
    os.makedirs('data/chess-openings', exist_ok=True)
    f = Path('data/chess-openings/openings.json')
    if not f.exists():
        sample = [
            {'pgn':'1. e4 e5 2. Nf3 Nc6','eco':'C60','name':'Ruy Lopez'},
            {'pgn':'1. d4 d5 2. c4','eco':'D00','name':'QG'},
            {'pgn':'1. e4 c5','eco':'B20','name':'Sicilian'}
        ]
        with open(f,'w',encoding='utf-8') as fh: json.dump(sample, fh, indent=2)
        print("Wrote sample openings")
    npz = build_openings_npz('data/chess-openings','openings_dataset.npz')
    if TORCH_AVAILABLE:
        try:
            train_openings(npz, epochs=1, batch_size=2, lr=1e-3, out_dir='openings_ckpt_sample')
        except Exception as e:
            print("Train failed:", e)
    else:
        print("Torch not available: skipping train")
    print("Selftest complete")

if __name__ == "_main_":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    if a.selftest:
        selftest()
