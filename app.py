import streamlit as st
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
import os
from datetime import datetime

# Neural Network for Position Evaluation
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(773, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x

class ChessAI:
    def __init__(self):
        self.model = ChessNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.training_history = []
        self.games_played = 0
        
        # Enhanced piece-square tables
        self.pst = {
            chess.PAWN: [
                [ 0,  0,  0,  0,  0,  0,  0,  0],
                [50, 50, 50, 50, 50, 50, 50, 50],
                [10, 10, 20, 30, 30, 20, 10, 10],
                [ 5,  5, 10, 25, 25, 10,  5,  5],
                [ 0,  0,  0, 20, 20,  0,  0,  0],
                [ 5, -5,-10,  0,  0,-10, -5,  5],
                [ 5, 10, 10,-20,-20, 10, 10,  5],
                [ 0,  0,  0,  0,  0,  0,  0,  0]
            ],
            chess.KNIGHT: [
                [-50,-40,-30,-30,-30,-30,-40,-50],
                [-40,-20,  0,  0,  0,  0,-20,-40],
                [-30,  0, 10, 15, 15, 10,  0,-30],
                [-30,  5, 15, 20, 20, 15,  5,-30],
                [-30,  0, 15, 20, 20, 15,  0,-30],
                [-30,  5, 10, 15, 15, 10,  5,-30],
                [-40,-20,  0,  5,  5,  0,-20,-40],
                [-50,-40,-30,-30,-30,-30,-40,-50]
            ],
            chess.BISHOP: [
                [-20,-10,-10,-10,-10,-10,-10,-20],
                [-10,  0,  0,  0,  0,  0,  0,-10],
                [-10,  0,  5, 10, 10,  5,  0,-10],
                [-10,  5,  5, 10, 10,  5,  5,-10],
                [-10,  0, 10, 10, 10, 10,  0,-10],
                [-10, 10, 10, 10, 10, 10, 10,-10],
                [-10,  5,  0,  0,  0,  0,  5,-10],
                [-20,-10,-10,-10,-10,-10,-10,-20]
            ],
            chess.ROOK: [
                [ 0,  0,  0,  0,  0,  0,  0,  0],
                [ 5, 10, 10, 10, 10, 10, 10,  5],
                [-5,  0,  0,  0,  0,  0,  0, -5],
                [-5,  0,  0,  0,  0,  0,  0, -5],
                [-5,  0,  0,  0,  0,  0,  0, -5],
                [-5,  0,  0,  0,  0,  0,  0, -5],
                [-5,  0,  0,  0,  0,  0,  0, -5],
                [ 0,  0,  0,  5,  5,  0,  0,  0]
            ],
            chess.QUEEN: [
                [-20,-10,-10, -5, -5,-10,-10,-20],
                [-10,  0,  0,  0,  0,  0,  0,-10],
                [-10,  0,  5,  5,  5,  5,  0,-10],
                [ -5,  0,  5,  5,  5,  5,  0, -5],
                [  0,  0,  5,  5,  5,  5,  0, -5],
                [-10,  5,  5,  5,  5,  5,  0,-10],
                [-10,  0,  5,  0,  0,  0,  0,-10],
                [-20,-10,-10, -5, -5,-10,-10,-20]
            ],
            chess.KING: [
                [-30,-40,-40,-50,-50,-40,-40,-30],
                [-30,-40,-40,-50,-50,-40,-40,-30],
                [-30,-40,-40,-50,-50,-40,-40,-30],
                [-30,-40,-40,-50,-50,-40,-40,-30],
                [-20,-30,-30,-40,-40,-30,-30,-20],
                [-10,-20,-20,-20,-20,-20,-20,-10],
                [ 20, 20,  0,  0,  0,  0, 20, 20],
                [ 20, 30, 10,  0,  0, 10, 30, 20]
            ]
        }
        
        self.piece_values = {
            chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
        }
        
        self.load_model()

    def board_to_tensor(self, board):
        """Convert board to neural network input"""
        tensor = np.zeros(773)
        
        # Piece positions (64 squares * 12 piece types)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_idx = (piece.piece_type - 1) + (0 if piece.color else 6)
                tensor[square * 12 + piece_idx] = 1
        
        # Game state features
        tensor[768] = 1 if board.turn else 0
        tensor[769] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
        tensor[770] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
        tensor[771] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
        tensor[772] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0
        
        return torch.FloatTensor(tensor)

    def evaluate_position(self, board, use_nn=True):
        """Advanced position evaluation"""
        if board.is_checkmate():
            return -30000 if board.turn else 30000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        # Neural network evaluation
        if use_nn:
            try:
                with torch.no_grad():
                    tensor = self.board_to_tensor(board)
                    nn_eval = self.model(tensor).item() * 2000
                    classical_eval = self.evaluate_classical(board)
                    # Blend evaluations (more NN weight as training progresses)
                    blend_factor = min(0.8, self.games_played / 1000)
                    return nn_eval * blend_factor + classical_eval * (1 - blend_factor)
            except:
                pass
        
        return self.evaluate_classical(board)

    def evaluate_classical(self, board):
        """Classical evaluation function"""
        score = 0
        
        # Material and position
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                
                # Add positional bonus
                if piece.piece_type in self.pst:
                    row, col = divmod(square, 8)
                    if not piece.color:  # Black
                        row = 7 - row
                    value += self.pst[piece.piece_type][row][col]
                
                score += value if piece.color else -value
        
        # Mobility
        mobility = len(list(board.legal_moves))
        board.push(chess.Move.null())
        opp_mobility = len(list(board.legal_moves))
        board.pop()
        
        mobility_bonus = (mobility - opp_mobility) * 10
        score += mobility_bonus if board.turn else -mobility_bonus
        
        # King safety
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        
        if white_king and black_king:
            # Penalize exposed kings in middlegame
            piece_count = len(board.piece_map())
            if piece_count > 20:  # Middlegame
                white_king_safety = self.king_safety(board, white_king, chess.WHITE)
                black_king_safety = self.king_safety(board, black_king, chess.BLACK)
                score += white_king_safety - black_king_safety
        
        return score

    def king_safety(self, board, king_square, color):
        """Evaluate king safety"""
        safety = 0
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        # Pawn shield
        shield_squares = []
        if color == chess.WHITE:
            if king_rank < 7:
                for f in range(max(0, king_file-1), min(8, king_file+2)):
                    shield_squares.append(chess.square(f, king_rank + 1))
        else:
            if king_rank > 0:
                for f in range(max(0, king_file-1), min(8, king_file+2)):
                    shield_squares.append(chess.square(f, king_rank - 1))
        
        for square in shield_squares:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                safety += 30
        
        return safety

    def minimax(self, board, depth, alpha, beta, maximizing_player, use_nn=True):
        """Enhanced minimax with advanced pruning"""
        if depth == 0 or board.is_game_over():
            return self.evaluate_position(board, use_nn)
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return self.evaluate_position(board, use_nn)
        
        # Advanced move ordering
        scored_moves = []
        for move in legal_moves:
            score = 0
            
            # Captures
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    score += self.piece_values[captured_piece.piece_type]
                    # MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
                    attacker = board.piece_at(move.from_square)
                    if attacker:
                        score += self.piece_values[captured_piece.piece_type] - self.piece_values[attacker.piece_type]
            
            # Checks
            board.push(move)
            if board.is_check():
                score += 50
            board.pop()
            
            # Promotions
            if move.promotion:
                score += 800
            
            scored_moves.append((score, move))
        
        # Sort by score (best first)
        scored_moves.sort(reverse=True)
        legal_moves = [move for _, move in scored_moves]
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth-1, alpha, beta, False, use_nn)
                board.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth-1, alpha, beta, True, use_nn)
                board.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def get_best_move(self, board, depth=5, use_nn=True):
        """Get best move with iterative deepening"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        best_move = None
        
        # Iterative deepening
        for current_depth in range(1, depth + 1):
            current_best = None
            best_value = float('-inf') if board.turn else float('inf')
            
            for move in legal_moves:
                board.push(move)
                move_value = self.minimax(board, current_depth-1, float('-inf'), float('inf'), not board.turn, use_nn)
                board.pop()
                
                if board.turn:  # White maximizes
                    if move_value > best_value:
                        best_value = move_value
                        current_best = move
                else:  # Black minimizes
                    if move_value < best_value:
                        best_value = move_value
                        current_best = move
            
            if current_best:
                best_move = current_best
        
        return best_move

    def train_from_game(self, moves, result):
        """Train neural network from game data"""
        if len(moves) < 10:  # Skip very short games
            return
        
        positions = []
        evaluations = []
        board = chess.Board()
        
        for i, move in enumerate(moves):
            positions.append(self.board_to_tensor(board))
            
            # Assign target values based on result and game phase
            game_progress = i / len(moves)
            
            if result == "1-0":  # White wins
                target = 1.0 - (game_progress * 0.3)
                if not board.turn:  # Black position
                    target = -target
            elif result == "0-1":  # Black wins
                target = -1.0 + (game_progress * 0.3)
                if not board.turn:  # Black position
                    target = -target
            else:  # Draw
                target = 0.0
            
            evaluations.append(target)
            board.push(move)
        
        # Batch training
        self.model.train()
        for i in range(0, len(positions), 32):  # Batch size 32
            batch_pos = positions[i:i+32]
            batch_eval = evaluations[i:i+32]
            
            if batch_pos:
                batch_tensor = torch.stack(batch_pos)
                target_tensor = torch.FloatTensor(batch_eval).unsqueeze(1)
                
                predictions = self.model(batch_tensor)
                loss = self.criterion(predictions, target_tensor)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def self_play_training(self, num_games=50, progress_callback=None):
        """Intensive self-play training"""
        training_results = []
        
        for game_num in range(num_games):
            board = chess.Board()
            moves = []
            move_count = 0
            
            # Vary playing strength for diversity
            depths = [3, 4, 5, 6]
            
            while not board.is_game_over() and move_count < 150:
                # Use different depths and occasionally random moves for exploration
                if random.random() < 0.1:  # 10% random moves for exploration
                    move = random.choice(list(board.legal_moves))
                else:
                    depth = random.choice(depths)
                    move = self.get_best_move(board, depth, use_nn=True)
                
                if move:
                    moves.append(move)
                    board.push(move)
                    move_count += 1
                else:
                    break
            
            result = board.result()
            if result != "*" and len(moves) >= 20:
                self.train_from_game(moves, result)
                training_results.append({
                    'game': game_num + 1,
                    'result': result,
                    'moves': len(moves),
                    'final_eval': self.evaluate_position(board, use_nn=True)
                })
                self.games_played += 1
            
            if progress_callback:
                progress_callback(game_num + 1, num_games, result)
        
        self.save_model()
        return training_results

    def save_model(self):
        """Save model and training history"""
        torch.save({
            'model_state': self.model.state_dict(),
            'games_played': self.games_played,
            'training_history': self.training_history
        }, 'chess_ai_model.pth')
        
        # Save readable stats
        stats = {
            'games_played': self.games_played,
            'last_updated': datetime.now().isoformat(),
            'model_version': '1.0'
        }
        with open('ai_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

    def load_model(self):
        """Load model and training history"""
        try:
            checkpoint = torch.load('chess_ai_model.pth')
            self.model.load_state_dict(checkpoint['model_state'])
            self.games_played = checkpoint.get('games_played', 0)
            self.training_history = checkpoint.get('training_history', [])
            self.model.eval()
        except:
            self.games_played = 0
            self.training_history = []

def display_board_text(board):
    """Display board in text format"""
    board_str = str(board)
    lines = board_str.split('\n')
    
    # Add file labels
    labeled_lines = []
    for i, line in enumerate(lines):
        rank = 8 - i
        labeled_lines.append(f"{rank} {line}")
    
    labeled_lines.append("  a b c d e f g h")
    return '\n'.join(labeled_lines)

def main():
    st.set_page_config(page_title="AI Chess Nemesis", layout="wide")
    
    st.title("🏆 AI Chess Nemesis - Master Training")
    st.markdown("*Pure chess AI that learns and evolves through self-play*")
    
    # Initialize
    if 'board' not in st.session_state:
        st.session_state.board = chess.Board()
    if 'ai' not in st.session_state:
        st.session_state.ai = ChessAI()
    if 'move_history' not in st.session_state:
        st.session_state.move_history = []
    if 'game_pgn' not in st.session_state:
        st.session_state.game_pgn = []

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Game Board")
        
        # Display board
        board_display = display_board_text(st.session_state.board)
        st.code(board_display, language=None)
        
        # Game status
        if st.session_state.board.is_checkmate():
            winner = "Black" if st.session_state.board.turn else "White"
            st.success(f"🏆 **{winner} wins by checkmate!**")
        elif st.session_state.board.is_check():
            st.warning("⚠️ **Check!**")
        elif st.session_state.board.is_stalemate():
            st.info("🤝 **Stalemate!**")
        else:
            turn = "White" if st.session_state.board.turn else "Black"
            st.info(f"▶️ **{turn} to move**")
        
        # Move input
        if not st.session_state.board.is_game_over():
            col_a, col_b, col_c = st.columns([2, 1, 1])
            
            with col_a:
                move_input = st.text_input("Your move (e.g., e4, Nf3, O-O):", key="move_input")
            
            with col_b:
                if st.button("▶️ Move"):
                    try:
                        move = st.session_state.board.parse_san(move_input)
                        if move in st.session_state.board.legal_moves:
                            st.session_state.board.push(move)
                            st.session_state.move_history.append(move)
                            st.session_state.game_pgn.append(move_input)
                            
                            if not st.session_state.board.is_game_over():
                                with st.spinner("AI thinking..."):
                                    ai_depth = min(6, 4 + st.session_state.ai.games_played // 100)
                                    ai_move = st.session_state.ai.get_best_move(
                                        st.session_state.board, depth=ai_depth, use_nn=True
                                    )
                                
                                if ai_move:
                                    st.session_state.board.push(ai_move)
                                    st.session_state.move_history.append(ai_move)
                                    st.session_state.game_pgn.append(str(ai_move))
                            
                            st.rerun()
                        else:
                            st.error("Illegal move!")
                    except:
                        st.error("Invalid move format!")
            
            with col_c:
                if st.button("💡 Hint"):
                    with st.spinner("Analyzing..."):
                        hint = st.session_state.ai.get_best_move(st.session_state.board, depth=4)
                    if hint:
                        st.success(f"💡 {hint}")
        
        # Game controls
        col_x, col_y, col_z = st.columns(3)
        if col_x.button("🔄 New Game"):
            st.session_state.board = chess.Board()
            st.session_state.move_history = []
            st.session_state.game_pgn = []
            st.rerun()
        
        if col_y.button("↩️ Undo") and len(st.session_state.move_history) >= 2:
            st.session_state.board.pop()
            st.session_state.board.pop()
            st.session_state.move_history = st.session_state.move_history[:-2]
            st.session_state.game_pgn = st.session_state.game_pgn[:-2]
            st.rerun()
        
        if col_z.button("📋 Export PGN"):
            pgn_text = " ".join(st.session_state.game_pgn)
            st.text_area("Game PGN:", pgn_text, height=100)
    
    with col2:
        st.subheader("🧠 AI Brain")
        
        # AI stats
        ai_strength = min(3000, 1200 + st.session_state.ai.games_played * 2)
        st.metric("🎯 Estimated Rating", f"{ai_strength}")
        st.metric("🎮 Training Games", st.session_state.ai.games_played)
        
        # Position evaluation
        current_eval = st.session_state.ai.evaluate_position(st.session_state.board, use_nn=True)
        st.metric("⚖️ Position Eval", f"{current_eval:.0f}")
        
        # Training controls
        st.subheader("🚀 Training Lab")
        
        training_games = st.selectbox("Training Intensity", [10, 25, 50, 100, 200], index=1)
        
        if st.button("🎯 Begin Training"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.empty()
            
            def progress_callback(current, total, result):
                progress_bar.progress(current / total)
                status_text.text(f"Game {current}/{total} - Result: {result}")
            
            with st.spinner(f"Training AI with {training_games} games..."):
                results = st.session_state.ai.self_play_training(
                    num_games=training_games,
                    progress_callback=progress_callback
                )
            
            # Show results
            if results:
                wins = sum(1 for r in results if r['result'] == '1-0')
                losses = sum(1 for r in results if r['result'] == '0-1')
                draws = sum(1 for r in results if r['result'] == '1/2-1/2')
                
                st.success(f"✅ Training complete!")
                st.write(f"**Results:** {wins}W-{losses}L-{draws}D")
                
                avg_moves = np.mean([r['moves'] for r in results])
                st.write(f"**Average game length:** {avg_moves:.1f} moves")
            
            progress_bar.empty()
            status_text.empty()
        
        # Quick training
        if st.button("⚡ Quick Session (10 games)"):
            with st.spinner("Quick training..."):
                results = st.session_state.ai.self_play_training(10)
            st.success(f"✅ +{len(results)} games trained!")
        
        # Move history
        st.subheader("📝 Game Moves")
        if st.session_state.game_pgn:
            move_text = ""
            for i in range(0, len(st.session_state.game_pgn), 2):
                move_num = i // 2 + 1
                white_move = st.session_state.game_pgn[i]
                black_move = st.session_state.game_pgn[i+1] if i+1 < len(st.session_state.game_pgn) else ""
                move_text += f"{move_num}. {white_move} {black_move}\n"
            
            st.text_area("Moves:", move_text, height=150)
        
        # Legal moves preview
        legal_moves = list(st.session_state.board.legal_moves)
        if legal_moves:
            st.caption(f"💭 {len(legal_moves)} legal moves available")
            sample_moves = [str(move) for move in legal_moves[:8]]
            st.caption(f"Examples: {', '.join(sample_moves)}")

if __name__ == "__main__":
    main()
