import streamlit as st
import chess
import chess.svg
import chess.engine
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle
import os
from io import BytesIO
import base64
from PIL import Image, ImageDraw


# Neural Network for Position Evaluation
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(773, 512)  # 64*12 + 5 features
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))  # Output between -1 and 1
        return x

class ChessAI:
    def __init__(self):
        self.model = ChessNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        self.load_model()
        
        # Piece-square tables for positional evaluation
        self.pst = {
            chess.PAWN: [
                [0,  0,  0,  0,  0,  0,  0,  0],
                [5, 10, 10,-20,-20, 10, 10,  5],
                [5, -5,-10,  0,  0,-10, -5,  5],
                [0,  0,  0, 20, 20,  0,  0,  0],
                [5,  5, 10, 25, 25, 10,  5,  5],
                [10, 10, 20, 30, 30, 20, 10, 10],
                [50, 50, 50, 50, 50, 50, 50, 50],
                [0,  0,  0,  0,  0,  0,  0,  0]
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
            ]
        }

    def board_to_tensor(self, board):
        """Convert chess board to neural network input tensor"""
        tensor = np.zeros(773)
        
        # Piece positions (64 squares * 12 piece types)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_idx = (piece.piece_type - 1) + (0 if piece.color else 6)
                tensor[square * 12 + piece_idx] = 1
        
        # Additional features
        tensor[768] = 1 if board.turn else 0  # Turn
        tensor[769] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
        tensor[770] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
        tensor[771] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
        tensor[772] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0
        
        return torch.FloatTensor(tensor)

    def evaluate_position_classical(self, board):
        """Classical evaluation function"""
        if board.is_checkmate():
            return -9999 if board.turn else 9999
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
            
        score = 0
        
        # Material count
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                if piece.piece_type in self.pst:
                    row, col = divmod(square, 8)
                    if not piece.color:  # Black pieces
                        row = 7 - row
                    value += self.pst[piece.piece_type][row][col] / 100
                
                score += value if piece.color else -value
        
        # Mobility bonus
        legal_moves = len(list(board.legal_moves))
        board.push(chess.Move.null())
        opponent_moves = len(list(board.legal_moves))
        board.pop()
        
        mobility = (legal_moves - opponent_moves) * 0.1
        score += mobility if board.turn else -mobility
        
        return score

    def evaluate_position_nn(self, board):
        """Neural network evaluation"""
        try:
            with torch.no_grad():
                tensor = self.board_to_tensor(board)
                evaluation = self.model(tensor).item()
                return evaluation * 1000  # Scale to centipawns
        except:
            return self.evaluate_position_classical(board)

    def minimax(self, board, depth, alpha, beta, maximizing_player, use_nn=True):
        """Minimax with alpha-beta pruning"""
        if depth == 0 or board.is_game_over():
            if use_nn:
                return self.evaluate_position_nn(board)
            else:
                return self.evaluate_position_classical(board)
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return self.evaluate_position_classical(board)
        
        # Move ordering: captures first
        legal_moves.sort(key=lambda move: board.is_capture(move), reverse=True)
        
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

    def get_best_move(self, board, depth=4, use_nn=True):
        """Get the best move using minimax"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        best_move = None
        best_value = float('-inf') if board.turn else float('inf')
        
        # Move ordering
        legal_moves.sort(key=lambda move: board.is_capture(move), reverse=True)
        
        for move in legal_moves:
            board.push(move)
            move_value = self.minimax(board, depth-1, float('-inf'), float('inf'), not board.turn, use_nn)
            board.pop()
            
            if board.turn:  # White maximizes
                if move_value > best_value:
                    best_value = move_value
                    best_move = move
            else:  # Black minimizes
                if move_value < best_value:
                    best_value = move_value
                    best_move = move
        
        return best_move

    def train_from_game(self, moves, result):
        """Train the neural network from a completed game"""
        positions = []
        board = chess.Board()
        
        for move in moves:
            positions.append(self.board_to_tensor(board))
            board.push(move)
        
        # Assign values based on game result
        if result == "1-0":  # White wins
            target_values = [1.0 - (i * 0.02) for i in range(len(positions))]
        elif result == "0-1":  # Black wins
            target_values = [-1.0 + (i * 0.02) for i in range(len(positions))]
        else:  # Draw
            target_values = [0.0] * len(positions)
        
        # Train on positions
        self.model.train()
        for pos, target in zip(positions, target_values):
            pred = self.model(pos)
            loss = self.criterion(pred, torch.FloatTensor([target]))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def self_play_training(self, num_games=10):
        """Self-play training session"""
        training_data = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for game_num in range(num_games):
            board = chess.Board()
            moves = []
            
            while not board.is_game_over() and len(moves) < 100:
                # Use different depths for variety
                depth = random.choice([2, 3, 4])
                move = self.get_best_move(board, depth, use_nn=True)
                if move:
                    moves.append(move)
                    board.push(move)
                else:
                    break
            
            # Get game result
            result = board.result()
            if result != "*":
                training_data.append((moves, result))
                self.train_from_game(moves, result)
            
            progress_bar.progress((game_num + 1) / num_games)
            status_text.text(f"Training game {game_num + 1}/{num_games} - Result: {result}")
        
        self.save_model()
        return len(training_data)

    def save_model(self):
        """Save the neural network model"""
        torch.save(self.model.state_dict(), 'chess_model.pth')

    def load_model(self):
        """Load the neural network model"""
        try:
            self.model.load_state_dict(torch.load('chess_model.pth'))
            self.model.eval()
        except:
            pass  # Use randomly initialized model

def create_board_image(board, size=400):
    """Create a simple board representation using PIL"""
    img = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(img)
    
    square_size = size // 8
    
    # Unicode chess pieces
    piece_symbols = {
        'P': '♙', 'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔',
        'p': '♟', 'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚'
    }
    
    # Draw squares and pieces
    for row in range(8):
        for col in range(8):
            square = chess.square(col, 7 - row)  # Flip vertically for display
            
            # Square color
            is_dark = (row + col) % 2 == 1
            color = '#D2691E' if is_dark else '#F5DEB3'  # Brown/beige
            
            x1, y1 = col * square_size, row * square_size
            x2, y2 = x1 + square_size, y1 + square_size
            draw.rectangle([x1, y1, x2, y2], fill=color)
            
            # Piece on square
            piece = board.piece_at(square)
            if piece:
                symbol = piece_symbols.get(piece.symbol(), piece.symbol())
                # Simple text drawing (limited font support)
                text_x = x1 + square_size // 2 - 10
                text_y = y1 + square_size // 2 - 10
                draw.text((text_x, text_y), symbol, fill='black')
    
    return img

# Streamlit App
def main():
    st.set_page_config(page_title="AI Chess Nemesis ♟️", page_icon="♟️", layout="wide")
    
    st.title("🏆 AI Chess Nemesis ♟️💻")
    st.markdown("*Train your own chess AI and challenge it to epic battles!*")
    
    # Initialize session state
    if 'board' not in st.session_state:
        st.session_state.board = chess.Board()
    if 'ai' not in st.session_state:
        st.session_state.ai = ChessAI()
    if 'move_history' not in st.session_state:
        st.session_state.move_history = []
    if 'game_over' not in st.session_state:
        st.session_state.game_over = False

    # Sidebar controls
    st.sidebar.header("🎮 Game Controls")
    
    col1, col2 = st.sidebar.columns(2)
    if col1.button("🔄 New Game"):
        st.session_state.board = chess.Board()
        st.session_state.move_history = []
        st.session_state.game_over = False
        #st.rerun()
    
    if col2.button("↩️ Undo Move") and len(st.session_state.move_history) >= 2:
        st.session_state.board.pop()  # Undo AI move
        st.session_state.board.pop()  # Undo player move
        st.session_state.move_history = st.session_state.move_history[:-2]
        st.session_state.game_over = False
       # st.rerun()
    
    # AI Settings
    st.sidebar.header("🤖 AI Settings")
    ai_depth = st.sidebar.slider("AI Thinking Depth", 1, 6, 4)
    use_neural_net = st.sidebar.checkbox("Use Neural Network", value=True)
    
    # Training Section
    st.sidebar.header("🧠 AI Training")
    if st.sidebar.button("🎯 Self-Play Training"):
        num_games = st.sidebar.number_input("Training Games", 1, 50, 10)
        with st.spinner("Training AI..."):
            games_trained = st.session_state.ai.self_play_training(num_games)
        st.sidebar.success(f"✅ Trained on {games_trained} games!")
    
    # Main game area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🏁 Game Board")
        
        # Generate board image directly
        board_img = create_board_image(st.session_state.board)
        st.image(board_img, width=400)
        
        # Move input
        if not st.session_state.game_over:
            move_input = st.text_input(
                "Enter your move (e.g., e2e4, Nf3, O-O):",
                key="move_input",
                help="Use standard algebraic notation or UCI format"
            )
            
            col_move1, col_move2 = st.columns(2)
            if col_move1.button("▶️ Make Move"):
                try:
                    # Parse move
                    if move_input in ['O-O', '0-0']:
                        move = chess.Move.from_uci("e1g1" if st.session_state.board.turn else "e8g8")
                    elif move_input in ['O-O-O', '0-0-0']:
                        move = chess.Move.from_uci("e1c1" if st.session_state.board.turn else "e8c8")
                    else:
                        try:
                            move = st.session_state.board.parse_san(move_input)
                        except:
                            move = chess.Move.from_uci(move_input)
                    
                    if move in st.session_state.board.legal_moves:
                        # Make player move
                        st.session_state.board.push(move)
                        st.session_state.move_history.append(move)
                        
                        # Check if game is over
                        if st.session_state.board.is_game_over():
                            st.session_state.game_over = True
                        else:
                            # AI move
                            with st.spinner("🤔 AI is thinking..."):
                                ai_move = st.session_state.ai.get_best_move(
                                    st.session_state.board, 
                                    depth=ai_depth,
                                    use_nn=use_neural_net
                                )
                            
                            if ai_move:
                                st.session_state.board.push(ai_move)
                                st.session_state.move_history.append(ai_move)
                                
                                if st.session_state.board.is_game_over():
                                    st.session_state.game_over = True
                        
                       # st.rerun()
                    else:
                        st.error("❌ Illegal move!")
                except Exception as e:
                    st.error(f"❌ Invalid move format: {e}")
            
            if col_move2.button("💡 Get Hint"):
                with st.spinner("Finding best move..."):
                    hint_move = st.session_state.ai.get_best_move(
                        st.session_state.board,
                        depth=ai_depth,
                        use_nn=use_neural_net
                    )
                if hint_move:
                    st.info(f"💡 Suggested move: **{hint_move}**")
    
    with col2:
        st.subheader("📊 Game Info")
        
        # Game status
        if st.session_state.board.is_checkmate():
            winner = "Black" if st.session_state.board.turn else "White"
            st.error(f"🏆 **{winner} wins by checkmate!**")
        elif st.session_state.board.is_stalemate():
            st.warning("🤝 **Stalemate - It's a draw!**")
        elif st.session_state.board.is_insufficient_material():
            st.warning("🤝 **Draw by insufficient material!**")
        elif st.session_state.board.is_check():
            st.warning("⚠️ **Check!**")
        else:
            turn = "White" if st.session_state.board.turn else "Black"
            st.success(f"▶️ **{turn} to move**")
        
        # Position evaluation
        eval_score = st.session_state.ai.evaluate_position_classical(st.session_state.board)
        if use_neural_net:
            nn_eval = st.session_state.ai.evaluate_position_nn(st.session_state.board)
            st.metric("🧠 Neural Net Eval", f"{nn_eval:.1f}")
        st.metric("⚖️ Classical Eval", f"{eval_score:.1f}")
        
        # Move history
        st.subheader("📝 Move History")
        if st.session_state.move_history:
            move_pairs = []
            for i in range(0, len(st.session_state.move_history), 2):
                white_move = str(st.session_state.move_history[i])
                black_move = str(st.session_state.move_history[i+1]) if i+1 < len(st.session_state.move_history) else ""
                move_pairs.append(f"{i//2 + 1}. {white_move} {black_move}")
            
            st.text_area("Moves", "\n".join(move_pairs[-10:]), height=200)
        
        # Legal moves
        legal_moves = list(st.session_state.board.legal_moves)
        st.caption(f"💭 Legal moves: {len(legal_moves)}")

if __name__ == "__main__":
    main()
