import chess
import random
import torch
import threading
import time
import logging
from datetime import datetime

# Logging setup
log_filename = f"chess_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')

# Setup
device_cpu = torch.device("cpu")
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"GPU available: {torch.cuda.is_available()}")
logging.info(f"GPU available: {torch.cuda.is_available()}")

# ANSI colors for board
WHITE_SQUARE = "\033[48;5;223m"
BLACK_SQUARE = "\033[48;5;137m"
RESET = "\033[0m"

# Unicode chess pieces
UNICODE_PIECES = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
}

def print_colored_board(board):
    board_str = ""
    for rank in range(7, -1, -1):
        line = f"{rank+1} "
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            bg = WHITE_SQUARE if (rank + file) % 2 == 0 else BLACK_SQUARE
            symbol = UNICODE_PIECES.get(piece.symbol(), ' ') if piece else ' '
            line += f"{bg} {symbol} {RESET}"
        board_str += line + "\n"
    board_str += "   a  b  c  d  e  f  g  h\n"
    print(board_str)
    logging.info(board_str)

def get_random_move(board):
    return random.choice(list(board.legal_moves))

def move_description(board, move):
    piece_symbol = UNICODE_PIECES[board.piece_at(move.from_square).symbol()]
    to_square = chess.square_name(move.to_square)
    return f"{piece_symbol} to {to_square}"

class ChessEngine:
    def __init__(self):
        self.board = chess.Board()

    def evaluate_board(self, device):
        piece_values = torch.tensor([1, 3, 3, 5, 9], device=device)
        eval_score = torch.tensor(0, device=device)
        for piece, value in zip([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN], piece_values):
            eval_score += len(self.board.pieces(piece, chess.WHITE)) * value
            eval_score -= len(self.board.pieces(piece, chess.BLACK)) * value
        return eval_score.item()

    def evaluate_parallel(self):
        result = {}

        def run_eval(name, device):
            start = time.time()
            score = self.evaluate_board(device)
            duration = time.time() - start
            result[name] = (score, duration)

        t1 = threading.Thread(target=run_eval, args=("CPU", device_cpu))
        t2 = threading.Thread(target=run_eval, args=("GPU", device_gpu))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        eval_log = (
            f"Evaluation results:\n"
            f"  CPU Score: {result['CPU'][0]} (in {result['CPU'][1]:.4f} sec)\n"
            f"  GPU Score: {result['GPU'][0]} (in {result['GPU'][1]:.4f} sec)"
        )
        print(eval_log)
        logging.info(eval_log)

    def play_game(self):
        print("Initial Board")
        logging.info("Initial Board")
        print_colored_board(self.board)

        while not self.board.is_game_over():
            time.sleep(0.4)
            self.evaluate_parallel()

            move = get_random_move(self.board)
            desc = move_description(self.board, move)
            move_log = f"{'White' if self.board.turn == chess.WHITE else 'Black'} plays: {desc}"
            print(move_log)
            logging.info(move_log)

            self.board.push(move)
            print_colored_board(self.board)
            print("\n" + "-"*30 + "\n")
            logging.info("-" * 30)

        game_over_msg = f"Game Over! Result: {self.board.result()}"
        print(game_over_msg)
        logging.info(game_over_msg)

def main():
    choice = input("Do you want to play manually (1) or let the engine play (2)? ")

    if choice == '1':
        board = chess.Board()
        print("Initial Board")
        logging.info("Initial Board")
        print_colored_board(board)

        while not board.is_game_over():
            move_input = input("Enter your move (e.g., e2e4): ").strip()
            try:
                move = chess.Move.from_uci(move_input)
                if move in board.legal_moves:
                    board.push(move)
                    print_colored_board(board)
                    logging.info("-" * 30)

                    ai_move = get_random_move(board)
                    desc = move_description(board, ai_move)
                    print(f"AI plays: {desc}")
                    logging.info(f"AI plays: {desc}")

                    board.push(ai_move)
                    print_colored_board(board)
                    logging.info("-" * 30)
                else:
                    print("Illegal move! Try again.")
            except ValueError:
                print("Invalid move format. Try again.")

        result_msg = f"Game Over! Result: {board.result()}"
        print(result_msg)
        logging.info(result_msg)

    elif choice == '2':
        engine = ChessEngine()
        engine.play_game()
    else:
        print("Invalid choice. Please restart and choose 1 or 2.")

if __name__ == "__main__":
    main()
