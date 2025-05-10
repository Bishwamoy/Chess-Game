import tkinter as tk
from tkinter import simpledialog
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

device_cpu = torch.device("cpu")
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UNICODE_PIECES = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
}

class ChessGUI:
    def __init__(self, master, mode="engine"):
        self.master = master
        self.master.title("Chess Game")
        self.board = chess.Board()
        self.mode = mode
        self.selected_square = None

        self.canvas = tk.Canvas(master, width=640, height=640)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.handle_click)

        self.status = tk.Label(master, text="", font=("Arial", 14))
        self.status.pack()

        self.square_size = 80
        self.draw_board()
        self.update_board()

        if self.mode == "engine":
            self.master.after(1000, self.play_engine_game)

    def draw_board(self):
        colors = ["#f0d9b5", "#b58863"]
        for rank in range(8):
            for file in range(8):
                x1 = file * self.square_size
                y1 = rank * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                color = colors[(rank + file) % 2]
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, tags="square")

    def update_board(self):
        self.canvas.delete("piece")
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                file = chess.square_file(square)
                rank = 7 - chess.square_rank(square)
                x = file * self.square_size + 40
                y = rank * self.square_size + 40
                symbol = UNICODE_PIECES[piece.symbol()]
                self.canvas.create_text(x, y, text=symbol, font=("Arial", 32), tags="piece")

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
        logging.info(eval_log)
        self.status.config(text=eval_log.split("\n")[1])

    def play_engine_game(self):
        if self.board.is_game_over():
            result = f"Game Over! Result: {self.board.result()}"
            self.status.config(text=result)
            logging.info(result)
            return

        self.evaluate_parallel()
        move = random.choice(list(self.board.legal_moves))
        piece = self.board.piece_at(move.from_square).symbol()
        to_sq = chess.square_name(move.to_square)
        move_desc = f"{UNICODE_PIECES[piece]} to {to_sq}"
        logging.info(f"{'White' if self.board.turn == chess.WHITE else 'Black'} plays: {move_desc}")

        self.board.push(move)
        self.update_board()
        self.master.after(1000, self.play_engine_game)

    def handle_click(self, event):
        if self.mode == "engine":
            return  # Disable manual play in engine mode

        col = event.x // self.square_size
        row = 7 - (event.y // self.square_size)
        clicked_square = chess.square(col, row)

        if self.selected_square is None:
            piece = self.board.piece_at(clicked_square)
            if piece and piece.color == self.board.turn:
                self.selected_square = clicked_square
        else:
            move = chess.Move(self.selected_square, clicked_square)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.update_board()
                logging.info(f"Player plays: {self.board.peek()}")
                self.selected_square = None

                if self.mode == "vs_ai" and not self.board.is_game_over():
                    self.master.after(500, self.ai_move)
            else:
                self.selected_square = None  # Reset selection if move is illegal

    def ai_move(self):
        move = random.choice(list(self.board.legal_moves))
        self.board.push(move)
        logging.info(f"AI plays: {move}")
        self.update_board()

def run_gui():
    root = tk.Tk()
    root.withdraw()

    mode = simpledialog.askstring("Choose Mode", "Enter mode:\nmanual / vs_ai / engine").lower()

    if mode not in ["manual", "vs_ai", "engine"]:
        print("Invalid mode selected. Exiting.")
        return

    root.deiconify()
    gui = ChessGUI(root, mode=mode)
    root.mainloop()

if __name__ == "__main__":
    run_gui()
