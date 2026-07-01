"""
Tic-Tac-Toe AI
==============
A command-line Tic-Tac-Toe game where you play against an unbeatable AI.

The AI uses the Minimax algorithm with Alpha-Beta Pruning to search the
full game tree and always pick the optimal move. Against perfect play,
the best a human can achieve is a draw.

Concepts demonstrated:
- Game tree search (Minimax)
- Alpha-Beta Pruning (cutting off branches that can't affect the outcome)
- Terminal state detection (win/lose/draw)
- Heuristic scoring for a solved game

Run it with:  python tictactoe_ai.py
"""

import math
import random

HUMAN = "X"
AI = "O"
EMPTY = " "


class TicTacToe:
    def __init__(self):
        # Board is a flat list of 9 cells, indexed 0-8:
        #  0 | 1 | 2
        #  3 | 4 | 5
        #  6 | 7 | 8
        self.board = [EMPTY] * 9

    # ---------- Board utilities ----------

    def print_board(self):
        b = self.board
        rows = [b[0:3], b[3:6], b[6:9]]
        print()
        for i, row in enumerate(rows):
            print(f" {row[0]} | {row[1]} | {row[2]} ")
            if i < 2:
                print("---+---+---")
        print()

    def available_moves(self, board=None):
        board = board if board is not None else self.board
        return [i for i, cell in enumerate(board) if cell == EMPTY]

    def make_move(self, index, player, board=None):
        board = board if board is not None else self.board
        board[index] = player

    def undo_move(self, index, board=None):
        board = board if board is not None else self.board
        board[index] = EMPTY

    WIN_LINES = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
        (0, 4, 8), (2, 4, 6),             # diagonals
    ]

    def winner(self, board=None):
        board = board if board is not None else self.board
        for a, b, c in self.WIN_LINES:
            if board[a] != EMPTY and board[a] == board[b] == board[c]:
                return board[a]
        return None

    def is_full(self, board=None):
        board = board if board is not None else self.board
        return EMPTY not in board

    def game_over(self, board=None):
        board = board if board is not None else self.board
        return self.winner(board) is not None or self.is_full(board)

    # ---------- Minimax with Alpha-Beta Pruning ----------

    def minimax(self, board, depth, is_maximizing, alpha, beta):
        """
        Returns the minimax score of `board` from the AI's perspective.
        AI (O) is the maximizer, Human (X) is the minimizer.
        Scores prefer faster wins and slower losses via the depth term.
        """
        winner = self.winner(board)
        if winner == AI:
            return 10 - depth
        if winner == HUMAN:
            return depth - 10
        if self.is_full(board):
            return 0

        if is_maximizing:
            best_score = -math.inf
            for move in self.available_moves(board):
                board[move] = AI
                score = self.minimax(board, depth + 1, False, alpha, beta)
                board[move] = EMPTY
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break  # beta cutoff: minimizer won't allow this branch
            return best_score
        else:
            best_score = math.inf
            for move in self.available_moves(board):
                board[move] = HUMAN
                score = self.minimax(board, depth + 1, True, alpha, beta)
                board[move] = EMPTY
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break  # alpha cutoff: maximizer won't allow this branch
            return best_score

    def best_move(self):
        """Find the AI's optimal move using minimax + alpha-beta pruning."""
        moves = self.available_moves()

        # Opening-move optimization: if the board is empty, skip the full
        # search (which is symmetric/slow-ish) and just take the center
        # or a corner — standard optimal opening play.
        if len(moves) == 9:
            return 4  # center is always at least as good as any other opening

        best_score = -math.inf
        best_moves = []
        for move in moves:
            self.board[move] = AI
            score = self.minimax(self.board, 0, False, -math.inf, math.inf)
            self.board[move] = EMPTY
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        # Among equally optimal moves, pick randomly so the AI isn't
        # predictably repetitive across games.
        return random.choice(best_moves)


def get_human_move(game):
    while True:
        raw = input("Your move (1-9, top-left to bottom-right): ").strip()
        if not raw.isdigit():
            print("Please enter a number from 1 to 9.")
            continue
        pos = int(raw) - 1
        if pos < 0 or pos > 8:
            print("Please enter a number from 1 to 9.")
            continue
        if game.board[pos] != EMPTY:
            print("That square is already taken. Try again.")
            continue
        return pos


def print_position_guide():
    print("Positions are numbered like this:")
    print(" 1 | 2 | 3 ")
    print("---+---+---")
    print(" 4 | 5 | 6 ")
    print("---+---+---")
    print(" 7 | 8 | 9 ")


def play_game():
    print("=" * 40)
    print("   TIC-TAC-TOE vs. an Unbeatable AI")
    print("=" * 40)
    print_position_guide()

    game = TicTacToe()

    first = input("\nDo you want to go first? (y/n): ").strip().lower()
    human_turn = first.startswith("y")

    game.print_board()

    while not game.game_over():
        if human_turn:
            pos = get_human_move(game)
            game.make_move(pos, HUMAN)
        else:
            print("AI is thinking...")
            pos = game.best_move()
            game.make_move(pos, AI)
            print(f"AI plays position {pos + 1}.")

        game.print_board()
        human_turn = not human_turn

    winner = game.winner()
    if winner == HUMAN:
        print("Congratulations, you won! (This shouldn't be possible against perfect play — nice job if it happened!)")
    elif winner == AI:
        print("The AI wins!")
    else:
        print("It's a draw!")


def main():
    while True:
        play_game()
        again = input("\nPlay again? (y/n): ").strip().lower()
        if not again.startswith("y"):
            print("Thanks for playing!")
            break


if __name__ == "__main__":
    main()
