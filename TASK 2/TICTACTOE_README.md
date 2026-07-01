# Tic-Tac-Toe AI (Minimax + Alpha-Beta Pruning)

A command-line Tic-Tac-Toe game where you play against an **unbeatable AI**. The AI uses the **Minimax algorithm with Alpha-Beta Pruning** to search the entire game tree and always choose the optimal move. Since Tic-Tac-Toe is a "solved" game, perfect play from the AI means the best a human can ever achieve is a **draw**.

## Features

- Unbeatable AI opponent powered by Minimax + Alpha-Beta Pruning
- Choose to go first or second
- Clean terminal board display with numbered positions (1-9)
- Input validation (rejects invalid or already-taken squares)
- Randomized tie-breaking so the AI doesn't play identically every game
- Play multiple rounds in one session

## Requirements

- Python 3.6+
- No external libraries needed (uses only the standard library: `math`, `random`)

## How to Run

```bash
python tictactoe_ai.py
```

You'll be shown a position guide, asked whether you want to go first, and then play by entering a number 1-9 for your move.

```
Positions are numbered like this:
 1 | 2 | 3
---+---+---
 4 | 5 | 6
---+---+---
 7 | 8 | 9

Do you want to go first? (y/n): y

Your move (1-9, top-left to bottom-right): 5
```

## How the AI Works

### Minimax

Minimax explores the **entire game tree** from the current position to every possible ending. It assumes:
- The AI (`O`) is the **maximizer** — trying to reach the highest score.
- The human (`X`) is the **minimizer** — trying to reach the lowest score.

Terminal positions are scored:
| Outcome        | Score       |
|----------------|-------------|
| AI wins        | `10 - depth`|
| Human wins     | `depth - 10`|
| Draw           | `0`         |

Subtracting/adding `depth` makes the AI prefer **faster wins** and **slower losses** — it won't take 8 moves to win when it could win in 3, and it'll delay a loss as long as possible if one is forced.

The AI evaluates every legal move by simulating it, then recursively calls minimax to see how the opponent would respond, all the way down to game-over states. It picks the move with the best guaranteed outcome, assuming the opponent also plays perfectly.

### Alpha-Beta Pruning

A plain minimax search on Tic-Tac-Toe is fast enough on its own, but alpha-beta pruning is included to demonstrate the standard optimization:

- `alpha` = the best score the maximizer (AI) can guarantee so far
- `beta` = the best score the minimizer (human) can guarantee so far

If at any point `beta <= alpha`, the current branch can't affect the final decision, so the search **stops exploring it** ("cuts it off"). This doesn't change the result — it just skips work that can't matter, which becomes significant in games with much larger search spaces.

### Opening Move Shortcut

On a completely empty board, minimax is symmetric (all corners are equivalent, etc.), so the AI skips the full search and immediately takes the **center** — the well-known optimal opening move. This is purely a speed optimization for the first move.

## Verifying "Unbeatable"

This was tested two ways:
1. **AI vs. random-move player**, 100 simulated games (both going first and second) — the AI never lost a single game.
2. **AI vs. AI** (perfect play on both sides) — the game always ends in a **draw**, exactly as game theory predicts for optimal Tic-Tac-Toe.

## File Structure

```
tictactoe_ai.py   # Game logic, Minimax + Alpha-Beta AI, and CLI game loop
README.md          # This file
```

## Extending the Project

- **Difficulty levels**: limit search depth or occasionally have the AI pick a random (non-optimal) move to make it beatable.
- **GUI**: wrap the game logic in Tkinter, Pygame, or a web frontend instead of the terminal.
- **Larger boards**: extend to variants like 4x4 or "gravity" Tic-Tac-Toe (Connect Four uses similar minimax ideas, though a full search becomes infeasible without heuristics and deeper pruning).
- **Move ordering**: for larger games, exploring likely-best moves first improves alpha-beta pruning efficiency even further.

## License

Free to use and modify for learning purposes.
