"""
env.py: Production-grade Connect Four MDP Environment.
Tracks states, handles valid actions, and performs vector-based win checks.
"""
import numpy as np

class ConnectFourEnv:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.reset()

    def reset(self):
        """Resets the environment board to a zero matrix."""
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.is_terminal = False
        return self.get_state()

    def get_state(self):
        """Returns a copy of the current board matrix."""
        return self.board.copy()

    def get_valid_actions(self):
        """An action (column) is valid if the top row slot is empty (0)."""
        if self.is_terminal:
            return []
        return [c for c in range(self.cols) if self.board[0, c] == 0]

    def step(self, action: int):
        """
        Executes a token drop in the specified column.
        Returns: (next_state, reward, is_terminal)
        """
        if action not in self.get_valid_actions():
            raise ValueError(f"Action column {action} is completely full or invalid.")

        # Find the lowest available row in the selected column
        for r in reversed(range(self.rows)):
            if self.board[r, action] == 0:
                self.board[r, action] = self.current_player
                break

        # Check for immediate win conditions
        if self._check_win(self.current_player):
            self.is_terminal = True
            # Reward is 1 for win, -1 for loss (handled from current player's viewpoint)
            return self.get_state(), 1.0, True

        # Check for a draw condition (board full, no valid actions left)
        if len(self.get_valid_actions()) == 0:
            self.is_terminal = True
            return self.get_state(), 0.0, True

        # Flip the player token turn (1 -> -1, -1 -> 1)
        self.current_player = -self.current_player
        return self.get_state(), 0.0, False

    def _check_win(self, player: int) -> bool:
        """Vectorized evaluation window sweeps to catch 4-in-a-row lines."""
        # 1. Horizontal sweep
        for r in range(self.rows):
            for c in range(self.cols - 3):
                if np.all(self.board[r, c:c+4] == player):
                    return True

        # 2. Vertical sweep
        for r in range(self.rows - 3):
            for c in range(self.cols):
                if np.all(self.board[r:r+4, c] == player):
                    return True

        # 3. Positively sloped diagonal sweep (/)
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                if all(self.board[r-i, c+i] == player for i in range(4)):
                    return True

        # 4. Negatively sloped diagonal sweep (\)
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if all(self.board[r+i, c+i] == player for i in range(4)):
                    return True

        return False

    def render(self):
        """Prints a scannable text representation of the board state."""
        symbols = {1: 'X', -1: 'O', 0: '.'}
        print("\n" + "\n".join(" ".join(symbols[val] for val in row) for row in self.board))
        print("-" * (self.cols * 2 - 1))
        print(" ".join(str(i) for i in range(self.cols)))