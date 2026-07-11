"""
play_gui.py: A modern, high-fidelity Pygame GUI for Connect Four.
Features dark-mode aesthetics, smooth gravitational coin-drop animations,
multi-threaded MCTS search to maintain 60fps responsiveness, hover column previews,
and custom neon overlays for winning matches.
"""
import sys
import os
import math
import threading
import pygame
import numpy as np

from src.environment.env import ConnectFourEnv
from src.mcts import NeuralMCTS

# Initialize pygame modules
pygame.init()
pygame.font.init()

# --- DESIGN & AESTHETIC CONFIGURATION ---
WINDOW_WIDTH = 700
WINDOW_HEIGHT = 800
BOARD_ROWS = 6
BOARD_COLS = 7
CELL_SIZE = 100
Y_OFFSET = 150  # Room for top status bar

# Frame rate controller
FPS = 60
clock = pygame.time.Clock()

# Color Palette (HSL curated RGB conversions)
COLOR_BG = (15, 15, 18)          # Sleek Deep Obsidian
COLOR_PANEL = (24, 24, 30)       # Status bar panel background
COLOR_BOARD = (30, 41, 59)       # Modern Slate blue/gray
COLOR_SLOT_EMPTY = (15, 15, 18)  # Empty board slot (matches background)

# Neon Player Colors
COLOR_PLAYER_HUMAN = (239, 68, 68)  # Vibrant Neon Red
COLOR_PLAYER_AI = (245, 158, 11)   # Glowing Amber Gold
COLOR_WIN_LINE = (34, 197, 94)     # Emerald Green Highlight

# Fonts
FONT_TITLE = pygame.font.SysFont("Inter", 38, bold=True)
FONT_SUBTITLE = pygame.font.SysFont("Inter", 24, bold=False)
FONT_BUTTON = pygame.font.SysFont("Inter", 20, bold=True)


class ConnectFourGUI:
    def __init__(self, sim_budget=100):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Connect Four: Neural-MCTS Engine")
        
        self.env = ConnectFourEnv()
        self.ai_engine = NeuralMCTS()
        self.sim_budget = sim_budget
        
        # State variables
        self.hover_col = -1
        self.board_state = self.env.board.copy()
        self.winner_coords = None
        self.game_over = False
        
        # Animation states
        self.animating = False
        self.animation_token = None  # Dict storing active drop metrics
        
        # Threading states for AI execution
        self.ai_thinking = False
        self.ai_recommended_move = None

    def reset_game(self):
        self.env.reset()
        self.board_state = self.env.board.copy()
        self.winner_coords = None
        self.game_over = False
        self.animating = False
        self.animation_token = None
        self.ai_thinking = False
        self.ai_recommended_move = None
        self.hover_col = -1

    def run(self):
        running = True
        while running:
            clock.tick(FPS)
            
            # 1. Update AI actions in non-blocking thread
            self._update_ai_logic()
            
            # 2. Process system events
            running = self._handle_events()
            
            # 3. Update active drop animations
            self._update_animations()
            
            # 4. Render graphics onto frame buffer
            self._render()
            
        pygame.quit()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if self.game_over:
                # Handle click on 'Play Again' button
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    btn_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, 725, 200, 45)
                    if btn_rect.collidepoint(mx, my):
                        self.reset_game()
                continue

            if event.type == pygame.MOUSEMOTION:
                mx, my = pygame.mouse.get_pos()
                if Y_OFFSET <= my < WINDOW_HEIGHT - 50:
                    self.hover_col = mx // CELL_SIZE
                else:
                    self.hover_col = -1

            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.env.current_player == 1 and not self.animating and not self.ai_thinking:
                    mx, my = pygame.mouse.get_pos()
                    col = mx // CELL_SIZE
                    if 0 <= col < BOARD_COLS and col in self.env.get_valid_actions():
                        self._initiate_token_drop(col, player=1)
                        
        return True

    def _initiate_token_drop(self, col, player):
        """Calculates final row coordinates and kicks off gravitational drop animation."""
        self.animating = True
        
        # Find the lowest available row in the selected column
        target_row = -1
        for r in reversed(range(BOARD_ROWS)):
            if self.board_state[r, col] == 0:
                target_row = r
                break
        
        # Start dropping from just above the board surface
        start_y = Y_OFFSET - 50
        target_y = Y_OFFSET + target_row * CELL_SIZE + CELL_SIZE // 2
        
        self.animation_token = {
            "col": col,
            "row": target_row,
            "current_y": float(start_y),
            "target_y": float(target_y),
            "velocity": 0.0,
            "player": player
        }

    def _update_animations(self):
        """Updates position of falling coin using realistic gravity simulation."""
        if not self.animating or not self.animation_token:
            return
            
        tok = self.animation_token
        gravity = 1.6
        bounce_attenuation = -0.35
        
        tok["velocity"] += gravity
        tok["current_y"] += tok["velocity"]
        
        # Collision detection with slot floor
        if tok["current_y"] >= tok["target_y"]:
            tok["current_y"] = tok["target_y"]
            
            # If speed is high enough, trigger a subtle rebound bounce
            if tok["velocity"] > 4.0:
                tok["velocity"] *= bounce_attenuation
            else:
                # Animation finished, write to environment
                self.animating = False
                _, reward, done = self.env.step(tok["col"])
                self.board_state = self.env.board.copy()
                
                # Check for win conditions
                if self.env._check_win(tok["player"]):
                    self.winner_coords = self._find_winning_line(tok["player"])
                    self.game_over = True
                elif done:  # Draw game
                    self.game_over = True
                    
                self.animation_token = None

    def _update_ai_logic(self):
        """Orchestrates non-blocking background thread for Neural-MCTS decision making."""
        if self.env.current_player == -1 and not self.animating and not self.game_over:
            if not self.ai_thinking:
                self.ai_thinking = True
                thread = threading.Thread(target=self._run_mcts_search, daemon=True)
                thread.start()
                
            elif self.ai_recommended_move is not None:
                move = self.ai_recommended_move
                self.ai_recommended_move = None
                self.ai_thinking = False
                self._initiate_token_drop(move, player=-1)

    def _run_mcts_search(self):
        """Executed inside a background thread to prevent GUI lagging."""
        move = self.ai_engine.run_search(
            self.env.board, 
            current_player=-1, 
            simulations=self.sim_budget
        )
        self.ai_recommended_move = move

    def _find_winning_line(self, player):
        """Returns the list of 4 coordinates that formed the winning connection."""
        # 1. Horizontal sweep
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS - 3):
                if np.all(self.board_state[r, c:c+4] == player):
                    return [(r, c+i) for i in range(4)]
                    
        # 2. Vertical sweep
        for r in range(BOARD_ROWS - 3):
            for c in range(BOARD_COLS):
                if np.all(self.board_state[r:r+4, c] == player):
                    return [(r+i, c) for i in range(4)]
                    
        # 3. Positive Diagonal (/)
        for r in range(3, BOARD_ROWS):
            for c in range(BOARD_COLS - 3):
                if all(self.board_state[r-i, c+i] == player for i in range(4)):
                    return [(r-i, c+i) for i in range(4)]
                    
        # 4. Negative Diagonal (\)
        for r in range(BOARD_ROWS - 3):
            for c in range(BOARD_COLS - 3):
                if all(self.board_state[r+i, c+i] == player for i in range(4)):
                    return [(r+i, c+i) for i in range(4)]
        return None

    def _render(self):
        self.screen.fill(COLOR_BG)
        
        # 1. Render upper status bar panel
        self._render_status_panel()
        
        # 2. Render tokens currently sitting on the board
        self._render_static_tokens()
        
        # 3. Render any token in drop animation
        self._render_animated_token()
        
        # 4. Draw Column Hover Selector Preview
        self._render_hover_preview()
        
        # 5. Render the front board structure (overlay with transparent circular mask holes)
        self._render_board_grid()
        
        # 6. If match is over, draw winning connection line and results overlay
        self._render_endgame_overlay()
        
        pygame.display.flip()

    def _render_status_panel(self):
        panel_rect = pygame.Rect(0, 0, WINDOW_WIDTH, Y_OFFSET)
        pygame.draw.rect(self.screen, COLOR_PANEL, panel_rect)
        
        # Draw soft boundary separator line
        pygame.draw.line(self.screen, (38, 38, 48), (0, Y_OFFSET), (WINDOW_WIDTH, Y_OFFSET), 2)
        
        title_text = FONT_TITLE.render("Neural-MCTS Connect 4", True, (255, 255, 255))
        self.screen.blit(title_text, (20, 20))
        
        # Subtitle state information
        if self.game_over:
            if self.winner_coords is not None:
                winner = "Human" if self.board_state[self.winner_coords[0]] == 1 else "AI"
                status_color = COLOR_PLAYER_HUMAN if winner == "Human" else COLOR_PLAYER_AI
                status_str = f"GAME OVER: {winner} wins matches!"
            else:
                status_str = "GAME OVER: Nash Equilibrium reached (Draw)."
                status_color = (150, 150, 160)
        elif self.ai_thinking:
            # Simple pulsing animation based on system clock
            pulsing_dots = "." * (int(pygame.time.get_ticks() / 300) % 4)
            status_str = f"AI is planning outcomes via MCTS ({self.sim_budget} simulations){pulsing_dots}"
            status_color = COLOR_PLAYER_AI
        elif self.env.current_player == 1:
            status_str = "Your Turn: Click column to drop piece"
            status_color = COLOR_PLAYER_HUMAN
        else:
            status_str = "AI Engine is choosing next action..."
            status_color = COLOR_PLAYER_AI
            
        subtitle_text = FONT_SUBTITLE.render(status_str, True, status_color)
        self.screen.blit(subtitle_text, (20, 75))

    def _render_static_tokens(self):
        """Draws player tokens behind the front board grid."""
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                val = self.board_state[r, c]
                if val != 0:
                    color = COLOR_PLAYER_HUMAN if val == 1 else COLOR_PLAYER_AI
                    x = c * CELL_SIZE + CELL_SIZE // 2
                    y = Y_OFFSET + r * CELL_SIZE + CELL_SIZE // 2
                    
                    # Draw solid coin
                    pygame.draw.circle(self.screen, color, (x, y), CELL_SIZE // 2 - 8)
                    # Add highlight for 3D depth effect
                    pygame.draw.circle(self.screen, (255, 255, 255), (x - 8, y - 8), 6)

    def _render_animated_token(self):
        if self.animating and self.animation_token:
            tok = self.animation_token
            x = tok["col"] * CELL_SIZE + CELL_SIZE // 2
            y = int(tok["current_y"])
            color = COLOR_PLAYER_HUMAN if tok["player"] == 1 else COLOR_PLAYER_AI
            pygame.draw.circle(self.screen, color, (x, y), CELL_SIZE // 2 - 8)
            pygame.draw.circle(self.screen, (255, 255, 255), (x - 8, y - 8), 6)

    def _render_hover_preview(self):
        """Draws a semi-transparent preview circle showing where the piece will drop."""
        if self.hover_col != -1 and not self.animating and not self.game_over and not self.ai_thinking:
            valid_actions = self.env.get_valid_actions()
            if self.hover_col in valid_actions:
                # Find resting row
                resting_row = -1
                for r in reversed(range(BOARD_ROWS)):
                    if self.board_state[r, self.hover_col] == 0:
                        resting_row = r
                        break
                        
                if resting_row != -1:
                    x = self.hover_col * CELL_SIZE + CELL_SIZE // 2
                    y = Y_OFFSET + resting_row * CELL_SIZE + CELL_SIZE // 2
                    color = (239, 68, 68, 80) if self.env.current_player == 1 else (245, 158, 11, 80)
                    
                    # Draw a transparent preview circle using a temporary surface
                    temp_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surface, color, (CELL_SIZE // 2, CELL_SIZE // 2), CELL_SIZE // 2 - 8)
                    self.screen.blit(temp_surface, (self.hover_col * CELL_SIZE, Y_OFFSET + resting_row * CELL_SIZE))

    def _render_board_grid(self):
        """
        Draws the solid board frame over the top of the circles,
        masking them with empty background-colored circular slot cutouts.
        """
        board_surface = pygame.Surface((WINDOW_WIDTH, BOARD_ROWS * CELL_SIZE), pygame.SRCALPHA)
        board_surface.fill(COLOR_BOARD)
        
        # Carve out circular windows in the solid slate board surface
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                cx = c * CELL_SIZE + CELL_SIZE // 2
                cy = r * CELL_SIZE + CELL_SIZE // 2
                # Alpha 0 represents complete transparency (carves out slot)
                pygame.draw.circle(board_surface, (0, 0, 0, 0), (cx, cy), CELL_SIZE // 2 - 8)
                # Outer shadow ring for slots
                pygame.draw.circle(board_surface, (18, 18, 22, 180), (cx, cy), CELL_SIZE // 2 - 6, 2)
                
        self.screen.blit(board_surface, (0, Y_OFFSET))

    def _render_endgame_overlay(self):
        if not self.game_over:
            return
            
        # Draw glowing win connection line
        if self.winner_coords:
            pts = []
            for r, c in self.winner_coords:
                x = c * CELL_SIZE + CELL_SIZE // 2
                y = Y_OFFSET + r * CELL_SIZE + CELL_SIZE // 2
                pts.append((x, y))
            pygame.draw.line(self.screen, COLOR_WIN_LINE, pts[0], pts[-1], 10)
            for pt in pts:
                pygame.draw.circle(self.screen, (255, 255, 255), pt, 8)
                
        # Draw translucent shadow overlay across board
        shade_surface = pygame.Surface((WINDOW_WIDTH, BOARD_ROWS * CELL_SIZE), pygame.SRCALPHA)
        shade_surface.fill((10, 10, 12, 160))
        self.screen.blit(shade_surface, (0, Y_OFFSET))
        
        # Banner overlay for text
        banner_rect = pygame.Rect(0, WINDOW_HEIGHT // 2 - 70, WINDOW_WIDTH, 140)
        pygame.draw.rect(self.screen, COLOR_PANEL, banner_rect)
        pygame.draw.line(self.screen, (34, 197, 94) if self.winner_coords else (150, 150, 160), (0, banner_rect.top), (WINDOW_WIDTH, banner_rect.top), 3)
        pygame.draw.line(self.screen, (34, 197, 94) if self.winner_coords else (150, 150, 160), (0, banner_rect.bottom), (WINDOW_WIDTH, banner_rect.bottom), 3)
        
        # Build text string
        if self.winner_coords:
            winner = "HUMAN" if self.board_state[self.winner_coords[0]] == 1 else "AI ENGINE"
            main_text = f"🏆 {winner} VICTORY!" if winner == "HUMAN" else "💀 AI ENGINE OUTPLAYED YOU!"
            sub_text = "AI defeated your logic. Try again!" if winner != "HUMAN" else "Outstanding! You outmaneuvered Neural-MCTS!"
        else:
            main_text = "🤝 PERFECT DRAW MATCH!"
            sub_text = "Full board state achieved. No legal operations remaining."
            
        render_main = FONT_TITLE.render(main_text, True, (255, 255, 255))
        render_sub = FONT_SUBTITLE.render(sub_text, True, (156, 163, 175))
        
        self.screen.blit(render_main, (WINDOW_WIDTH // 2 - render_main.get_width() // 2, WINDOW_HEIGHT // 2 - 50))
        self.screen.blit(render_sub, (WINDOW_WIDTH // 2 - render_sub.get_width() // 2, WINDOW_HEIGHT // 2 + 5))
        
        # Interactive Play Again Button at the bottom
        btn_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, 725, 200, 45)
        mx, my = pygame.mouse.get_pos()
        
        # Change color on hover
        btn_color = (37, 99, 235) if btn_rect.collidepoint(mx, my) else (29, 78, 216)
        pygame.draw.rect(self.screen, btn_color, btn_rect, border_radius=8)
        
        btn_text = FONT_BUTTON.render("Play Again", True, (255, 255, 255))
        self.screen.blit(btn_text, (btn_rect.centerx - btn_text.get_width() // 2, btn_rect.centery - btn_text.get_height() // 2))


def play_gui_main(sim_budget=100):
    gui = ConnectFourGUI(sim_budget=sim_budget)
    gui.run()


if __name__ == "__main__":
    play_gui_main(sim_budget=100)
