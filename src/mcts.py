"""
mcts.py: Monte-Carlo Tree Search Engine integrated with PyTorch DQN value evaluations.
Uses the UCB1 equation to select search branches dynamically.
"""
import os
import torch
import numpy as np
import copy
from src.environment.env import ConnectFourEnv
from src.agents.agents import ConnectFourNet

class MCTSNode:
    def __init__(self, board, parent=None, action=None, player_turn=1):
        self.board = board
        self.parent = parent
        self.action = action  # The action column that led to this node
        self.player_turn = player_turn
        
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


class NeuralMCTS:
    def __init__(self, model_path=None, exploration_constant=1.414):
        self.c = exploration_constant
        
        if model_path is None:
            # Resolve relative to the project root directory
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "connect4_dqn.pt")
            
        # Load the saved brain parameters from Stage 2
        self.policy_net = ConnectFourNet()
        self.policy_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.policy_net.eval()

    def run_search(self, root_board, current_player, simulations=100):
        """Runs selective forward lookahead simulations from the current root state."""
        root_node = MCTSNode(root_board.copy(), player_turn=current_player)

        for _ in range(simulations):
            node = root_node
            scratch_env = ConnectFourEnv()
            scratch_env.board = node.board.copy()
            scratch_env.current_player = node.player_turn

            # 1. SELECTION PHASE (Lecture 8 & 9: Navigating using UCB1)
            while len(node.children) > 0 and not scratch_env.is_terminal:
                next_node = self._select_ucb_child(node)
                if next_node is None:
                    break
                node = next_node
                scratch_env.step(node.action)

            # 2. EXPANSION PHASE
            if not scratch_env.is_terminal:
                valid_actions = scratch_env.get_valid_actions()
                for act in valid_actions:
                    # Fictional clone prediction step inside our mind matrix
                    sim_env = copy.deepcopy(scratch_env)
                    next_st, _, terminal = sim_env.step(act)
                    
                    child_node = MCTSNode(
                        next_st, 
                        parent=node, 
                        action=act, 
                        player_turn=sim_env.current_player
                    )
                    node.children[act] = child_node
                
                # Step into the first expanded child node
                if len(node.children) > 0:
                    random_act = list(node.children.keys())[0]
                    node = node.children[random_act]
                    scratch_env.step(node.action)

            # 3. EVALUATION PHASE (Lecture 6 & 10: Deep Function Approximation instead of random rollouts)
            if scratch_env.is_terminal:
                # If terminal, get exact rule reward from environment
                if scratch_env._check_win(-scratch_env.current_player):
                    estimated_value = 1.0  # Last player to move won
                else:
                    estimated_value = 0.0  # Draw
            else:
                # Use the Neural Network to estimate position quality from acting player perspective
                norm_board = scratch_env.board.flatten() * scratch_env.current_player
                state_t = torch.FloatTensor(norm_board).unsqueeze(0)
                with torch.no_grad():
                    q_outputs = self.policy_net(state_t).squeeze(0).numpy()
                
                # Approximate value is the maximum potential action state score available
                valid_acts = scratch_env.get_valid_actions()
                raw_max = np.max(q_outputs[valid_acts])
                if np.isnan(raw_max):
                    estimated_value = 0.0
                else:
                    estimated_value = float(raw_max)

            # 4. BACKPROPAGATION PHASE (Lecture 4 & 8: Walk backwards updating stats)
            # The value alternates signs layer-by-layer because players oppose each other
            val = estimated_value
            while node is not None:
                node.visit_count += 1
                node.total_value += val
                val = -val  # Flip value perspective for parent level
                node = node.parent

        # Return the best real-world column action (the one with the highest count)
        best_action = max(root_node.children.items(), key=lambda item: item[1].visit_count)[0]
        return best_action

    def _select_ucb_child(self, node):
        """Calculates the UCB1 mathematical balance equation for selection (Lecture 9)."""
        best_score = -np.inf
        best_child = None

        for action, child in node.children.items():
            if child.visit_count == 0:
                # Prioritize unvisited nodes to ensure basic verification coverage
                exploration_term = np.inf
            else:
                # Complete UCB formula: exploitation + c * sqrt(ln(N_total) / N_child)
                exploration_term = self.c * np.sqrt(np.log(node.visit_count) / child.visit_count)

            # Score from parent's perspective: minimize opponent's utility or maximize yours
            ucb_score = child.q_value + exploration_term
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child

        # Fallback to the first child if best_child is None (e.g. if scores are NaN)
        if best_child is None and len(node.children) > 0:
            best_child = list(node.children.values())[0]

        return best_child