# 🎮 Connect Four: Reinforcement Learning & MCTS Playground

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12%2B-orange?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Pygame](https://img.shields.io/badge/Pygame-2.6%2B-green?logo=pygame&logoColor=white)](https://www.pygame.org/)
[![UV](https://img.shields.io/badge/Package%20Manager-UV-blueviolet?logo=cargo&logoColor=white)](https://github.com/astral-sh/uv)

A hands-on playground to practically understand the fundamental concepts of **Reinforcement Learning (RL)** and search algorithms.

### 🎯 Purpose of this Project
The aim of this project is to help you learn the **crux of RL** (such as the Bellman equation, Monte Carlo Tree Search (MCTS), self-play, and Double-DQN) by experimenting with a lightweight, simple game.

**No heavy or rented cloud GPUs are required**—the entire training pipeline and interactive visual engine run smoothly on consumer-level laptop CPUs/GPUs.

---

## 🏛️ Project Directory Structure

```text
connect_four_rl/
├── Makefile                # Running shortcuts for development
├── pyproject.toml          # UV project configuration
├── main.py                 # Interactive launcher menu
├── notebooks/
│   └── rough_notebook.ipynb# Exploratory notebook
├── tests/
│   └── test_env.py         # Connect 4 environment logic tests
└── src/
    ├── mcts.py             # Monte-Carlo Tree Search search tree code
    ├── play.py             # Terminal CLI game loop
    ├── play_gui.py         # Visual dark-mode Pygame interface
    ├── train.py            # Double-DQN self-play training script
    ├── agents/
    │   └── agents.py       # Replay memory & DQN policy networks
    └── environment/
        └── env.py          # Custom Connect 4 board step & win checks
```

---

## 🚀 How to Run

You can launch any feature using the **`Makefile`** from the project root:

```bash
make play     # [Recommended] Play game visually against the AI (Pygame GUI)
make run      # Run the interactive console launcher menu
make play-cli # Play game against the AI in the terminal (CLI)
make train    # Run self-play DQN training on your machine
make test     # Execute environment validation tests
make clean    # Clear Python build files & caches
```

*(Note: Under the hood, these make shortcuts use `PYTHONPATH=. uv run ...` to ensure package imports and virtualenv dependencies resolve correctly.)*

If you don't have `make` installed:
```bash
uv run main.py  # Launches the interactive console launcher
```

---

## 💡 Core RL Concepts Implemented Simply

1. **Bellman Equation & DQN**: The AI uses a Deep Q-Network (DQN) to learn how to evaluate any given board state. It uses the standard Bellman equation target to optimize state-action value mappings via self-play:
   $$Q(s, a) \leftarrow r + \gamma \max_{a'} Q(s', a')$$
2. **Monte Carlo Tree Search (MCTS)**: MCTS runs forward lookahead simulations of the game. It balances exploration and exploitation using the UCB1 formula:
   $$\text{UCB1} = \frac{V_i}{N_i} + C \times \sqrt{\frac{\ln(N_{\text{parent}})}{N_i}}$$
3. **Neural-Guided Search**: Instead of playing games completely at random to see who wins (rollouts), the MCTS tree queries our trained DQN network to approximate the quality of non-terminal boards, uniting search and deep learning.
4. **Action Masking**: Prevents the DQN and MCTS from trying to play tokens in columns that are already full.
5. **Perspective Inversion**: Normalizes board states so a single network learns to play both as Player 1 (Red) and Player -1 (Yellow) seamlessly.
