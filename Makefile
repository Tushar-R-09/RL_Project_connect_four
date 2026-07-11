.PHONY: run play play-cli train test clean

# Default: Run the unified console launcher
run:
	PYTHONPATH=. uv run python main.py

# Play game visually (Pygame GUI) - RECOMMENDED
play:
	PYTHONPATH=. uv run python -m src.play_gui

# Play game in terminal (CLI)
play-cli:
	PYTHONPATH=. uv run python -m src.play

# Train the DQN agent via self-play
train:
	PYTHONPATH=. uv run python -m src.train

# Run environment verification assertions
test:
	PYTHONPATH=. uv run python -m tests.test_env

# Clean up pycache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
