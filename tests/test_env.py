"""test_env.py: Verification script to ensure perfect environment mechanics."""
from src.environment.env import ConnectFourEnv

def run_test_game():
    env = ConnectFourEnv()
    print("Empty Environment Board:")
    env.render()

    # Simulate basic moves down a vertical column for Player 1 (X)
    print("\n--- Playing 4 sequential moves in Column 0 ---")
    states_and_rewards = [env.step(0) for _ in range(4)]
    
    env.render()
    final_reward = states_and_rewards[-1][1]
    final_terminal_state = states_and_rewards[-1][2]
    
    print(f"\nTerminal status reached: {final_terminal_state}")
    print(f"Final Step Reward Observed: {final_reward}")

if __name__ == "__main__":
    run_test_game()