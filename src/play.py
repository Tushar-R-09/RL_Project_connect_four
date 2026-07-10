"""
play.py: Terminal CLI interface to play live matches against the MCTS+DQN AI engine.
"""
from environment.env import ConnectFourEnv
from mcts import NeuralMCTS

def play_against_ai():
    print("🤖 Connect Four Neural-MCTS AI Engine Initialized!")
    sim_budget = int(input("Enter AI planning simulation budget (e.g., 50 for fast, 200 for smart): "))
    
    env = ConnectFourEnv()
    ai_engine = NeuralMCTS()
    
    state = env.reset()
    print("\n--- Match Commenced! You are X (Player 1), AI is O (Player -1) ---")
    env.render()

    while not env.is_terminal:
        if env.current_player == 1:
            # Human Player Turn
            valid_moves = env.get_valid_actions()
            move = -1
            while move not in valid_moves:
                try:
                    move = int(input(f"Your Turn (Select column {valid_moves}): "))
                except ValueError:
                    print("Please enter a valid column integer.")
            
            env.step(move)
        else:
            # AI Player Turn
            print(f"\n🧠 AI is hallucinating future matches across {sim_budget} branches...")
            ai_move = ai_engine.run_search(env.board, current_player=-1, simulations=sim_budget)
            print(f"🤖 AI chooses Column: {ai_move}")
            env.step(ai_move)
        
        env.render()

    # Match over, determine output parameters
    if env._check_win(1):
        print("\n🏆 Incredible! You defeated the Neural-MCTS engine!")
    elif env._check_win(-1):
        print("\n💀 Checkmate! The AI engine outmaneuvered you.")
    else:
        print("\n🤝 A perfect tie game! Nash Equilibrium achieved.")

if __name__ == "__main__":
    play_against_ai()