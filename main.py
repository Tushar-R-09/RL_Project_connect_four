"""
main.py: Unified launcher console for the Connect Four Neural-MCTS RL system.
Provides command-line redirection to visual play, terminal play, training, or environment tests.
"""
import sys

def main():
    print("=" * 60)
    print("      🎮 CONNECT FOUR NEURAL-MCTS & RL ENGINE 🎮      ")
    print("=" * 60)
    print("Select one of the following execution modes:")
    print("  [1] Play Game Visually (Pygame GUI)  <-- RECOMMENDED")
    print("  [2] Play Game in Terminal (CLI)")
    print("  [3] Run Self-Play Agent Training (DQN learning)")
    print("  [4] Execute Environment Verification Tests")
    print("-" * 60)
    
    choice = input("Enter choice (1-4, default=1): ").strip()
    if not choice:
        choice = "1"
        
    if choice == "1":
        # Launch Pygame GUI
        try:
            budget_str = input("Enter AI planning simulation budget (e.g. 50=Fast, 150=Smart, default=100): ").strip()
            budget = int(budget_str) if budget_str else 100
        except ValueError:
            print("Invalid budget, defaulting to 100.")
            budget = 100
            
        print("\n🚀 Starting Pygame GUI context...")
        from src.play_gui import play_gui_main
        play_gui_main(sim_budget=budget)
        
    elif choice == "2":
        # Launch Terminal CLI
        print("\n🚀 Initializing Terminal Game...")
        from src.play import play_against_ai
        play_against_ai()
        
    elif choice == "3":
        # Run Training Loop
        try:
            episodes_str = input("Enter number of self-play episodes to train (default=1000): ").strip()
            episodes = int(episodes_str) if episodes_str else 1000
            
            freq_str = input("Enter target network update frequency (default=10): ").strip()
            freq = int(freq_str) if freq_str else 10
        except ValueError:
            print("Invalid inputs, defaulting to 1000 episodes and freq=10.")
            episodes = 1000
            freq = 10
            
        print(f"\n🚀 Launching training orchestration: {episodes} episodes...")
        from src.train import train_self_play
        train_self_play(episodes=episodes, target_update_freq=freq)
        
    elif choice == "4":
        # Execute Tests
        print("\n🚀 Executing environment unit assertions...")
        from tests.test_env import run_test_game
        run_test_game()
        print("\n✅ Verification complete!")
        
    else:
        print("❌ Invalid selection. Program exiting.")

if __name__ == "__main__":
    main()
