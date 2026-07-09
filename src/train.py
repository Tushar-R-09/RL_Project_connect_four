"""
train.py: Self-play orchestration routine for DQN Connect Four training.
"""

from environment.env import ConnectFourEnv
from agents.agents import DQNAgent


def train_self_play(episodes=1000, target_update_freq=10):
    env = ConnectFourEnv()
    agent = DQNAgent()

    # Epsilon decay configuration to move from exploration to exploitation smoothly
    epsilon = 0.5
    min_epsilon = 0.05
    decay_rate = 0.995

    print("🚀 Initiating Self-Play Training Loop...")

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False

        # Track history of execution steps within the current match
        episode_history = []

        while not done:
            player = env.current_player
            valid_actions = env.get_valid_actions()

            action = agent.select_action(state, valid_actions, player, epsilon)
            assert action is not None, "Action cannot be None during active play"
            next_state, reward, done = env.step(action)

            # Record current step transitions
            episode_history.append((state, action, reward, next_state, done, player))
            state = next_state

        # --- Post-Game Backpropagation Attribution Process ---
        # Since it's zero-sum, the final player action caused the win/loss outcome.
        # We process transitions sequentially and assign appropriate reward outcomes.
        final_winner = episode_history[-1][5]
        is_draw = episode_history[-1][2] == 0.0

        for state, action, reward, next_state, done, player in episode_history:
            if is_draw:
                step_reward = 0.0
            else:
                # If this player was the final winner, assign +1.0, else assign -1.0
                step_reward = 1.0 if player == final_winner else -1.0

            agent.store_transition(state, action, step_reward, next_state, done, player)

        # Optimize the network parameters
        loss = agent.train_step()

        # Decay exploration curiosity over time
        epsilon = max(min_epsilon, epsilon * decay_rate)

        # Periodic Target network synchronization
        if ep % target_update_freq == 0:
            agent.update_target_network()

        # Print performance check-ins
        if ep % 100 == 0:
            loss_str = f"{loss:.4f}" if loss else "Not enough data"
            print(
                f"Episode {ep:4d}/{episodes} | Epsilon: {epsilon:.3f} | Last Loss: {loss_str}"
            )

    print("🏁 Training complete! Saving network parameters...")
    import torch

    torch.save(agent.policy_net.state_dict(), "connect4_dqn.pt")
    print("Model metrics saved securely to 'connect4_dqn.pt'")


if __name__ == "__main__":
    train_self_play()
