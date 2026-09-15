import gymnasium as gym 
from .model import QNetwork 
from .train import train_agent
from .visualize import plot_training

def main():
    # Training hyperparameters
    learning_rate = 0.01
    n_episodes = 100_000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1

    # RecordEpisodeStatistics stores one return and length per completed episode.
    env = gym.make("Blackjack-v1", sab=False)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    agent = QNetwork(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    train_agent(env=env, episodes=n_episodes, agent=agent)
    plot_training(
        episode_returns=env.return_queue,
        td_errors=agent.training_error,
        output_path="training_progress.png",
    )
    env.close()


if __name__ == "__main__":
    main()


