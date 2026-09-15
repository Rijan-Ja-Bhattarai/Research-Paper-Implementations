import gymnasium as gym 
from .model import QNetwork
from tqdm import tqdm 

def train_agent(
        env: gym.Env,
        episodes: int,
        agent: QNetwork
):
    for _ in tqdm(range(episodes)):
        # Start a new hand 
        obs, info = env.reset()
        done = False 

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action=action)
            agent.update(obs, action, reward, terminated, next_obs)
            done = terminated or truncated

        agent.decay_epsilon()
