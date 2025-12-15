import gymnasium as gym
import numpy as np
import cupy as cp
import torch
from agents.actor_critic_agent_v3 import PPOAgent, Trainer
from pvz import config
import matplotlib.pyplot as plt

HP_NORM = 100
SUN_NORM = 200

def transform_observation_batch(observations, grid_size):
    # observations: (N, D) numpy array
    observations = cp.asarray(observations, dtype=cp.float32)
    
    # Grid part: [:, :grid_size]
    # Zombie grid part: [:, grid_size:2*grid_size]
    zombie_grid = observations[:, grid_size:2*grid_size]
    
    # Reshape zombie grid to (N, N_LANES, LANE_LENGTH)
    zombie_grid = zombie_grid.reshape((-1, config.N_LANES, config.LANE_LENGTH))
    # Sum over lane length: (N, N_LANES)
    zombie_lanes = cp.sum(zombie_grid, axis=2) / HP_NORM
    
    # Sun value: [:, 2*grid_size] -> (N, 1)
    sun_val = (observations[:, 2*grid_size] / SUN_NORM).reshape(-1, 1)
    
    # Rest: [:, 2*grid_size+1:]
    rest = observations[:, 2*grid_size+1:]
    
    # Concatenate
    new_obs = cp.concatenate([
        observations[:, :grid_size],
        zombie_lanes,
        sun_val,
        rest
    ], axis=1)
    
    return new_obs

def make_env():
    env = gym.make('gym_pvz:pvz-env-v2')
    return env

def train_vec(num_envs=32, n_iter=1000, n_steps=1024):
    print(f"Training with {num_envs} environments...")
    # Create vector env
    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(num_envs)])
    
    # Get dimensions
    dummy_env = Trainer(render=False)
    input_size = dummy_env.num_observations()
    possible_actions = list(range(dummy_env.num_actions()))
    grid_size = dummy_env._grid_size
    
    agent = PPOAgent(
        input_size=input_size,
        possible_actions=possible_actions,
        mini_batch_size=256
    )
    
    obs, _ = envs.reset()
    obs = transform_observation_batch(obs, grid_size)
    
    score_history = []
    loss_history = []
    
    # Track scores
    episode_scores = np.zeros(num_envs)
    
    for iteration in range(n_iter):
        for step in range(n_steps):
            action = agent.decide_action(obs)
            
            next_obs, rewards, terminations, truncations, infos = envs.step(action)
            
            # Update scores
            episode_scores += rewards
            
            # Handle done
            dones = terminations | truncations
            
            # If any env is done, print score and reset score
            for i, done in enumerate(dones):
                if done:
                    # print(f"Env {i} done. Score: {episode_scores[i]}")
                    score_history.append(episode_scores[i])
                    episode_scores[i] = 0
            
            agent.store_reward_done(rewards, dones)
            
            next_obs = transform_observation_batch(next_obs, grid_size)
            obs = next_obs
            
        # Update
        loss = agent.update(obs)
        loss_history.append(loss)
        
        avg_score = np.mean(score_history[-100:]) if score_history else 0
        print(f"Iteration {iteration}, Loss: {loss:.4f}, Avg Score (last 100): {avg_score:.2f}")
        
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(score_history)
    plt.title("Scores")
    plt.savefig("training_vec_plot.png")
    plt.close()
    
    agent.save("ppo_vec_agent_model.pth")

    envs.close()

if __name__ == "__main__":
    train_vec()
