import gymnasium as gym
import numpy as np
from pathlib import Path
from agents.ac_mlx import PPOAgent, Trainer
from pvz import config
import matplotlib.pyplot as plt

HP_NORM = 100
SUN_NORM = 200

def make_env():
    env = gym.make('gym_pvz:pvz-env-v3')
    return env

def train_vec(num_envs=32, n_iter=5000, n_steps=1024, checkpoint_interval=50):
    print(f"Training with {num_envs} environments using MLX...")
    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(num_envs)])
    
    model_name = "ppo_vec_agent_mlx"
    save_dir = Path("agents/agent_zoo") / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = save_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    dummy_env = Trainer(render=False)
    num_actions = dummy_env.num_actions()
    possible_actions = list(range(num_actions))
    
    agent = PPOAgent(
        possible_actions=possible_actions,
        mini_batch_size=512
    )
    
    dummy_env.compile_agent_network(agent)
    dummy_env.close()
    
    obs, _ = envs.reset()
    obs = dummy_env._transform_observation(obs)
    
    score_history = []
    win_history = []
    loss_history = []
    entropy_history = []
    
    episode_scores = np.zeros(num_envs)
    
    for iteration in range(n_iter):
        for step in range(n_steps):
            action = agent.decide_action(obs)
            
            next_obs, rewards, terminations, truncations, infos = envs.step(action)
                        
            episode_scores += rewards
            
            dones = terminations | truncations
            
            for i, done in enumerate(dones):
                if done:
                    score_history.append(episode_scores[i])
                    episode_scores[i] = 0
                    
                    if "is_victory" in infos:
                        is_win = infos["is_victory"][i]
                        win_history.append(1 if is_win else 0)
            
            agent.store_reward_done(rewards, dones)
            
            next_obs = dummy_env._transform_observation(next_obs)
            obs = next_obs
            
        loss, entropy = agent.update(obs)
        loss_history.append(loss)
        entropy_history.append(entropy)
        
        avg_score = np.mean(score_history[-100:]) if score_history else 0
        avg_win_rate = np.mean(win_history[-100:]) if win_history else 0
        print(f"Iteration {iteration}, Loss: {loss:.4f}, Entropy: {entropy:.4f}, Avg Score (last 100): {avg_score:.2f}, Win Rate: {avg_win_rate:.2%}")
        
        if (iteration + 1) % checkpoint_interval == 0:
            checkpoint_path = checkpoints_dir / f"checkpoint_iter_{iteration+1}.safetensors"
            agent.save(str(checkpoint_path))
            print(f"Saved checkpoint to {checkpoint_path}")
        
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.plot(loss_history)
    plt.title("Loss")
    plt.subplot(1, 4, 2)
    plt.plot(score_history, alpha=0.3, label='Raw')
    if len(score_history) >= 100:
        moving_avg = np.convolve(score_history, np.ones(100)/100, mode='valid')
        plt.plot(range(99, len(score_history)), moving_avg, color='red', label='Avg (100)')
    plt.title("Scores")
    plt.legend()
    plt.subplot(1, 4, 3)
    plt.plot(entropy_history)
    plt.title("Entropy")
    plt.subplot(1, 4, 4)
    if len(win_history) >= 100:
        win_rate_avg = np.convolve(win_history, np.ones(100)/100, mode='valid')
        plt.plot(range(99, len(win_history)), win_rate_avg, color='green', label='Win Rate (100)')
    plt.plot(win_history, alpha=0.1, label='Raw Win')
    plt.title("Win Rate")
    plt.legend()
    plot_path = save_dir / "training_plot.png"
    plt.savefig(str(plot_path))
    plt.close()
    
    final_model_path = save_dir / f"{model_name}.safetensors"
    agent.save(str(final_model_path))
    print(f"\nTraining complete!")
    print(f"Models saved to: {save_dir}")
    print(f"  - Final model: {final_model_path}")
    print(f"  - Training plot: {plot_path}")

    envs.close()

if __name__ == "__main__":
    train_vec()
