import gymnasium as gym
from agents.dqn_agent import experienceReplayBuffer_DQN, DQNAgent, QNetwork_DQN
import torch
import argparse
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100000, help="训练episode数量")
    parser.add_argument("--name", type=str, default=None, help="模型保存名称")
    args = parser.parse_args()
    
    n_iter = args.episodes
    env = gym.make('gym_pvz:pvz-env-v2')
    
    # 支持命令行指定名称
    if args.name:
        nn_name = args.name
    else:
        nn_name = input("Save name: ")
    
    # 创建保存目录
    save_dir = Path("agents/agent_zoo") / nn_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / nn_name
    
    buffer = experienceReplayBuffer_DQN(memory_size=100000, burn_in=10000)
    net = QNetwork_DQN(env, device='cpu', use_zombienet=False, use_gridnet=False)
    # old_agent = torch.load("agents/benchmark/dfq5_znet_epslinear")
    # net.zombienet.load_state_dict(old_agent.zombienet.state_dict())
    # for p in net.zombienet.parameters():
    #     p.requires_grad = False
    # net.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
    #                                       lr=net.learning_rate)
    agent = DQNAgent(env, net, buffer, n_iter=n_iter, batch_size=200)
    agent.train(max_episodes=n_iter, evaluate_frequency=5000, evaluate_n_iter=1000)
    
    torch.save(agent.network, str(save_path))
    agent._save_training_data(str(save_path))
    print(f"\nModel saved to: {save_dir}")
    print(f"  - Model: {save_path}")