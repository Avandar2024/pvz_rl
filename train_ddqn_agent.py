import gymnasium as gym
from agents import experienceReplayBuffer, DDQNAgent, QNetwork
import torch
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100000, help="训练episode数量")
    parser.add_argument("--name", type=str, default=None, help="模型保存名称（避免交互式输入）")
    args = parser.parse_args()
    
    n_iter = args.episodes
    # V3环境有更精细的策略奖励，需要更多训练轮数（建议100k+）
    # V2环境更简单直接，25k-50k即可看到效果
    env = gym.make('gym_pvz:pvz-env-v3')
    
    # 支持命令行指定名称，避免input()在某些终端下阻塞
    if args.name:
        nn_name = args.name
    else:
        nn_name = input("Save name: ")
    # Windows 上若出现 0xc000012d（提交内存不足/分页文件不足），优先从降低 replay/batch 开始。
    # burn_in: 预填充一些经验，让早期学习更稳定
    buffer = experienceReplayBuffer(memory_size=50000, burn_in=1000)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device} (torch={torch.__version__}, cuda_available={torch.cuda.is_available()})")
    net = QNetwork(env, device=device, use_zombienet=False, use_gridnet=False)
    # old_agent = torch.load("agents/benchmark/dfq5_znet_epslinear")
    # net.zombienet.load_state_dict(old_agent.zombienet.state_dict())
    # for p in net.zombienet.parameters():
    #     p.requires_grad = False
    # net.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
    #                                       lr=net.learning_rate)
    agent = DDQNAgent(env, net, buffer, n_iter=n_iter, batch_size=64)
    agent.train(max_episodes=n_iter, evaluate_frequency=5000, evaluate_n_iter=1000)
    torch.save(agent.network, nn_name)
    agent._save_training_data(nn_name)
