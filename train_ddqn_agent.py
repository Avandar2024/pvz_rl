import gymnasium as gym
from agents import experienceReplayBuffer, DDQNAgent, QNetwork
import torch
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100000, help="训练episode数量")
    parser.add_argument("--name", type=str, default=None, help="模型保存名称")
    args = parser.parse_args()
    
    n_iter = args.episodes
    # 更精细的策略奖励，需要更多训练轮数（建议100k+）
    # v2环境训练轮数建议取100k
    env = gym.make('gym_pvz:pvz-env-v3')
    
    # 支持命令行指定名称，避免input()在某些终端下阻塞
    if args.name:
        nn_name = args.name
    else:
        nn_name = input("Save name: ")
    buffer = experienceReplayBuffer(memory_size=100000, burn_in=10000)
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
    
    # 保存最终模型
    torch.save(agent.network, nn_name)
    agent._save_training_data(nn_name)
    
    # 保存训练过程中表现最好的模型
    agent.save_best_model(nn_name)
    
    print(f"\n Models saved:")
    print(f"  - Final model: {nn_name}")
    print(f"  - Best model:  {nn_name}_best")
