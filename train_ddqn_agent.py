import gymnasium as gym
from agents import experienceReplayBuffer, DDQNAgent, QNetwork
import torch
import argparse
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100000, help="训练episode数量")
    parser.add_argument("--name", type=str, default=None, help="模型保存名称")
    args = parser.parse_args()
    
    n_iter = args.episodes
    env = gym.make('gym_pvz:pvz-env-v3')
    
    # 支持命令行指定名称，避免input()在某些终端下阻塞
    if args.name:
        nn_name = args.name
    else:
        nn_name = input("Save name: ")
    
    # 创建保存目录
    save_dir = Path("agents/agent_zoo") / nn_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / nn_name
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
    torch.save(agent.network, str(save_path))
    agent._save_training_data(str(save_path))
    
    # 保存训练过程中表现最好的模型
    agent.save_best_model(str(save_path))
    
    print(f"\n Models saved to: {save_dir}")
    print(f"  - Final model: {save_path}")
    print(f"  - Best model:  {save_path}_best")
