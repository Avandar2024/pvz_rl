"""
Dueling Double DQN (D3QN/DDDQN) 训练脚本

结合了:
1. Double DQN: 减少Q值过估计
2. Dueling DQN: V(s) + A(s,a) 网络架构

用法:
    python train_dddqn_agent.py --episodes 100000 --name my_model
    python train_dddqn_agent.py --episodes 50000 --name test --env v2
"""

import gymnasium as gym
from agents.dddqn_agent import experienceReplayBuffer, D3QNAgent, DuelingQNetwork
import torch
import argparse
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dueling Double DQN Training")
    parser.add_argument("--episodes", type=int, default=100000, help="训练episode数量")
    parser.add_argument("--name", type=str, default=None, help="模型保存名称")
    parser.add_argument("--env", type=str, default="v3", choices=["v2", "v3"], 
                        help="环境版本: v2=简单PBRS, v3=策略性PBRS")
    parser.add_argument("--hidden", type=int, default=64, help="隐藏层大小")
    parser.add_argument("--buffer", type=int, default=100000, help="经验回放缓冲区大小")
    parser.add_argument("--burnin", type=int, default=10000, help="预填充经验数量")
    parser.add_argument("--batch", type=int, default=64, help="批次大小")
    parser.add_argument("--eval-freq", type=int, default=5000, help="评估频率(episodes)")
    parser.add_argument("--eval-iter", type=int, default=1000, help="每次评估的迭代次数")
    args = parser.parse_args()
    
    n_iter = args.episodes
    
    # 选择环境版本
    env_name = f'gym_pvz:pvz-env-{args.env}'
    print(f"Using environment: {env_name}")
    env = gym.make(env_name)
    
    # 模型名称
    if args.name:
        nn_name = args.name
    else:
        nn_name = input("Save name: ")
    
    # 创建保存目录
    save_dir = Path("agents/agent_zoo") / nn_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / nn_name
    
    # 创建经验回放缓冲区
    buffer = experienceReplayBuffer(memory_size=args.buffer, burn_in=args.burnin)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device} (torch={torch.__version__}, cuda_available={torch.cuda.is_available()})")
    
    # 创建Dueling DQN网络
    net = DuelingQNetwork(
        env, 
        device=device, 
        use_zombienet=False, 
        use_gridnet=False,
        hidden_size=args.hidden
    )
    net.to(device)  # 确保网络在正确的设备上
    
    # 打印网络结构
    print(f"\n=== Dueling DQN Network Architecture ===")
    print(f"Input size: {net.n_inputs}")
    print(f"Hidden size: {args.hidden}")
    print(f"Output size (actions): {net.n_outputs}")
    print(f"Total parameters: {sum(p.numel() for p in net.parameters())}")
    print(f"=========================================\n")
    
    # 创建D3QN Agent
    agent = D3QNAgent(
        env, 
        net, 
        buffer, 
        n_iter=n_iter, 
        batch_size=args.batch
    )
    
    # 开始训练
    print(f"Starting D3QN training for {n_iter} episodes...")
    agent.train(
        max_episodes=n_iter, 
        evaluate_frequency=args.eval_freq, 
        evaluate_n_iter=args.eval_iter
    )
    
    # 保存模型和训练数据
    torch.save(agent.network, str(save_path))
    agent._save_training_data(str(save_path))
    print(f"\nModel saved to: {save_dir}")
    print(f"  - Model: {save_path}")
    print(f"  - Training data: {save_path}_rewards.npy, {save_path}_iterations.npy, etc.")
