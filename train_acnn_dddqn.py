"""
ACNN-D3QN (Attention CNN + Dueling Double DQN) 训练脚本

在CNN-D3QN基础上添加CBAM注意力机制：
- Channel Attention: 学习哪个特征通道更重要
- Spatial Attention: 学习哪个空间位置更重要

用法:
    python train_acnn_dddqn.py --episodes 100000 --name acnn_test
    python train_acnn_dddqn.py --episodes 50000 --name test --env v3 --channels 64
"""

import gymnasium as gym
from agents.acnn_dddqn_agent import experienceReplayBuffer, ACNN_D3QNAgent, ACNNDuelingQNetwork
import torch
import argparse
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ACNN-D3QN (Attention CNN + Dueling Double DQN) Training")
    parser.add_argument("--episodes", type=int, default=100000, help="训练episode数量")
    parser.add_argument("--name", type=str, default=None, help="模型保存名称")
    parser.add_argument("--env", type=str, default="v3", choices=["v2", "v3"], 
                        help="环境版本: v2=简单PBRS, v3=策略性PBRS")
    parser.add_argument("--channels", type=int, default=32, help="CNN隐藏通道数")
    parser.add_argument("--features", type=int, default=128, help="特征向量维度")
    parser.add_argument("--buffer", type=int, default=200000, help="经验回放缓冲区大小")
    parser.add_argument("--burnin", type=int, default=20000, help="预填充经验数量")
    parser.add_argument("--batch", type=int, default=256, help="批次大小")
    parser.add_argument("--eval-freq", type=int, default=5000, help="评估频率(episodes)")
    parser.add_argument("--eval-iter", type=int, default=1000, help="每次评估的迭代次数")
    parser.add_argument("--lr", type=float, default=7e-5, help="学习率")
    # 稳定性参数
    parser.add_argument("--tau", type=float, default=0.001, help="软更新系数(越小越稳定)")
    parser.add_argument("--grad-clip", type=float, default=7.0, help="梯度裁剪阈值")
    parser.add_argument("--end-epsilon", type=float, default=0.2, help="最终探索率")
    # CBAM注意力参数
    parser.add_argument("--attn-reduction", type=int, default=4, help="通道注意力缩减比例")
    parser.add_argument("--attn-kernel", type=int, default=3, help="空间注意力卷积核大小")
    parser.add_argument("--no-residual", action="store_true", help="禁用CBAM残差连接")
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
    
    # 创建ACNN Dueling网络
    print(f"\nCreating ACNN-D3QN (Attention CNN + Dueling) network...")
    print(f"  Hidden channels: {args.channels}")
    print(f"  Feature size: {args.features}")
    print(f"  Learning rate: {args.lr}")
    
    network = ACNNDuelingQNetwork(
        env, 
        learning_rate=args.lr,
        device=device,
        hidden_channels=args.channels,
        feature_size=args.features,
        attention_reduction=args.attn_reduction,
        attention_kernel=args.attn_kernel,
        use_residual=not args.no_residual
    )
    
    # 打印网络结构
    total_params = sum(p.numel() for p in network.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Architecture: ACNN Feature Extractor (CNN + CBAM) -> Value Stream + Advantage Stream")
    
    # 创建Agent
    agent = ACNN_D3QNAgent(
        env=env,
        network=network,
        buffer=buffer,
        n_iter=n_iter,
        batch_size=args.batch,
        tau=args.tau,
        grad_clip=args.grad_clip,
        end_epsilon=args.end_epsilon
    )
    
    print(f"  Soft update tau: {args.tau}")
    print(f"  Gradient clipping: {args.grad_clip}")
    print(f"  End epsilon: {args.end_epsilon}")
    
    # 开始训练
    print(f"\nStarting training for {n_iter} episodes...")
    agent.train(
        max_episodes=n_iter,
        evaluate_frequency=args.eval_freq,
        evaluate_n_iter=args.eval_iter
    )
    
    # 保存最终模型
    torch.save(agent.network, str(save_path))
    agent._save_training_data(str(save_path))
    
    print(f"\n Models saved to: {save_dir}")
    print(f"  - Final model: {save_path}")
    print(f"  - Best model:  {save_path}_best")
