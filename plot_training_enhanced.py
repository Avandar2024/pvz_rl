
"""
增强版训练结果可视化脚本
用法: python plot_training_enhanced.py <model_name>
例如: python plot_training_enhanced.py my_model
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch


def plot_training_results(name):
    """绘制训练结果的完整可视化"""
    
    print(f"加载训练数据: {name}")
    
    # 加载数据
    try:
        rewards = np.load(name + "_rewards.npy")
        iterations = np.load(name + "_iterations.npy")
        loss = torch.load(name + "_loss", weights_only=False)
        # 转换为 numpy 数组
        if isinstance(loss, list):
            loss = np.array(loss)
        real_rewards = np.load(name + "_real_rewards.npy")
        real_iterations = np.load(name + "_real_iterations.npy")
    except FileNotFoundError as e:
        print(f"错误: 找不到文件 {e.filename}")
        print(f"请确保以下文件存在:")
        print(f"  - {name}_rewards.npy")
        print(f"  - {name}_iterations.npy")
        print(f"  - {name}_loss")
        print(f"  - {name}_real_rewards.npy")
        print(f"  - {name}_real_iterations.npy")
        return
    
    n_iter = rewards.shape[0]
    n_record = real_rewards.shape[0]
    record_period = n_iter // n_record
    slice_size = 500
    
    # 平滑训练数据
    n_slices = n_iter // slice_size
    rewards_smooth = np.reshape(rewards[:n_slices*slice_size], 
                                (n_slices, slice_size)).mean(axis=1)
    iterations_smooth = np.reshape(iterations[:n_slices*slice_size], 
                                   (n_slices, slice_size)).mean(axis=1)
    loss_smooth = np.reshape(loss[:n_slices*slice_size], 
                            (n_slices, slice_size)).mean(axis=1)
    
    x = np.arange(0, n_slices * slice_size, slice_size)
    xx = np.arange(record_period - 1, n_iter, record_period)[:n_record]
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Results: {name}', fontsize=16, fontweight='bold')
    
    # 1. 奖励曲线
    ax1 = axes[0, 0]
    ax1.plot(x, rewards_smooth, label='Training Rewards (smoothed)', alpha=0.7, linewidth=2)
    ax1.plot(xx, real_rewards, label='Evaluation Rewards', 
             color='red', marker='o', markersize=4, linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Rewards Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 游戏帧数（存活时间）
    ax2 = axes[0, 1]
    ax2.plot(x, iterations_smooth, label='Training Iterations (smoothed)', 
             alpha=0.7, linewidth=2)
    ax2.plot(xx, real_iterations, label='Evaluation Iterations', 
             color='red', marker='o', markersize=4, linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Frames')
    ax2.set_title('Game Duration (Frames)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 损失函数
    ax3 = axes[1, 0]
    ax3.plot(x, loss_smooth, color='green', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Loss (MSE)')
    ax3.grid(True, alpha=0.3)
    
    # 4. 评估奖励趋势（带最佳标记）
    ax4 = axes[1, 1]
    ax4.plot(xx, real_rewards, marker='o', markersize=6, linewidth=2, color='blue')
    best_idx = np.argmax(real_rewards)
    best_score = real_rewards[best_idx]
    best_episode = xx[best_idx]
    ax4.scatter([best_episode], [best_score], color='gold', s=200, 
                marker='*', zorder=5, label=f'Best: {best_score:.1f} @ ep {best_episode}')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Evaluation Reward')
    ax4.set_title('Evaluation Performance Trend')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_file = f"{name}_training_plot.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 图片已保存: {output_file}")
    
    plt.show()
    
    # 打印统计信息
    print("\n" + "="*60)
    print("训练统计:")
    print("="*60)
    print(f"总训练回合数: {n_iter}")
    print(f"评估次数: {n_record}")
    print(f"\n训练奖励:")
    print(f"  最终平均: {rewards[-100:].mean():.2f}")
    print(f"  最高平均: {rewards_smooth.max():.2f}")
    print(f"  最低平均: {rewards_smooth.min():.2f}")
    print(f"\n评估奖励:")
    print(f"  最佳分数: {best_score:.2f} (Episode {best_episode})")
    print(f"  最终分数: {real_rewards[-1]:.2f}")
    print(f"  平均分数: {real_rewards.mean():.2f}")
    print(f"\n游戏时长 (帧):")
    print(f"  训练最终: {iterations[-100:].mean():.1f}")
    print(f"  评估最佳: {real_iterations.max():.1f}")
    print(f"\n损失:")
    print(f"  最终: {loss[-100:].mean():.4f}")
    print(f"  最小: {loss_smooth.min():.4f}")
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python plot_training_enhanced.py <model_name>")
        print("例如: python plot_training_enhanced.py my_model")
        print("\n这将加载以下文件:")
        print("  - my_model_rewards.npy")
        print("  - my_model_iterations.npy")
        print("  - my_model_loss")
        print("  - my_model_real_rewards.npy")
        print("  - my_model_real_iterations.npy")
        sys.exit(1)
    
    name = sys.argv[1]
    plot_training_results(name)
