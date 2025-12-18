"""
Lastone模型训练数据分析与可视化

分析 lastone 和 lastone_best 的训练数据，生成图表和统计报告
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_training_data(model_name):
    """加载训练数据"""
    base_path = Path("agents/agent_zoo") / model_name
    
    data = {}
    
    # 加载各项数据
    try:
        data['rewards'] = np.load(base_path / f"{model_name}_rewards.npy")
        print(f"✓ 加载 rewards: {len(data['rewards'])} episodes")
    except Exception as e:
        print(f"✗ 加载 rewards 失败: {e}")
        data['rewards'] = None
    
    try:
        data['iterations'] = np.load(base_path / f"{model_name}_iterations.npy")
        print(f"✓ 加载 iterations: {len(data['iterations'])} episodes")
    except Exception as e:
        print(f"✗ 加载 iterations 失败: {e}")
        data['iterations'] = None
    
    try:
        data['real_rewards'] = np.load(base_path / f"{model_name}_real_rewards.npy")
        print(f"✓ 加载 real_rewards (评估): {len(data['real_rewards'])} 次评估")
    except Exception as e:
        print(f"✗ 加载 real_rewards 失败: {e}")
        data['real_rewards'] = None
    
    try:
        data['real_iterations'] = np.load(base_path / f"{model_name}_real_iterations.npy")
        print(f"✓ 加载 real_iterations (评估): {len(data['real_iterations'])} 次评估")
    except Exception as e:
        print(f"✗ 加载 real_iterations 失败: {e}")
        data['real_iterations'] = None
    
    try:
        data['eval_stats'] = np.load(base_path / f"{model_name}_eval_stats.npy")
        print(f"✓ 加载 eval_stats (胜率统计): {len(data['eval_stats'])} 次评估")
    except Exception as e:
        print(f"✗ 加载 eval_stats 失败: {e}")
        data['eval_stats'] = None
    
    try:
        data['loss'] = torch.load(base_path / f"{model_name}_loss", weights_only=False)
        print(f"✓ 加载 loss: {len(data['loss'])} episodes")
    except Exception as e:
        print(f"✗ 加载 loss 失败: {e}")
        data['loss'] = None
    
    return data

def analyze_data(data, model_name):
    """分析训练数据"""
    analysis = {
        'model_name': model_name,
        'total_episodes': 0,
        'total_evaluations': 0,
    }
    
    # 训练奖励统计
    if data['rewards'] is not None:
        analysis['total_episodes'] = len(data['rewards'])
        analysis['avg_reward'] = float(np.mean(data['rewards']))
        analysis['max_reward'] = float(np.max(data['rewards']))
        analysis['min_reward'] = float(np.min(data['rewards']))
        analysis['final_100_avg_reward'] = float(np.mean(data['rewards'][-100:]))
        
        # 计算移动平均（窗口100）
        window = 100
        moving_avg = np.convolve(data['rewards'], np.ones(window)/window, mode='valid')
        analysis['best_moving_avg_reward'] = float(np.max(moving_avg))
    
    # 训练迭代统计
    if data['iterations'] is not None:
        analysis['avg_iteration'] = float(np.mean(data['iterations']))
        analysis['max_iteration'] = float(np.max(data['iterations']))
        analysis['final_100_avg_iteration'] = float(np.mean(data['iterations'][-100:]))
    
    # 评估奖励统计
    if data['real_rewards'] is not None:
        analysis['total_evaluations'] = len(data['real_rewards'])
        analysis['eval_avg_reward'] = float(np.mean(data['real_rewards']))
        analysis['eval_max_reward'] = float(np.max(data['real_rewards']))
        analysis['eval_final_reward'] = float(data['real_rewards'][-1])
        
        # 找到最佳评估结果
        best_eval_idx = np.argmax(data['real_rewards'])
        analysis['best_eval_reward'] = float(data['real_rewards'][best_eval_idx])
        analysis['best_eval_episode'] = int((best_eval_idx + 1) * 5000)  # 假设每5000轮评估一次
    
    # 评估迭代统计
    if data['real_iterations'] is not None:
        analysis['eval_avg_iteration'] = float(np.mean(data['real_iterations']))
        analysis['eval_final_iteration'] = float(data['real_iterations'][-1])
    
    # 胜率统计
    if data['eval_stats'] is not None:
        total_wins = 0
        total_losses = 0
        total_timeouts = 0
        total_games = 0
        
        win_rates = []
        for wins, losses, timeouts, n_games in data['eval_stats']:
            total_wins += wins
            total_losses += losses
            total_timeouts += timeouts
            total_games += n_games
            win_rates.append(wins / n_games * 100 if n_games > 0 else 0)
        
        analysis['total_games'] = int(total_games)
        analysis['total_wins'] = int(total_wins)
        analysis['total_losses'] = int(total_losses)
        analysis['total_timeouts'] = int(total_timeouts)
        analysis['overall_win_rate'] = float(total_wins / total_games * 100) if total_games > 0 else 0
        analysis['avg_win_rate'] = float(np.mean(win_rates))
        analysis['max_win_rate'] = float(np.max(win_rates))
        analysis['final_win_rate'] = float(win_rates[-1])
        
        # 找到最佳胜率
        best_wr_idx = np.argmax(win_rates)
        analysis['best_win_rate'] = float(win_rates[best_wr_idx])
        analysis['best_win_rate_episode'] = int((best_wr_idx + 1) * 5000)
    
    # Loss统计
    if data['loss'] is not None:
        analysis['avg_loss'] = float(np.mean(data['loss']))
        analysis['final_100_avg_loss'] = float(np.mean(data['loss'][-100:]))
    
    return analysis

def plot_training_curves(data, model_name, save_dir="."):
    """绘制训练曲线"""
    save_dir = Path(save_dir)
    
    # 创建大图
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 训练奖励曲线（移动平均）
    if data['rewards'] is not None:
        ax1 = plt.subplot(2, 3, 1)
        window = 100
        moving_avg = np.convolve(data['rewards'], np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(data['rewards'])), moving_avg, 'b-', linewidth=1.5, label='Moving Avg (100)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title(f'{model_name} - Training Reward (Moving Avg)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # 2. 训练迭代曲线
    if data['iterations'] is not None:
        ax2 = plt.subplot(2, 3, 2)
        window = 100
        moving_avg = np.convolve(data['iterations'], np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(data['iterations'])), moving_avg, 'g-', linewidth=1.5, label='Moving Avg (100)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Frames')
        ax2.set_title(f'{model_name} - Survival Frames (Moving Avg)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    # 3. Loss曲线
    if data['loss'] is not None:
        ax3 = plt.subplot(2, 3, 3)
        window = 100
        moving_avg = np.convolve(data['loss'], np.ones(window)/window, mode='valid')
        ax3.plot(range(window-1, len(data['loss'])), moving_avg, 'r-', linewidth=1.5, label='Moving Avg (100)')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss')
        ax3.set_title(f'{model_name} - Training Loss (Moving Avg)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_yscale('log')
    
    # 4. 评估奖励曲线
    if data['real_rewards'] is not None:
        ax4 = plt.subplot(2, 3, 4)
        eval_episodes = np.arange(1, len(data['real_rewards']) + 1) * 5000  # 假设每5000轮评估一次
        ax4.plot(eval_episodes, data['real_rewards'], 'b-o', linewidth=2, markersize=4, label='Eval Reward')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Avg Reward')
        ax4.set_title(f'{model_name} - Evaluation Reward')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # 标注最佳点
        best_idx = np.argmax(data['real_rewards'])
        ax4.plot(eval_episodes[best_idx], data['real_rewards'][best_idx], 'r*', markersize=15, label='Best')
        ax4.legend()
    
    # 5. 评估迭代曲线
    if data['real_iterations'] is not None:
        ax5 = plt.subplot(2, 3, 5)
        eval_episodes = np.arange(1, len(data['real_iterations']) + 1) * 5000
        ax5.plot(eval_episodes, data['real_iterations'], 'g-o', linewidth=2, markersize=4, label='Eval Frames')
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Avg Frames')
        ax5.set_title(f'{model_name} - Evaluation Survival Frames')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
    
    # 6. 胜率曲线
    if data['eval_stats'] is not None:
        ax6 = plt.subplot(2, 3, 6)
        win_rates = []
        loss_rates = []
        timeout_rates = []
        for wins, losses, timeouts, n_games in data['eval_stats']:
            win_rates.append(wins / n_games * 100 if n_games > 0 else 0)
            loss_rates.append(losses / n_games * 100 if n_games > 0 else 0)
            timeout_rates.append(timeouts / n_games * 100 if n_games > 0 else 0)
        
        eval_episodes = np.arange(1, len(win_rates) + 1) * 5000
        ax6.plot(eval_episodes, win_rates, 'g-o', linewidth=2, markersize=4, label='Win Rate')
        ax6.plot(eval_episodes, loss_rates, 'r-s', linewidth=2, markersize=4, label='Loss Rate')
        ax6.plot(eval_episodes, timeout_rates, 'b-^', linewidth=2, markersize=4, label='Timeout Rate')
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('Percentage (%)')
        ax6.set_title(f'{model_name} - Win/Loss/Timeout Rates')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        # 标注最佳胜率
        best_wr_idx = np.argmax(win_rates)
        ax6.plot(eval_episodes[best_wr_idx], win_rates[best_wr_idx], 'y*', markersize=15, label='Best Win Rate')
        ax6.legend()
    
    plt.tight_layout()
    
    # 保存图片
    save_path = save_dir / f"{model_name}_training_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 训练曲线图已保存: {save_path}")
    plt.close()

def save_analysis_report(analysis, save_dir="."):
    """保存分析报告"""
    save_dir = Path(save_dir)
    model_name = analysis['model_name']
    
    # 保存JSON格式
    json_path = save_dir / f"{model_name}_analysis.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"✓ 分析报告(JSON)已保存: {json_path}")
    
    # 保存可读文本格式
    txt_path = save_dir / f"{model_name}_analysis.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"=" * 80 + "\n")
        f.write(f"{model_name.upper()} 训练数据分析报告\n")
        f.write(f"=" * 80 + "\n\n")
        
        f.write(f"【训练概览】\n")
        f.write(f"  总训练轮数: {analysis.get('total_episodes', 'N/A')}\n")
        f.write(f"  总评估次数: {analysis.get('total_evaluations', 'N/A')}\n")
        f.write(f"  总游戏局数: {analysis.get('total_games', 'N/A')}\n\n")
        
        f.write(f"【训练奖励】\n")
        f.write(f"  平均奖励: {analysis.get('avg_reward', 'N/A'):.2f}\n")
        f.write(f"  最大奖励: {analysis.get('max_reward', 'N/A'):.2f}\n")
        f.write(f"  最小奖励: {analysis.get('min_reward', 'N/A'):.2f}\n")
        f.write(f"  最后100轮平均: {analysis.get('final_100_avg_reward', 'N/A'):.2f}\n")
        f.write(f"  最佳移动平均: {analysis.get('best_moving_avg_reward', 'N/A'):.2f}\n\n")
        
        f.write(f"【训练迭代】\n")
        f.write(f"  平均存活帧数: {analysis.get('avg_iteration', 'N/A'):.1f}\n")
        f.write(f"  最大存活帧数: {analysis.get('max_iteration', 'N/A'):.1f}\n")
        f.write(f"  最后100轮平均: {analysis.get('final_100_avg_iteration', 'N/A'):.1f}\n\n")
        
        f.write(f"【评估结果】\n")
        f.write(f"  评估平均奖励: {analysis.get('eval_avg_reward', 'N/A'):.2f}\n")
        f.write(f"  评估最大奖励: {analysis.get('eval_max_reward', 'N/A'):.2f}\n")
        f.write(f"  评估最终奖励: {analysis.get('eval_final_reward', 'N/A'):.2f}\n")
        f.write(f"  最佳评估奖励: {analysis.get('best_eval_reward', 'N/A'):.2f} (Episode {analysis.get('best_eval_episode', 'N/A')})\n")
        f.write(f"  评估平均帧数: {analysis.get('eval_avg_iteration', 'N/A'):.1f}\n")
        f.write(f"  评估最终帧数: {analysis.get('eval_final_iteration', 'N/A'):.1f}\n\n")
        
        f.write(f"【胜率统计】\n")
        f.write(f"  总胜局: {analysis.get('total_wins', 'N/A')}\n")
        f.write(f"  总败局: {analysis.get('total_losses', 'N/A')}\n")
        f.write(f"  总超时: {analysis.get('total_timeouts', 'N/A')}\n")
        f.write(f"  总体胜率: {analysis.get('overall_win_rate', 'N/A'):.2f}%\n")
        f.write(f"  平均胜率: {analysis.get('avg_win_rate', 'N/A'):.2f}%\n")
        f.write(f"  最大胜率: {analysis.get('max_win_rate', 'N/A'):.2f}%\n")
        f.write(f"  最终胜率: {analysis.get('final_win_rate', 'N/A'):.2f}%\n")
        f.write(f"  最佳胜率: {analysis.get('best_win_rate', 'N/A'):.2f}% (Episode {analysis.get('best_win_rate_episode', 'N/A')})\n\n")
        
        f.write(f"【训练Loss】\n")
        avg_loss = analysis.get('avg_loss', 'N/A')
        final_loss = analysis.get('final_100_avg_loss', 'N/A')
        f.write(f"  平均Loss: {avg_loss if avg_loss == 'N/A' else f'{avg_loss:.4f}'}\n")
        f.write(f"  最后100轮平均: {final_loss if final_loss == 'N/A' else f'{final_loss:.4f}'}\n\n")
        
        f.write(f"=" * 80 + "\n")
    
    print(f"✓ 分析报告(TXT)已保存: {txt_path}")

def main():
    print("=" * 80)
    print("Lastone 模型训练数据分析")
    print("=" * 80)
    
    model_name = "lastone"
    
    print(f"\n正在加载 {model_name} 的训练数据...")
    data = load_training_data(model_name)
    
    print(f"\n正在分析训练数据...")
    analysis = analyze_data(data, model_name)
    
    print(f"\n正在生成训练曲线图...")
    plot_training_curves(data, model_name, save_dir=".")
    
    print(f"\n正在保存分析报告...")
    save_analysis_report(analysis, save_dir=".")
    
    # 打印关键统计信息到控制台
    print("\n" + "=" * 80)
    print("关键统计信息")
    print("=" * 80)
    print(f"总训练轮数: {analysis.get('total_episodes', 'N/A')}")
    print(f"总评估次数: {analysis.get('total_evaluations', 'N/A')}")
    print(f"最终胜率: {analysis.get('final_win_rate', 'N/A'):.2f}%")
    print(f"最佳胜率: {analysis.get('best_win_rate', 'N/A'):.2f}% (Episode {analysis.get('best_win_rate_episode', 'N/A')})")
    print(f"最佳评估奖励: {analysis.get('best_eval_reward', 'N/A'):.2f} (Episode {analysis.get('best_eval_episode', 'N/A')})")
    print(f"最终评估奖励: {analysis.get('eval_final_reward', 'N/A'):.2f}")
    print("=" * 80)
    
    print("\n✅ 分析完成！所有文件已保存到根目录。")

if __name__ == "__main__":
    main()
