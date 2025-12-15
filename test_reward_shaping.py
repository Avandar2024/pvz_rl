"""
奖励塑形诊断脚本
用于验证 PBRS 奖励塑形是否正常工作，并观察各个组件的贡献

运行方式:
    python test_reward_shaping.py
"""

import gymnasium as gym
import numpy as np
from collections import defaultdict


def run_episode_with_diagnostics(env, policy="random", max_steps=500, verbose=True):
    """
    运行一个 episode 并收集奖励诊断信息
    
    policy: "random" 随机策略, "do_nothing" 什么都不做, "greedy_sunflower" 优先种向日葵
    """
    obs, info = env.reset()
    
    # 解包底层环境 (绕过 Gymnasium wrapper)
    inner_env = env
    while hasattr(inner_env, 'env'):
        inner_env = inner_env.env
    
    episode_stats = {
        "total_reward": 0.0,
        "base_rewards": [],
        "shaped_rewards": [],
        "potentials": [],
        "kills": [],
        "sun_history": [],
        "plant_counts": [],
        "zombie_counts": [],
        "steps": 0,
        "terminated": False,
        "truncated": False,
    }
    
    for step in range(max_steps):
        # 选择动作
        if policy == "random":
            mask = inner_env.mask_available_actions()
            valid_actions = np.where(mask)[0]
            action = np.random.choice(valid_actions)
        elif policy == "do_nothing":
            action = 0
        elif policy == "greedy_sunflower":
            # 优先种向日葵，其次豌豆射手
            mask = inner_env.mask_available_actions()
            valid_actions = np.where(mask)[0]
            # 动作编码: 1 + plant_id + 4 * (lane + 5 * pos)
            # plant_id: 0=sunflower, 1=peashooter, 2=wallnut, 3=potatomine
            sunflower_actions = [a for a in valid_actions if a > 0 and (a - 1) % 4 == 0]
            peashooter_actions = [a for a in valid_actions if a > 0 and (a - 1) % 4 == 1]
            
            if sunflower_actions and inner_env._scene.sun >= 50:
                action = sunflower_actions[0]
            elif peashooter_actions and inner_env._scene.sun >= 100:
                action = peashooter_actions[0]
            else:
                action = 0
        else:
            action = 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 记录统计
        episode_stats["total_reward"] += reward
        episode_stats["base_rewards"].append(info.get("base_reward", 0))
        episode_stats["shaped_rewards"].append(reward)
        episode_stats["kills"].append(info.get("total_kills", 0))
        episode_stats["sun_history"].append(info.get("sun", 0))
        episode_stats["plant_counts"].append(info.get("plants", 0))
        episode_stats["zombie_counts"].append(info.get("zombies", 0))
        episode_stats["steps"] += 1
        
        # 记录潜力值
        current_potential = inner_env._compute_potential()
        episode_stats["potentials"].append(current_potential)
        
        if verbose and step % 50 == 0:
            print(f"Step {step}: reward={reward:.3f}, sun={info.get('sun',0)}, "
                  f"plants={info.get('plants',0)}, zombies={info.get('zombies',0)}, "
                  f"kills={info.get('total_kills',0)}")
        
        if terminated or truncated:
            episode_stats["terminated"] = terminated
            episode_stats["truncated"] = truncated
            break
    
    return episode_stats


def print_episode_summary(stats, policy_name):
    """打印 episode 总结"""
    print(f"\n{'='*60}")
    print(f"策略: {policy_name}")
    print(f"{'='*60}")
    print(f"总步数: {stats['steps']}")
    print(f"总奖励: {stats['total_reward']:.2f}")
    print(f"总击杀: {stats['kills'][-1] if stats['kills'] else 0}")
    print(f"结束状态: {'失败(僵尸进家)' if stats['terminated'] else '胜利(撑过时间)' if stats['truncated'] else '进行中'}")
    print(f"最终阳光: {stats['sun_history'][-1] if stats['sun_history'] else 0}")
    print(f"最终植物数: {stats['plant_counts'][-1] if stats['plant_counts'] else 0}")
    
    # 奖励分解
    base_total = sum(stats['base_rewards'])
    shaped_total = stats['total_reward']
    shaping_contribution = shaped_total - base_total
    
    print(f"\n奖励分解:")
    print(f"  基础奖励总和: {base_total:.2f}")
    print(f"  塑形贡献: {shaping_contribution:.2f}")
    print(f"  塑形后总奖励: {shaped_total:.2f}")
    
    # 潜力变化
    if stats['potentials']:
        print(f"\n潜力函数变化:")
        print(f"  初始潜力: {stats['potentials'][0]:.4f}")
        print(f"  最终潜力: {stats['potentials'][-1]:.4f}")
        print(f"  最大潜力: {max(stats['potentials']):.4f}")


def compare_policies():
    """比较不同策略的奖励情况"""
    env = gym.make('gym_pvz:pvz-env-v2')
    
    policies = ["do_nothing", "random", "greedy_sunflower"]
    results = {}
    
    for policy in policies:
        print(f"\n\n{'#'*60}")
        print(f"# 测试策略: {policy}")
        print(f"{'#'*60}")
        
        # 运行多次取平均
        all_rewards = []
        all_steps = []
        all_kills = []
        
        n_episodes = 5
        for ep in range(n_episodes):
            stats = run_episode_with_diagnostics(env, policy=policy, verbose=(ep == 0))
            all_rewards.append(stats["total_reward"])
            all_steps.append(stats["steps"])
            all_kills.append(stats["kills"][-1] if stats["kills"] else 0)
            
            if ep == 0:
                print_episode_summary(stats, policy)
        
        results[policy] = {
            "avg_reward": np.mean(all_rewards),
            "std_reward": np.std(all_rewards),
            "avg_steps": np.mean(all_steps),
            "avg_kills": np.mean(all_kills),
        }
    
    # 对比总结
    print(f"\n\n{'='*60}")
    print("策略对比总结 (5 episodes 平均)")
    print(f"{'='*60}")
    print(f"{'策略':<20} {'平均奖励':>12} {'标准差':>10} {'平均步数':>10} {'平均击杀':>10}")
    print("-" * 62)
    for policy, r in results.items():
        print(f"{policy:<20} {r['avg_reward']:>12.2f} {r['std_reward']:>10.2f} "
              f"{r['avg_steps']:>10.1f} {r['avg_kills']:>10.1f}")
    
    env.close()


def test_potential_components():
    """测试潜力函数各个组件"""
    env = gym.make('gym_pvz:pvz-env-v2')
    obs, info = env.reset()
    
    # 获取底层环境
    inner_env = env
    while hasattr(inner_env, 'env'):
        inner_env = inner_env.env
    
    print("\n潜力函数组件测试:")
    print("=" * 50)
    
    # 初始状态
    print("\n1. 初始状态 (无植物无僵尸):")
    print(f"   Φ_sun = {inner_env._potential_sun():.4f}")
    print(f"   Φ_defense = {inner_env._potential_defense():.4f}")
    print(f"   Φ_threat = {inner_env._potential_threat():.4f}")
    print(f"   Φ_kills = {inner_env._potential_kills():.4f}")
    print(f"   Φ_total = {inner_env._compute_potential():.4f}")
    
    # 种一个向日葵
    print("\n2. 种一个向日葵后:")
    # 找到种向日葵的有效动作
    mask = inner_env.mask_available_actions()
    sunflower_actions = [a for a in np.where(mask)[0] if a > 0 and (a - 1) % 4 == 0]
    if sunflower_actions:
        env.step(sunflower_actions[0])
        print(f"   Φ_sun = {inner_env._potential_sun():.4f}")
        print(f"   Φ_defense = {inner_env._potential_defense():.4f}")
        print(f"   Φ_threat = {inner_env._potential_threat():.4f}")
        print(f"   Φ_total = {inner_env._compute_potential():.4f}")
    
    # 让时间推进，等僵尸出现
    print("\n3. 等待僵尸出现后:")
    for _ in range(20):
        env.step(0)  # 不操作
    print(f"   当前僵尸数: {len(inner_env._scene.zombies)}")
    print(f"   Φ_sun = {inner_env._potential_sun():.4f}")
    print(f"   Φ_defense = {inner_env._potential_defense():.4f}")
    print(f"   Φ_threat = {inner_env._potential_threat():.4f}")
    print(f"   Φ_total = {inner_env._compute_potential():.4f}")
    
    env.close()


if __name__ == "__main__":
    print("=" * 60)
    print("PvZ 奖励塑形诊断工具")
    print("=" * 60)
    
    # 1. 测试潜力函数组件
    test_potential_components()
    
    # 2. 比较不同策略
    compare_policies()
    
    print("\n\n诊断完成！")
    print("如果 'greedy_sunflower' 策略的平均奖励明显高于 'do_nothing',")
    print("说明奖励塑形正在有效地引导智能体学习经济和防御策略。")
