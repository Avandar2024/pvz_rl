import torch
from pvz import config

from agents import evaluate, PlayerV2, ReinforceAgentV2
from agents import PlayerQ
from agents import PlayerQ_DQN
from agents import PPOAgent, KeyboardAgent, Trainer
from agents.ddqn_agent import QNetwork

agent_type = "AC"  # DDQN or Reinforce or AC or Keyboard


if __name__ == "__main__":
    if agent_type == "Reinforce":
        env = PlayerV2(render=False, max_frames=500 * config.FPS)
        agent = ReinforceAgentV2(
            input_size=env.num_observations(), possible_actions=env.get_actions()
        )
        agent.load("agents/agent_zoo/dfp5")

    if agent_type == "AC":
        env = Trainer(render=False, max_frames=500 * config.FPS, training=False)
        agent = PPOAgent(
            possible_actions=list(range(env.num_actions()))
        )
        # Note: Architecture changed, requires new model file
        agent.load("agents/agent_zoo/ppo_vec_agent/checkpoints/checkpoint_iter_200.pth")

    if agent_type == "DDQN":
        env = PlayerQ(render=False)
        # 修复PyTorch 2.6兼容性：允许加载QNetwork类
        if hasattr(torch.serialization, "safe_globals"):
            from torch.serialization import safe_globals as _safe_globals
            with _safe_globals([QNetwork]):
                agent = torch.load("agents/agent_zoo/dfq5_epsexp", weights_only=False, map_location="cpu")
        else:
            torch.serialization.add_safe_globals([QNetwork])
            agent = torch.load("agents/agent_zoo/dfq5_epsexp", weights_only=False, map_location="cpu")

    if agent_type == "DQN":
        env = PlayerQ_DQN(render=False)
        # 修复PyTorch 2.6兼容性：允许加载QNetwork类
        if hasattr(torch.serialization, "safe_globals"):
            from torch.serialization import safe_globals as _safe_globals
            with _safe_globals([QNetwork]):
                agent = torch.load("agents/agent_zoo/dfq5_dqn", weights_only=False, map_location="cpu")
        else:
            torch.serialization.add_safe_globals([QNetwork])
            agent = torch.load("agents/agent_zoo/dfq5_dqn", weights_only=False, map_location="cpu")

    if agent_type == "Keyboard":
        env = PlayerV2(render=True, max_frames=500 * config.FPS)
        agent = KeyboardAgent()

    avg_score, avg_iter, win_rate, loss_rate, timeout_rate, wins, losses, timeouts = evaluate(env, agent)
    total_games = wins + losses + timeouts
    print("\n" + "="*50)
    print(f"Mean score: {avg_score:.2f}")
    print(f"Mean iterations: {avg_iter:.1f}")
    print(f"Win: {win_rate:.1f}% ({wins}/{total_games})")
    print(f"Loss: {loss_rate:.1f}% ({losses}/{total_games})")
    print(f"Timeout: {timeout_rate:.1f}% ({timeouts}/{total_games})")
    print("="*50)
