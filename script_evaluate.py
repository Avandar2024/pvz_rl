import torch
from pvz import config

from agents import evaluate, PlayerV2, ReinforceAgentV2
from agents import PlayerQ
from agents import PlayerQ_DQN
from agents import ACAgent3, TrainerAC3, KeyboardAgent

agent_type = "DDQN"  # DDQN or Reinforce or AC or Keyboard


if __name__ == "__main__":
    if agent_type == "Reinforce":
        env = PlayerV2(render=False, max_frames=500 * config.FPS)
        agent = ReinforceAgentV2(
            input_size=env.num_observations(), possible_actions=env.get_actions()
        )
        agent.load("agents/agent_zoo/dfp5")

    if agent_type == "AC":
        env = TrainerAC3(render=False, max_frames=500 * config.FPS)
        agent = ACAgent3(
            input_size=env.num_observations(), possible_actions=env.get_actions()
        )
        agent.load("agents/agent_zoo/ac_policy_v1", "agents/agent_zoo/ac_value_v1")

    if agent_type == "DDQN":
        env = PlayerQ(render=False)
        agent = torch.load("agents/agent_zoo/dfq5_epsexp", weights_only=False)

    if agent_type == "DQN":
        env = PlayerQ_DQN(render=False)
        agent = torch.load("agents/agent_zoo/dfq5_dqn", weights_only=False)

    if agent_type == "Keyboard":
        env = PlayerV2(render=True, max_frames=500 * config.FPS)
        agent = KeyboardAgent()

    avg_score, avg_iter = evaluate(env, agent)
    print("\nMean score {}".format(avg_score))
    print("Mean iterations {}".format(avg_iter))
