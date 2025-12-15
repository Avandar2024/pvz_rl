import gymnasium as gym
from agents.dqn_agent import experienceReplayBuffer_DQN, DQNAgent, QNetwork_DQN
import torch
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100000, help="训练episode数量")
    args = parser.parse_args()
    
    n_iter = args.episodes
    env = gym.make('gym_pvz:pvz-env-v2')
    nn_name = input("Save name: ")
    buffer = experienceReplayBuffer_DQN(memory_size=100000, burn_in=10000)
    net = QNetwork_DQN(env, device='cpu', use_zombienet=False, use_gridnet=False)
    # old_agent = torch.load("agents/benchmark/dfq5_znet_epslinear")
    # net.zombienet.load_state_dict(old_agent.zombienet.state_dict())
    # for p in net.zombienet.parameters():
    #     p.requires_grad = False
    # net.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
    #                                       lr=net.learning_rate)
    agent = DQNAgent(env, net, buffer, n_iter=n_iter, batch_size=200)
    agent.train(max_episodes=n_iter, evaluate_frequency=5000, evaluate_n_iter=1000)
    torch.save(agent.network, nn_name)
    agent._save_training_data(nn_name)