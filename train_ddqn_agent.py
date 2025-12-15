import gymnasium as gym
from agents import experienceReplayBuffer, DDQNAgent, QNetwork
import torch



if __name__ == "__main__":
    n_iter = 100000
    env = gym.make('gym_pvz:pvz-env-v2')
    nn_name = input("Save name: ")
    # Windows 上若出现 0xc000012d（提交内存不足/分页文件不足），优先从降低 replay/batch 开始。
    buffer = experienceReplayBuffer(memory_size=50000, burn_in=5000)
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
    torch.save(agent.network, nn_name)
    agent._save_training_data(nn_name)
