import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
from pathlib import Path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python plot_training_ddqn.py <model_name>")
        print("例如: python plot_training_ddqn.py my_model")
        sys.exit(1)
    
    name = sys.argv[1]
    
    # 尝试从 agent_zoo 加载
    model_dir = Path("agents/agent_zoo") / name
    if model_dir.exists():
        print(f"从 {model_dir} 加载数据...")
        base_path = model_dir / name
    else:
        print(f"从当前目录加载数据: {name}")
        base_path = Path(name)
    
    base_path = str(base_path)
    rewards = np.load(base_path+"_rewards.npy")
    iterations = np.load(base_path+"_iterations.npy")
    loss = torch.load(base_path+"_loss", weights_only=False)
    real_rewards = np.load(base_path+"_real_rewards.npy")
    real_iterations = np.load(base_path+"_real_iterations.npy")

    n_iter = rewards.shape[0]
    n_record = real_rewards.shape[0]
    record_period = n_iter//n_record
    slice_size = 500

    rewards = np.reshape(rewards, (n_iter//slice_size, slice_size)).mean(axis=1)
    iterations = np.reshape(iterations, (n_iter//slice_size, slice_size)).mean(axis=1)
    loss = np.reshape(loss, (n_iter//slice_size, slice_size)).mean(axis=1)

    x = list(range(0, n_iter, slice_size))
    xx = list(range(1, n_iter, record_period))
    plt.plot(x, rewards)
    plt.plot(xx, real_rewards, color='red')
    plt.show()
    plt.plot(x, iterations)
    plt.plot(xx, real_iterations, color='red')
    plt.show()
    # plt.plot(x, loss)
    # plt.show()