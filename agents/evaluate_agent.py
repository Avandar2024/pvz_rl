import numpy as np
from pvz import config
import matplotlib.pyplot as plt
# from game_render import render


def _get_inner_env(env):
    """解包 Gymnasium wrapper 以访问底层环境"""
    inner = env
    while hasattr(inner, 'env'):
        inner = inner.env
    return inner


def evaluate(env, agent, n_iter=1000, verbose = True):
    sum_score = 0
    sum_iter = 0
    score_hist = []
    iter_hist = []
    n_iter = n_iter
    actions = []
    
    # 胜率统计
    wins = 0
    losses = 0
    timeouts = 0

    # 解包所有wrapper
    def _get_base_env(e):
        base_env = e
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        return base_env

    for episode_idx in range(n_iter):
        if verbose:
            print("\r{}/{}".format(episode_idx, n_iter), end="")

        # play episodes
        summary = env.play(agent)
        summary['score'] = np.sum(summary["rewards"])

        # 获取底层环境
        inner_env = _get_inner_env(env.env)

        score_hist.append(summary['score'])
        iter_hist.append(min(inner_env._scene._chrono, config.MAX_FRAMES))

        sum_score += summary['score']
        sum_iter += min(inner_env._scene._chrono, config.MAX_FRAMES)
        
        # 统计胜负
        scene = inner_env._scene
        if scene.is_victory():
            wins += 1
        elif scene.is_defeat():
            losses += 1
        else:
            timeouts += 1

        # if env.env._scene._chrono >= 1000:
        #    render_info = env.env._scene._render_info
        #    render(render_info)
        #    input()
        actions.append(summary['actions'])

    actions = np.concatenate(actions)
    plant_action = np.mod(actions - 1, 4)
    if verbose:
        # Plot of the score
        plt.hist(score_hist)
        plt.title("Score per play over {} plays".format(n_iter))
        plt.show()
        # Plot of the iterations
        plt.hist(iter_hist)
        plt.title("Survived frames per play over {} plays".format(n_iter))
        plt.show()
        # Plot of the action
        plt.hist(np.concatenate(actions), (np.arange(0, config.N_LANES * config.LANE_LENGTH * 4 + 2) - 0.5).tolist(), density=True)
        plt.title("Action usage density over {} plays".format(n_iter))
        plt.show()
        plt.hist(plant_action, (np.arange(0,5) - 0.5).tolist(), density=True)
        plt.title("Plant usage density over {} plays".format(n_iter))
        plt.show()

    win_rate = wins / n_iter * 100
    loss_rate = losses / n_iter * 100
    timeout_rate = timeouts / n_iter * 100
    
    return sum_score/n_iter, sum_iter/n_iter, win_rate, loss_rate, timeout_rate, wins, losses, timeouts
