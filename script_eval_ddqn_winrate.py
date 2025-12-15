import argparse
import torch

from pvz import config
from agents import PlayerQ
from agents.ddqn_agent import QNetwork


def _base_env(env):
    e = env
    while hasattr(e, "env"):
        e = e.env
    return e


def main():
    parser = argparse.ArgumentParser(description="Evaluate DDQN model: mean return + win rate")
    parser.add_argument("--model", type=str, required=True, help="Path to torch-saved QNetwork (torch.save(network, path))")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"], help="Force device (default: auto)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    env = PlayerQ(render=False)

    if hasattr(torch.serialization, "safe_globals"):
        from torch.serialization import safe_globals as _safe_globals
        with _safe_globals([QNetwork]):
            agent = torch.load(args.model, weights_only=False, map_location=device)
    else:
        torch.serialization.add_safe_globals([QNetwork])
        agent = torch.load(args.model, weights_only=False, map_location=device)

    returns = []
    wins = 0
    losses = 0
    survived_frames = []

    for _ in range(args.episodes):
        summary = env.play(agent, epsilon=args.epsilon)
        episode_return = float(summary["rewards"].sum())
        returns.append(episode_return)

        inner = _base_env(env.env)
        chrono = int(inner._scene._chrono)
        lives = int(inner._scene.lives)
        survived_frames.append(min(chrono, config.MAX_FRAMES))

        # In pvz_env_v2: win is reaching time limit with lives > 0
        if chrono > config.MAX_FRAMES and lives > 0:
            wins += 1
        elif lives <= 0:#太阳足够
            losses += 1

    mean_return = sum(returns) / len(returns)
    mean_frames = sum(survived_frames) / len(survived_frames)
    win_rate = wins / len(returns)

    print(f"device={device}")
    print(f"episodes={len(returns)}")
    print(f"mean_return={mean_return:.2f}")
    print(f"win_rate={win_rate:.3f} ({wins}/{len(returns)})")
    print(f"mean_survived_frames={mean_frames:.1f}")


if __name__ == "__main__":
    main()
