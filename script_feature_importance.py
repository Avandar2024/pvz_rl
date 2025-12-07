import torch
import shap
import numpy as np

from agents import PlayerQ
from pvz import config
# Import class used in the saved checkpoint so we can allowlist it for safe unpickling
from agents.ddqn_agent import QNetwork

n_ep = 100
obs = []

DEVICE = "cpu"
load_path = "agents/agent_zoo/dfq5_epsexp"

# Newer PyTorch defaults to weights_only=True which fails when the file contains
# full pickled objects. The error suggests allowing the QNetwork global so the
# object can be unpickled safely. We'll try to use safe_globals if available,
# else fall back to add_safe_globals, and finally loading without allowlist if
# necessary. All of these should only be used if you trust the source of the
# checkpoint.
agent = None
try:
    # Prefer the context manager API if present
    if hasattr(torch.serialization, "safe_globals"):
        from torch.serialization import safe_globals as _safe_globals
        with _safe_globals([QNetwork]):
            agent = torch.load(load_path, weights_only=False, map_location=DEVICE)
    elif hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([QNetwork])
        agent = torch.load(load_path, weights_only=False, map_location=DEVICE)
    else:
        # Last resort: attempt to load the full object (may be unsafe)
        agent = torch.load(load_path, weights_only=False, map_location=DEVICE)
except Exception:
    # If full object load fails, try loading as a state-dict or weights-only file.
    # This will succeed if the file contains only tensors/state_dict.
    try:
        agent = torch.load(load_path, map_location=DEVICE)
    except Exception:
        # Re-raise the original error if nothing worked
        raise

# Move loaded agent/network to the desired device safely.
try:
    # If the loaded object is a nn.Module, .to will work and return the module
    if isinstance(agent, torch.nn.Module):
        agent = agent.to(DEVICE)
    else:
        # If it's an agent-like object, move its networks if present
        if hasattr(agent, "network") and isinstance(agent.network, torch.nn.Module):
            agent.network = agent.network.to(DEVICE)
        if hasattr(agent, "target_network") and isinstance(agent.target_network, torch.nn.Module):
            agent.target_network = agent.target_network.to(DEVICE)
except Exception:
    # Fall back to leaving the loaded object as-is; most operations below will fail clearly
    pass
player = PlayerQ(render=False)

for episode_idx in range(n_ep):
    print("\r{}/{}".format(episode_idx, n_ep), end="")
    summary = player.play(agent)
    obs.append(summary["observations"])

_grid_size = config.N_LANES * config.LANE_LENGTH

obs = np.concatenate(obs)
obs = np.array([np.concatenate([state[:_grid_size],
                       np.sum(state[_grid_size: 2 * _grid_size].reshape(-1, config.LANE_LENGTH), axis=1),
                       state[2 * _grid_size:]]) for state in obs])

n_obs = len(obs)

def _sample_tensor_from_obs(obs_arr, k):
    """Return a torch.FloatTensor on DEVICE sampled from obs_arr with up to k unique samples."""
    k = min(len(obs_arr), int(k))
    if k <= 0:
        raise ValueError("No observations available for SHAP sampling")
    idx = np.random.choice(len(obs_arr), k, replace=False)
    return torch.from_numpy(obs_arr[idx]).float().to(DEVICE)

e = shap.DeepExplainer(
        agent.network,
        _sample_tensor_from_obs(obs, 100))

shap_values = e.shap_values(
    _sample_tensor_from_obs(obs, 30)
)

s = np.stack([np.sum(s, axis=0) for s in shap_values])
print(np.sum(s, axis=0))
shap.summary_plot(shap_values)
