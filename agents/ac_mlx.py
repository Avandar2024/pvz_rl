import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from pvz import config
import gymnasium as gym

HP_NORM = 100
SUN_NORM = 200

def log_prob_categorical(logits, actions):
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return mx.take_along_axis(log_probs, mx.expand_dims(actions, -1), axis=-1).squeeze(-1)

def entropy_categorical(logits):
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    probs = mx.exp(log_probs)
    return -mx.sum(probs * log_probs, axis=-1)

class ActorCriticNetwork(nn.Module):
    def __init__(self, base_dim=3, plant_dim=4, n_lanes=5, lane_length=9, global_feat_dim=5):
        super().__init__()
        self.n_lanes = n_lanes
        self.lane_length = lane_length
        
        self.resource_mlp = nn.Sequential(
            nn.Linear(global_feat_dim, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU()
        )
        
        self.lane_cnn = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        cnn_out_dim = 16 * lane_length
        
        self.lane_mlp = nn.Sequential(
            nn.Linear(64 + cnn_out_dim, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU()
        )
        
        self.global_noop_head = nn.Linear(64, 1)
        self.lane_score_head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        self.actor_plant = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, plant_dim))
        self.actor_location = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, lane_length))
        
        self.critic_head = nn.Sequential(
            nn.Linear(n_lanes * 128, 256), nn.GELU(),
            nn.Linear(256, 1)
        )
        
        # Orthogonal initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                m.weight = nn.init.orthogonal()(m.weight)
                if 'bias' in m and m.bias is not None:
                    m.bias = nn.init.constant(0.0)(m.bias)
            elif isinstance(m, nn.Conv1d):
                m.weight = nn.init.orthogonal()(m.weight)
                if 'bias' in m and m.bias is not None:
                    m.bias = nn.init.constant(0.0)(m.bias)
            return m
        
        self.apply(init_weights)

    def _extract_features(self, grid, global_features):
        global_emb = self.resource_mlp(global_features)
        
        batch_size = grid.shape[0]
        lane_grid = mx.reshape(mx.transpose(grid, (0, 2, 3, 1)), (-1, self.lane_length, 2))
        
        cnn_feat = self.lane_cnn(lane_grid)
        cnn_feat = mx.reshape(cnn_feat, (-1, 16 * self.lane_length))
        
        global_emb_expanded = mx.reshape(mx.broadcast_to(mx.expand_dims(global_emb, 1), (batch_size, self.n_lanes, 64)), (-1, 64))
        
        lane_input = mx.concatenate([global_emb_expanded, cnn_feat], axis=1)
        
        lane_emb = self.lane_mlp(lane_input)
        
        lane_emb = mx.reshape(lane_emb, (batch_size, self.n_lanes, -1))
        
        return global_emb, lane_emb, batch_size

    def __call__(self, grid, global_features, mask=None):
        global_emb, lane_emb, batch_size = self._extract_features(grid, global_features)
        
        lane_scores = mx.squeeze(self.lane_score_head(lane_emb), -1)
        noop_score = self.global_noop_head(global_emb)
        
        base_logits = mx.concatenate([noop_score, lane_scores], axis=1)
        
        plant_logits = self.actor_plant(lane_emb)
        loc_logits = self.actor_location(lane_emb)
        
        if mask is not None:
            mask_reshaped = mx.reshape(mask, (batch_size, self.n_lanes, self.lane_length))
            loc_logits = loc_logits + (mask_reshaped - 1) * 1e9
            
        lane_emb_flat = mx.reshape(lane_emb, (batch_size, -1))
        state_value = self.critic_head(lane_emb_flat)
        
        return base_logits, plant_logits, loc_logits, state_value

    def get_value(self, grid, global_features):
        global_emb, lane_emb, batch_size = self._extract_features(grid, global_features)
        lane_emb_flat = mx.reshape(lane_emb, (batch_size, -1))
        return self.critic_head(lane_emb_flat)

class PPOAgent:
    def __init__(self, lr=8e-5, gamma=0.99, eps_clip=0.1, K_epochs=80, gae_lambda=0.95, mini_batch_size=256, entropy_coef=0.03, reward_scale=1000.0, possible_actions=None):
        self.base_dim = 3
        self.plant_dim = 4
        self.location_dim = config.N_LANES * config.LANE_LENGTH
        self.n_lanes = config.N_LANES
        self.lane_length = config.LANE_LENGTH

        self.network = ActorCriticNetwork(self.base_dim, self.plant_dim, self.n_lanes, self.lane_length, global_feat_dim=5)
        
        self.optimizer = optim.Adam(learning_rate=lr, eps=1e-5)
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda
        self.mini_batch_size = mini_batch_size
        self.entropy_coef = entropy_coef
        self.reward_scale = reward_scale
        self.possible_actions = possible_actions
        
        self.reset_storage()

    def reset_storage(self):
        self.saved_states = []
        self.saved_actions = []
        self.saved_log_probs = []
        self.saved_rewards = []
        self.saved_dones = []
        self.saved_values = []

    def _split_state(self, state_tensor):
        grid_size = self.n_lanes * self.lane_length
        grid_end = 2 * grid_size
        
        mask = state_tensor[:, -grid_size:]
        global_features = state_tensor[:, grid_end:-grid_size]
        grid_flat = state_tensor[:, :grid_end]
        
        batch_size = state_tensor.shape[0]
        grid = mx.reshape(grid_flat, (batch_size, 2, self.n_lanes, self.lane_length))
        return grid, global_features, mask

    def decide_action(self, state):
        state_tensor = mx.array(state, dtype=mx.float32)
        
        is_single_env = len(state_tensor.shape) == 1
        if is_single_env:
            state_tensor = mx.expand_dims(state_tensor, 0)
            
        grid, global_features, mask = self._split_state(state_tensor)
        base_logits, plant_logits, loc_logits, value = self.network(grid, global_features, mask)
        value = mx.squeeze(value, -1)
        
        action_base = mx.random.categorical(base_logits)
        log_prob_base = log_prob_categorical(base_logits, action_base)

        B, N, P = plant_logits.shape
        action_plant = mx.reshape(mx.random.categorical(mx.reshape(plant_logits, (-1, P))), (B, N))
        log_prob_plant = mx.reshape(log_prob_categorical(mx.reshape(plant_logits, (-1, P)), mx.reshape(action_plant, (-1,))), (B, N))

        B, N, L = loc_logits.shape
        action_loc = mx.reshape(mx.random.categorical(mx.reshape(loc_logits, (-1, L))), (B, N))
        log_prob_loc = mx.reshape(log_prob_categorical(mx.reshape(loc_logits, (-1, L)), mx.reshape(action_loc, (-1,))), (B, N))

        is_placement = (action_base > 0)
        lane_indices = mx.maximum(action_base - 1, mx.array(0))
        
        selected_log_prob_plant = mx.take_along_axis(log_prob_plant, mx.expand_dims(lane_indices, 1), axis=1).squeeze(1)
        selected_log_prob_loc = mx.take_along_axis(log_prob_loc, mx.expand_dims(lane_indices, 1), axis=1).squeeze(1)
        
        combined_log_prob = log_prob_base + is_placement.astype(mx.float32) * (selected_log_prob_plant + selected_log_prob_loc)
        
        stored_actions = mx.concatenate([
            mx.expand_dims(action_base, 1), 
            action_plant, 
            action_loc
        ], axis=1)

        self.saved_states.append(state_tensor if not is_single_env else state_tensor[0])
        self.saved_actions.append(stored_actions if not is_single_env else stored_actions[0])
        self.saved_log_probs.append(combined_log_prob if not is_single_env else combined_log_prob[0])
        self.saved_values.append(value if not is_single_env else value[0])

        selected_plant = mx.take_along_axis(action_plant, mx.expand_dims(lane_indices, 1), axis=1).squeeze(1)
        selected_loc = mx.take_along_axis(action_loc, mx.expand_dims(lane_indices, 1), axis=1).squeeze(1)
        
        env_action = 1 + (selected_loc * self.n_lanes + lane_indices) * self.plant_dim + selected_plant
        env_action = mx.where(action_base == 0, mx.array(0), env_action)

        env_action_np = np.array(env_action)

        if is_single_env:
            return env_action_np.item()
        else:
            return env_action_np

    def store_reward_done(self, reward, done):
        scaled_reward = reward / self.reward_scale
        self.saved_rewards.append(mx.array(scaled_reward, dtype=mx.float32))
        self.saved_dones.append(mx.array(done, dtype=mx.float32))

    def update(self, next_observation):
        next_obs_tensor = mx.array(next_observation, dtype=mx.float32)
        
        if next_obs_tensor.size > 1:
            if len(next_obs_tensor.shape) == 1:
                next_obs_tensor = mx.expand_dims(next_obs_tensor, 0)
            grid, global_features, _ = self._split_state(next_obs_tensor)
            next_value = mx.squeeze(self.network.get_value(grid, global_features), -1)
            if next_value.size == 1:
                next_value = mx.squeeze(next_value)
        else:
            if len(self.saved_rewards[0].shape) > 0:
                next_value = mx.zeros_like(self.saved_rewards[0])
            else:
                next_value = mx.array(0.0, dtype=mx.float32)

        old_states = mx.stack(self.saved_states)
        old_actions = mx.stack(self.saved_actions)
        old_log_probs = mx.stack(self.saved_log_probs)
        old_values = mx.stack(self.saved_values)
        rewards = mx.stack(self.saved_rewards)
        dones = mx.stack(self.saved_dones)
        
        num_steps = len(rewards)
        advantages = np.zeros(rewards.shape, dtype=np.float32)
        last_gae = 0
        
        np_rewards = np.array(rewards)
        np_values = np.array(old_values)
        np_dones = np.array(dones)
        np_next_value = np.array(next_value)
        
        for t in reversed(range(num_steps)):
            V_next = np_next_value if t == num_steps - 1 else np_values[t+1]
            non_terminal = 1.0 - np_dones[t]
            
            delta = np_rewards[t] + self.gamma * V_next * non_terminal - np_values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae

        advantages = mx.array(advantages)
        returns = advantages + old_values
        
        b_states = mx.reshape(old_states, (-1, old_states.shape[-1]))
        b_actions = mx.reshape(old_actions, (-1, old_actions.shape[-1]))
        b_log_probs = mx.reshape(old_log_probs, (-1,))
        b_advantages = mx.reshape(advantages, (-1,))
        b_returns = mx.reshape(returns, (-1,))
        
        if b_advantages.size > 1:
            b_advantages = (b_advantages - mx.mean(b_advantages)) / (mx.std(b_advantages) + 1e-8)
            
        dataset_size = b_states.shape[0]
        total_loss, total_entropy, n_updates = 0, 0, 0
        
        def loss_fn(model, batch_states, batch_actions, batch_log_probs, batch_advantages, batch_returns):
            grid, global_features, mask = self._split_state(batch_states)
            base_logits, plant_logits, loc_logits, state_values = model(grid, global_features, mask)
            state_values = mx.squeeze(state_values, -1)
            
            a_base = batch_actions[:, 0]
            a_plant = batch_actions[:, 1 : 1 + self.n_lanes]
            a_loc = batch_actions[:, 1 + self.n_lanes :]

            new_log_prob_base = log_prob_categorical(base_logits, a_base)
            
            B, N, P = plant_logits.shape
            new_log_prob_plant = mx.reshape(log_prob_categorical(mx.reshape(plant_logits, (-1, P)), mx.reshape(a_plant, (-1,))), (B, N))
            
            B, N, L = loc_logits.shape
            new_log_prob_loc = mx.reshape(log_prob_categorical(mx.reshape(loc_logits, (-1, L)), mx.reshape(a_loc, (-1,))), (B, N))

            is_placement = (a_base > 0)
            lane_indices = mx.maximum(a_base - 1, mx.array(0)).astype(mx.int32)
            
            selected_log_prob_plant = mx.take_along_axis(new_log_prob_plant, mx.expand_dims(lane_indices, 1), axis=1).squeeze(1)
            selected_log_prob_loc = mx.take_along_axis(new_log_prob_loc, mx.expand_dims(lane_indices, 1), axis=1).squeeze(1)

            log_probs = new_log_prob_base + is_placement.astype(mx.float32) * (selected_log_prob_plant + selected_log_prob_loc)

            probs_base = mx.softmax(base_logits, axis=1)
            
            entropy_base = entropy_categorical(base_logits)
            entropy_plant = mx.reshape(entropy_categorical(mx.reshape(plant_logits, (-1, P))), (B, N))
            entropy_loc = mx.reshape(entropy_categorical(mx.reshape(loc_logits, (-1, L))), (B, N))
         
            lane_probs = probs_base[:, 1:]
            weighted_sub_entropy = mx.sum(lane_probs * (entropy_plant + entropy_loc), axis=1)
            
            entropy = entropy_base + weighted_sub_entropy
            
            ratios = mx.exp(log_probs - batch_log_probs)
            surr1 = ratios * batch_advantages
            surr2 = mx.clip(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * batch_advantages
            
            policy_loss = -mx.mean(mx.minimum(surr1, surr2))
            
            diff = mx.abs(state_values - batch_returns)
            value_loss = 0.5 * mx.mean(mx.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5))
            
            entropy_loss = -self.entropy_coef * mx.mean(entropy)
            
            loss = policy_loss + value_loss + entropy_loss
            
            # Add gradient clipping inside the loss function if possible, or handle it outside
            return loss, mx.mean(entropy)

        loss_and_grad_fn = nn.value_and_grad(self.network, loss_fn)
        
        for _ in range(self.K_epochs):
            indices = np.random.permutation(dataset_size)
            
            for start_idx in range(0, dataset_size, self.mini_batch_size):
                batch_indices = mx.array(indices[start_idx:start_idx + self.mini_batch_size])
                
                batch_states = b_states[batch_indices]
                batch_actions = b_actions[batch_indices]
                batch_log_probs = b_log_probs[batch_indices]
                batch_advantages = b_advantages[batch_indices]
                batch_returns = b_returns[batch_indices]
                
                (loss, entropy), grads = loss_and_grad_fn(self.network, batch_states, batch_actions, batch_log_probs, batch_advantages, batch_returns)
                
                # Clip gradients to prevent exploding gradients
                max_norm = 0.5
                
                # Flatten gradients and compute global norm
                from mlx.utils import tree_flatten, tree_map
                flat_grads = []
                for k, v in tree_flatten(grads):
                    flat_grads.append(mx.reshape(v, (-1,)))
                
                if flat_grads:
                    concat_grads = mx.concatenate(flat_grads)
                    total_norm = mx.sqrt(mx.sum(mx.square(concat_grads)))
                    clip_coef = mx.minimum(mx.array(1.0), max_norm / (total_norm + 1e-6))
                    clipped_grads = tree_map(lambda g: g * clip_coef, grads)
                else:
                    clipped_grads = grads
                # print(f"Clipped_grads: {clipped_grads}")
                
                self.optimizer.update(self.network, clipped_grads)
                mx.eval(self.network.parameters(), self.optimizer.state)
                
                total_loss += loss.item()
                total_entropy += entropy.item()
                n_updates += 1
                
        self.optimizer.learning_rate = self.optimizer.learning_rate * 0.99
        
        self.reset_storage()
        return (total_loss / n_updates, total_entropy / n_updates) if n_updates > 0 else (0, 0)

    def save(self, nn_name):
        self.network.save_weights(nn_name)

    def load(self, nn_name):
        self.network.load_weights(nn_name)

class Trainer():
    def __init__(self, render=False, max_frames=1000, training=True):
        self.env = gym.make('gym_pvz:pvz-env-v3')

        self.max_frames = max_frames
        self.render = render
        self.training = training
        
        self.env_base = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        self._grid_size = config.N_LANES * config.LANE_LENGTH

    def compile_agent_network(self, agent):
        pass

    def num_actions(self):
        return int(self.env.action_space.n)

    def _transform_observation(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        
        is_single = observation.ndim == 1
        if is_single:
            observation = observation[None, :]
            
        plant_grid = observation[:, :self._grid_size]
        zombie_grid = observation[:, self._grid_size:2*self._grid_size] / HP_NORM
        sun_val = observation[:, 2 * self._grid_size:2 * self._grid_size + 1] / SUN_NORM

        total_len = observation.shape[-1]
        n_plants = total_len - 3 * self._grid_size - 1

        card_info = observation[:, 2 * self._grid_size+1 : 2 * self._grid_size+1 + n_plants]
        location_mask = observation[:, 2 * self._grid_size+1 + n_plants:]

        new_observation = np.concatenate([
            plant_grid,
            zombie_grid,
            sun_val,
            card_info,
            location_mask
        ], axis=1).astype(np.float32)

        if is_single:
            return new_observation[0]
            
        return new_observation

    def _run_episode(self, agent):
        observation, _ = self.env.reset()
        observation = self._transform_observation(observation)
        
        episode_steps = 0
        episode_reward = 0.0
        actions = []
        rewards = []
        
        while episode_steps < self.max_frames:
            if self.render:
                self.env.render()

            action = agent.decide_action(observation)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            
            done = terminated or truncated
            next_obs = self._transform_observation(next_obs)
            
            agent.store_reward_done(reward, done)
            
            actions.append(action)
            rewards.append(reward)
            episode_reward += reward
            episode_steps += 1
            
            observation = next_obs
            
            if done:
                break
                
        return observation, done, actions, rewards, episode_steps, episode_reward

    def play(self, agent):
        last_obs, done, actions, rewards, steps, ep_reward = self._run_episode(agent)
        
        loss, entropy = 0, 0
        if self.training:
            next_value = np.array(0.0, dtype=np.float32) if done else last_obs
            loss, entropy = agent.update(next_value)
        else:
            agent.reset_storage()
        
        return {
            "rewards": np.array(rewards),
            "actions": np.array(actions),
            "episode_steps": steps,
            "episode_reward": ep_reward,
            "loss": loss,
            "entropy": entropy
        }

    def close(self):
        self.env.close()
