from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _set_global_seeds(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PatchEncoder(nn.Module):
    """Small CNN encoder for 5x5 patches."""

    def __init__(self, d_model: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 5 * 5, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        return self.fc(x)


class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.embedding(positions)


class TransformerQNetwork(nn.Module):
    def __init__(
        self,
        action_dim: int,
        seq_len: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        action_pad_idx: int = 4,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.action_pad_idx = action_pad_idx
        self.encoder = PatchEncoder(d_model)
        self.action_embed = nn.Embedding(action_dim + 1, d_model, padding_idx=action_pad_idx)
        self.reward_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional = PositionalEncoding(max_len=seq_len, d_model=d_model)
        self.q_head = nn.Linear(d_model, action_dim)

    def forward(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        reward_seq: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            obs_seq: (B, L, 5, 5)
            action_seq: (B, L)
            reward_seq: (B, L)
            lengths: (B,) valid lengths
        Returns:
            q_values: (B, action_dim)
        """
        bsz, seq_len, _, _ = obs_seq.shape
        x = obs_seq.view(bsz * seq_len, 1, 5, 5)
        patch_emb = self.encoder(x).view(bsz, seq_len, self.d_model)
        action_emb = self.action_embed(action_seq)
        reward_emb = self.reward_mlp(reward_seq.unsqueeze(-1))
        tokens = patch_emb + action_emb + reward_emb

        positions = torch.arange(seq_len, device=obs_seq.device).unsqueeze(0).expand(bsz, seq_len)
        tokens = tokens + self.positional(positions)

        # padding mask: True for padded positions
        max_len = seq_len
        mask = torch.arange(max_len, device=obs_seq.device).expand(bsz, max_len) >= lengths.unsqueeze(1)
        encoded = self.transformer(tokens, src_key_padding_mask=mask)

        # Gather last valid token per sequence
        idx = (lengths - 1).clamp(min=0).unsqueeze(1).unsqueeze(2).expand(bsz, 1, self.d_model)
        last_token = encoded.gather(1, idx).squeeze(1)
        return self.q_head(last_token)


@dataclass
class TransitionSequence:
    obs_seq: torch.Tensor
    action_seq: torch.Tensor
    reward_seq: torch.Tensor
    length: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs_seq: torch.Tensor
    next_action_seq: torch.Tensor
    next_reward_seq: torch.Tensor
    next_length: torch.Tensor
    done: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque[TransitionSequence] = deque(maxlen=capacity)

    def add(self, transition: TransitionSequence) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[TransitionSequence]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        actions: Sequence[int],
        seed: int | None = 42,
        gamma: float = 0.95,
        lr: float = 1e-3,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 8000,
        batch_size: int = 64,
        learning_start: int = 200,
        train_every: int = 4,
        target_sync_interval: int = 200,
        huber_loss: bool = True,
        normalize_factor: float = 8.0,
        seq_len: int = 16,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        device: str = "cpu",
    ):
        self.seed = seed
        _set_global_seeds(seed)
        self.device = torch.device(device)
        self.actions = list(actions)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learning_start = learning_start
        self.train_every = train_every
        self.target_sync_interval = target_sync_interval
        self.huber_loss = huber_loss
        self.normalize_factor = normalize_factor
        self.random = random.Random(seed)
        self.step_count = 0
        self.buffer_size = buffer_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.action_pad_idx = len(self.actions)
        self.obs_pad = [[1 for _ in range(5)] for _ in range(5)]
        self.replay_buffer = ReplayBuffer(buffer_size)
        self._init_networks(lr, d_model, nhead, num_layers)

    def _init_networks(self, lr: float, d_model: int, nhead: int, num_layers: int) -> None:
        self.policy_net = TransformerQNetwork(
            action_dim=len(self.actions),
            seq_len=self.seq_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            action_pad_idx=self.action_pad_idx,
        ).to(self.device)
        self.target_net = TransformerQNetwork(
            action_dim=len(self.actions),
            seq_len=self.seq_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            action_pad_idx=self.action_pad_idx,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

    def reset(self) -> None:
        _set_global_seeds(self.seed)
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.epsilon = 1.0
        self.step_count = 0
        self._init_networks(
            self.optimizer.param_groups[0]["lr"],
            self.d_model,
            self.nhead,
            self.num_layers,
        )

    def _prep_batch(
        self,
        obs_seq: List[List[List[int]]],
        action_seq: List[int],
        reward_seq: List[float],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        truncated_obs = obs_seq[-self.seq_len :]
        truncated_actions = action_seq[-self.seq_len :]
        truncated_rewards = reward_seq[-self.seq_len :]
        pad_count = self.seq_len - len(truncated_obs)
        length = torch.tensor([len(truncated_obs)], dtype=torch.long, device=self.device)
        obs_list = truncated_obs + [self.obs_pad for _ in range(pad_count)]
        obs_tensor = torch.tensor(obs_list, dtype=torch.float32, device=self.device) / self.normalize_factor
        pad_actions = [self.action_pad_idx for _ in range(pad_count)]
        action_list = truncated_actions + pad_actions
        action_tensor = torch.tensor(action_list, dtype=torch.long, device=self.device)
        pad_rewards = [0.0 for _ in range(pad_count)]
        reward_list = truncated_rewards + pad_rewards
        reward_tensor = torch.tensor(reward_list, dtype=torch.float32, device=self.device)
        return (
            obs_tensor.unsqueeze(0),
            action_tensor.unsqueeze(0),
            reward_tensor.unsqueeze(0),
            length,
        )

    def choose_action(
        self,
        obs_seq: List[List[List[int]]],
        action_seq: List[int],
        reward_seq: List[float],
    ) -> Tuple[int, List[float]]:
        obs_tensor, action_tensor, reward_tensor, length = self._prep_batch(obs_seq, action_seq, reward_seq)
        with torch.no_grad():
            q_values = self.policy_net(obs_tensor, action_tensor, reward_tensor, length).squeeze(0)
        q_list = q_values.detach().cpu().tolist()
        if self.random.random() < self.epsilon:
            action = self.random.choice(self.actions)
        else:
            max_q = max(q_list)
            best_actions = [a for a, q in enumerate(q_list) if q == max_q]
            action = self.random.choice(best_actions)
        return action, q_list

    def learn(
        self,
        obs_seq: List[List[List[int]]],
        action_seq: List[int],
        reward_seq: List[float],
        action: int,
        reward: float,
        next_obs_seq: List[List[List[int]]],
        next_action_seq: List[int],
        next_reward_seq: List[float],
        done: bool,
    ) -> float | None:
        obs_tensor, act_tensor, rew_tensor, length = self._prep_batch(obs_seq, action_seq, reward_seq)
        next_obs_tensor, next_act_tensor, next_rew_tensor, next_length = self._prep_batch(
            next_obs_seq, next_action_seq, next_reward_seq
        )

        transition = TransitionSequence(
            obs_seq=obs_tensor,
            action_seq=act_tensor,
            reward_seq=rew_tensor,
            length=length,
            action=torch.tensor([action], dtype=torch.long, device=self.device),
            reward=torch.tensor([reward], dtype=torch.float32, device=self.device),
            next_obs_seq=next_obs_tensor,
            next_action_seq=next_act_tensor,
            next_reward_seq=next_rew_tensor,
            next_length=next_length,
            done=torch.tensor([done], dtype=torch.float32, device=self.device),
        )
        self.replay_buffer.add(transition)
        self.step_count += 1

        if len(self.replay_buffer) < self.learning_start:
            return None
        if self.step_count % self.train_every != 0:
            return None
        if len(self.replay_buffer) < self.batch_size:
            return None

        transitions = self.replay_buffer.sample(self.batch_size)
        batch_obs = torch.cat([t.obs_seq for t in transitions], dim=0)
        batch_actions_seq = torch.cat([t.action_seq for t in transitions], dim=0)
        batch_rewards_seq = torch.cat([t.reward_seq for t in transitions], dim=0)
        batch_length = torch.cat([t.length for t in transitions], dim=0)
        batch_taken_action = torch.cat([t.action for t in transitions], dim=0)
        batch_reward = torch.cat([t.reward for t in transitions], dim=0)
        batch_done = torch.cat([t.done for t in transitions], dim=0)
        batch_next_obs = torch.cat([t.next_obs_seq for t in transitions], dim=0)
        batch_next_actions_seq = torch.cat([t.next_action_seq for t in transitions], dim=0)
        batch_next_rewards_seq = torch.cat([t.next_reward_seq for t in transitions], dim=0)
        batch_next_length = torch.cat([t.next_length for t in transitions], dim=0)

        q_values = self.policy_net(batch_obs, batch_actions_seq, batch_rewards_seq, batch_length)
        q_taken = q_values.gather(1, batch_taken_action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_policy_q = self.policy_net(
                batch_next_obs, batch_next_actions_seq, batch_next_rewards_seq, batch_next_length
            )
            next_actions = next_policy_q.argmax(dim=1, keepdim=True)
            next_target_q = self.target_net(
                batch_next_obs, batch_next_actions_seq, batch_next_rewards_seq, batch_next_length
            )
            next_q = next_target_q.gather(1, next_actions).squeeze(1)
            target_q = batch_reward + (1 - batch_done) * self.gamma * next_q

        if self.huber_loss:
            loss = F.smooth_l1_loss(q_taken, target_q)
        else:
            loss = F.mse_loss(q_taken, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()

        if self.step_count % self.target_sync_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.detach().cpu().item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
