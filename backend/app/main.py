from __future__ import annotations

import asyncio
import contextlib
import random
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .gridworld import GridWorld
from .dqn import DQNAgent


class TrainingSession:
    def __init__(self, seed: int = 42, maze_mode: str = "random", reward_shaping: bool = True):
        self.random = random.Random(seed)
        self.maze_mode = maze_mode  # "random", "fixed", "memory"
        self.reward_shaping = reward_shaping
        self.env = self._make_env()
        self.agent = DQNAgent(actions=[0, 1, 2, 3], seed=seed)
        self.episode = 1
        self.step = 0
        self.total_reward = 0.0
        self.running = False
        self.learning_enabled = True
        self.speed = 1.0
        self.max_steps = 200
        self.success_window = deque(maxlen=50)
        self._stop = False
        self.last_action = None
        self.last_q_values: List[float] | None = None
        self.distance_reward_weight = 0.02
        self.seq_len = self.agent.seq_len
        self.obs_history: Deque[List[List[int]]] = deque(maxlen=self.seq_len)
        self.action_history: Deque[int] = deque(maxlen=self.seq_len)
        self.reward_history: Deque[float] = deque(maxlen=self.seq_len)
        self.last_loss: float | None = None
        self.loss_window: Deque[float] = deque(maxlen=200)
        self.episode_rewards: Deque[float] = deque(maxlen=50)

    def _make_env(self) -> GridWorld:
        if self.maze_mode == "memory":
            return GridWorld.memory_challenge()
        if self.maze_mode == "random":
            return GridWorld.random(seed=self.random.randint(0, 1_000_000))
        return GridWorld.default()

    def reset(self) -> None:
        self.env = self._make_env()
        self.agent.reset()
        self.episode = 1
        self.step = 0
        self.total_reward = 0.0
        self.running = False
        self.last_action = None
        self.last_q_values = None
        self.success_window.clear()
        self.loss_window.clear()
        self.episode_rewards.clear()
        self._init_histories()

    def toggle_learning(self) -> bool:
        self.learning_enabled = not self.learning_enabled
        return self.learning_enabled

    def set_speed(self, multiplier: float) -> float:
        self.speed = max(0.1, min(multiplier, 50.0))
        return self.speed

    def stop(self) -> None:
        self._stop = True

    def _compute_success_rate(self) -> float:
        if not self.success_window:
            return 0.0
        return sum(self.success_window) / len(self.success_window)

    async def training_loop(self, websocket: WebSocket) -> None:
        self._init_histories()
        await self._send_state(websocket, done=False, info={"status": "init"})
        while not self._stop:
            if not self.running:
                await asyncio.sleep(0.05)
                continue

            obs_seq, action_seq, reward_seq = self._get_sequences()
            prev_distance = self.env.shortest_path_length()
            if self.learning_enabled:
                action, q_values = self.agent.choose_action(obs_seq, action_seq, reward_seq)
            else:
                action = self.agent.random.choice(self.agent.actions)
                _, q_values = self.agent.choose_action(obs_seq, action_seq, reward_seq)
            self.last_q_values = q_values
            result = self.env.step(action)
            self.last_action = action
            self.step += 1
            current_distance = self.env.shortest_path_length()

            shaped_reward = result.reward
            if self.reward_shaping and prev_distance is not None and current_distance is not None:
                shaped_reward += self.distance_reward_weight * (prev_distance - current_distance)
            self.total_reward += shaped_reward

            done = result.done or self.step >= self.max_steps
            if done and not result.done and self.step >= self.max_steps:
                result.info["status"] = "timeout"

            if self.learning_enabled:
                next_obs_seq, next_action_seq, next_reward_seq = self._build_next_sequences(
                    action, shaped_reward
                )
                loss = self.agent.learn(
                    obs_seq=obs_seq,
                    action_seq=action_seq,
                    reward_seq=reward_seq,
                    action=action,
                    reward=shaped_reward,
                    next_obs_seq=next_obs_seq,
                    next_action_seq=next_action_seq,
                    next_reward_seq=next_reward_seq,
                    done=done,
                )
                if loss is not None:
                    self.last_loss = loss
                    self.loss_window.append(loss)

            self._append_transition(action, shaped_reward)

            await self._send_state(websocket, done=done, info=result.info)

            if done:
                self.success_window.append(1 if result.info.get("status") == "goal" else 0)
                self.episode_rewards.append(self.total_reward)
                self.env = self._make_env()
                self.agent.decay_epsilon()
                self.episode += 1
                self.step = 0
                self.total_reward = 0.0
                self.last_action = None
                self.last_q_values = None
                self._init_histories()

            await asyncio.sleep(max(0.005, 0.05 / self.speed))

    async def _send_state(self, websocket: WebSocket, done: bool, info: Dict[str, Any]) -> None:
        obs = self.env.get_observation()
        message = {
            "grid": self.env.grid,
            "agent_pos": {"x": self.env.position[0], "y": self.env.position[1]},
            "episode": self.episode,
            "step": self.step,
            "epsilon": round(self.agent.epsilon, 4),
            "total_reward": round(self.total_reward, 3),
            "done": done,
            "info": info,
            "last_action": self.last_action,
            "success_rate": round(self._compute_success_rate(), 3),
            "learning": self.learning_enabled,
            "speed": self.speed,
            "observation": obs,
            "observation_patch": self.env.get_local_patch(5),
            "q_values": self.last_q_values,
            "random_maze": self.maze_mode == "random",
            "maze_mode": self.maze_mode,
            "last_loss": self.last_loss,
            "avg_recent_loss": round(sum(self.loss_window) / len(self.loss_window), 4)
            if self.loss_window
            else None,
            "avg_episode_reward": round(sum(self.episode_rewards) / len(self.episode_rewards), 3)
            if self.episode_rewards
            else None,
        }
        await websocket.send_json(message)

    def _init_histories(self) -> None:
        self.obs_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
        self.obs_history.append(self.env.get_local_patch(5))
        self.action_history.append(self.agent.action_pad_idx)
        self.reward_history.append(0.0)

    def _get_sequences(self) -> Tuple[List[List[List[int]]], List[int], List[float]]:
        return list(self.obs_history), list(self.action_history), list(self.reward_history)

    def _build_next_sequences(
        self, action: int, reward: float
    ) -> Tuple[List[List[List[int]]], List[int], List[float]]:
        next_obs_history = deque(self.obs_history, maxlen=self.seq_len)
        next_action_history = deque(self.action_history, maxlen=self.seq_len)
        next_reward_history = deque(self.reward_history, maxlen=self.seq_len)
        next_obs_history.append(self.env.get_local_patch(5))
        next_action_history.append(action)
        next_reward_history.append(reward)
        return list(next_obs_history), list(next_action_history), list(next_reward_history)

    def _append_transition(self, action: int, reward: float) -> None:
        self.obs_history.append(self.env.get_local_patch(5))
        self.action_history.append(action)
        self.reward_history.append(reward)


app = FastAPI()

# Allow local dev in case of different ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = Path(__file__).resolve().parents[2] / "frontend"


@app.get("/")
async def index():
    return FileResponse(frontend_dir / "index.html")

app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = TrainingSession()
    training_task = asyncio.create_task(session.training_loop(websocket))

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "start":
                session.running = True
            elif msg_type == "pause":
                session.running = False
            elif msg_type == "reset":
                session.reset()
                await session._send_state(websocket, done=False, info={"status": "reset"})
            elif msg_type == "toggle_learning":
                session.toggle_learning()
                await session._send_state(websocket, done=False, info={"status": "learning_toggled"})
            elif msg_type == "speed":
                speed = float(data.get("value", 1.0))
                session.set_speed(speed)
                await session._send_state(websocket, done=False, info={"status": "speed_updated"})
            elif msg_type == "maze_mode":
                mode_value = data.get("value", True)
                if isinstance(mode_value, str) and mode_value.lower() == "memory":
                    session.maze_mode = "memory"
                elif isinstance(mode_value, bool):
                    session.maze_mode = "random" if mode_value else "fixed"
                else:
                    session.maze_mode = "random"
                session.reset()
                await session._send_state(
                    websocket,
                    done=False,
                    info={"status": "maze_mode_updated", "random_maze": session.maze_mode == "random", "maze_mode": session.maze_mode},
                )
            else:
                await websocket.send_json({"info": {"status": "unknown_command", "command": msg_type}})
    except WebSocketDisconnect:
        session.stop()
        training_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await training_task
    except Exception:
        session.stop()
        training_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await training_task
        raise
