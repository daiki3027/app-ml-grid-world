const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const episodeEl = document.getElementById("episode");
const stepEl = document.getElementById("step");
const epsilonEl = document.getElementById("epsilon");
const rewardEl = document.getElementById("total_reward");
const successEl = document.getElementById("success_rate");
const learningEl = document.getElementById("learning");
const speedEl = document.getElementById("speed");
const lastActionEl = document.getElementById("last_action");
const lastLossEl = document.getElementById("last_loss");
const avgLossEl = document.getElementById("avg_loss");
const avgEpRewardEl = document.getElementById("avg_ep_reward");

let ws;
let grid = [];
let agent = { x: 0, y: 0 };
let cellSize = 40;

const actionNames = ["Up", "Down", "Left", "Right"];

function connect() {
  ws = new WebSocket(`ws://${window.location.host}/ws`);

  ws.onopen = () => {
    statusEl.textContent = "connected";
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.grid) {
      grid = data.grid;
      agent = data.agent_pos;
      updateStats(data);
      draw();
      if (data.done) {
        statusEl.textContent = `episode finished: ${data.info?.status || ""}`;
      } else {
        statusEl.textContent = data.info?.status || "running";
      }
    }
  };

  ws.onclose = () => {
    statusEl.textContent = "disconnected, reconnecting...";
    setTimeout(connect, 1000);
  };

  ws.onerror = (err) => {
    console.error("websocket error", err);
    statusEl.textContent = "ws error";
  };
}

function sendMessage(type, payload = {}) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type, ...payload }));
  }
}

function updateStats(data) {
  episodeEl.textContent = data.episode;
  stepEl.textContent = data.step;
  epsilonEl.textContent = data.epsilon;
  rewardEl.textContent = data.total_reward;
  successEl.textContent = data.success_rate;
  learningEl.textContent = data.learning ? "ON" : "OFF";
  speedEl.textContent = `${data.speed}x`;
  lastActionEl.textContent =
    data.last_action === null || data.last_action === undefined
      ? "-"
      : actionNames[data.last_action] || data.last_action;
  lastLossEl.textContent = data.last_loss !== null && data.last_loss !== undefined ? data.last_loss : "-";
  avgLossEl.textContent =
    data.avg_recent_loss !== null && data.avg_recent_loss !== undefined ? data.avg_recent_loss : "-";
  avgEpRewardEl.textContent =
    data.avg_episode_reward !== null && data.avg_episode_reward !== undefined ? data.avg_episode_reward : "-";
}

function resizeCanvas() {
  if (!grid.length) return;
  const rows = grid.length;
  const cols = grid[0].length;
  const maxPx = 640;
  cellSize = Math.floor(maxPx / Math.max(rows, cols));
  canvas.width = cellSize * cols;
  canvas.height = cellSize * rows;
}

function draw() {
  if (!grid.length) return;
  resizeCanvas();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const colors = {
    0: "#eef4ff",
    1: "#cbd5e1",
    2: "#93c5fd",
    3: "#86efac",
    4: "#fecdd3",
  };

  for (let y = 0; y < grid.length; y++) {
    for (let x = 0; x < grid[0].length; x++) {
      ctx.fillStyle = colors[grid[y][x]] || "#1e293b";
      ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);

      ctx.strokeStyle = "rgba(15, 23, 42, 0.08)";
      ctx.strokeRect(x * cellSize, y * cellSize, cellSize, cellSize);
    }
  }

  // Agent
  ctx.fillStyle = "#fbbf24";
  ctx.beginPath();
  ctx.arc(
    agent.x * cellSize + cellSize / 2,
    agent.y * cellSize + cellSize / 2,
    cellSize * 0.35,
    0,
    Math.PI * 2
  );
  ctx.fill();
}

document.getElementById("start").addEventListener("click", () => {
  sendMessage("start");
});

document.getElementById("pause").addEventListener("click", () => {
  sendMessage("pause");
});

document.getElementById("reset").addEventListener("click", () => {
  sendMessage("reset");
});

document.getElementById("toggle-learning").addEventListener("click", () => {
  sendMessage("toggle_learning");
});

document.querySelectorAll(".speed button").forEach((btn) => {
  btn.addEventListener("click", () => {
    const speed = Number(btn.dataset.speed);
    sendMessage("speed", { value: speed });
  });
});

connect();
