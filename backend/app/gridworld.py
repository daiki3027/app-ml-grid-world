from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple


Grid = List[List[int]]
Position = Tuple[int, int]


@dataclass
class StepResult:
    next_state: Position
    reward: float
    done: bool
    info: dict


class GridWorld:
    """
    Simple GridWorld environment with walls, goal, and optional traps.
    Cells:
        0: empty
        1: wall
        2: start
        3: goal
        4: trap
    """

    def __init__(self, grid: Grid, start: Position, goal: Position, traps: List[Position] | None = None):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.traps = traps or []
        self.height = len(grid)
        self.width = len(grid[0]) if grid else 0
        self.position = start

    @classmethod
    def default(cls) -> "GridWorld":
        grid = [
            [2, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 1, 0, 1, 1, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [1, 1, 0, 0, 0, 1, 0, 0, 0, 3],
        ]
        start = (0, 0)
        goal = (9, 9)
        traps = [(4, 4), (8, 7)]
        grid[start[1]][start[0]] = 2
        grid[goal[1]][goal[0]] = 3
        for tx, ty in traps:
            grid[ty][tx] = 4
        return cls(grid=grid, start=start, goal=goal, traps=traps)

    @classmethod
    def random(
        cls,
        width: int = 10,
        height: int = 10,
        wall_density: float = 0.22,
        trap_count: int = 2,
        protect_radius: int = 1,
        seed: int | None = None,
        max_tries: int = 200,
    ) -> "GridWorld":
        """
        Generate a random maze with a guaranteed safe path (no walls/traps) from start to goal.
        The generator first carves a safe corridor, then sprinkles walls/traps outside a
        protected buffer and validates with BFS that treats traps as blocking.
        """
        rng = random.Random(seed)
        start = (0, 0)
        goal = (width - 1, height - 1)

        for _ in range(max_tries):
            grid = [[0 for _ in range(width)] for _ in range(height)]

            # 1) carve a guaranteed safe path
            safe_path = cls._carve_safe_path(start, goal, rng)
            for x, y in safe_path:
                grid[y][x] = 0

            # 2) place walls outside the safe path
            for y in range(height):
                for x in range(width):
                    if (x, y) in safe_path or (x, y) in (start, goal):
                        continue
                    if rng.random() < wall_density:
                        grid[y][x] = 1

            # 3) place traps away from the safe path (and its buffer)
            free_cells = [
                (x, y)
                for y in range(height)
                for x in range(width)
                if grid[y][x] == 0 and (x, y) not in (start, goal)
            ]
            rng.shuffle(free_cells)
            traps: List[Position] = []
            for x, y in free_cells:
                if len(traps) >= trap_count:
                    break
                if cls._is_near_path((x, y), safe_path, protect_radius):
                    continue
                grid[y][x] = 4
                traps.append((x, y))

            grid[start[1]][start[0]] = 2
            grid[goal[1]][goal[0]] = 3

            # 4) validate with BFS treating traps as blocking
            if cls._path_exists(grid, start, goal, traps=traps, treat_traps_blocking=True):
                return cls(grid=grid, start=start, goal=goal, traps=traps)

        # fallback: empty maze with no traps to guarantee reachability
        fallback_grid = [[0 for _ in range(width)] for _ in range(height)]
        fallback_grid[start[1]][start[0]] = 2
        fallback_grid[goal[1]][goal[0]] = 3
        return cls(grid=fallback_grid, start=start, goal=goal, traps=[])

    @classmethod
    def memory_challenge(cls) -> "GridWorld":
        """
        Maze with a loop/T-junction so local 5x5 views can be ambiguous without memory.
        Agent starts at top-left, must remember past turns to avoid cycling.
        """
        grid = [
            [2, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [1, 1, 0, 0, 0, 1, 0, 0, 0, 3],
        ]
        start = (0, 0)
        goal = (9, 9)
        traps: List[Position] = [(5, 0), (5, 2)]
        for tx, ty in traps:
            grid[ty][tx] = 4
        grid[start[1]][start[0]] = 2
        grid[goal[1]][goal[0]] = 3
        return cls(grid=grid, start=start, goal=goal, traps=traps)

    def step(self, action: int) -> StepResult:
        # Actions: 0=up,1=down,2=left,3=right
        moves = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        dx, dy = moves[action]
        new_x = self.position[0] + dx
        new_y = self.position[1] + dy

        reward = -0.01  # step penalty
        status = "move"

        if not self._in_bounds(new_x, new_y) or self._is_wall(new_x, new_y):
            # Bump into wall: stay in place
            reward -= 0.05
            new_x, new_y = self.position
            status = "wall"
        elif (new_x, new_y) == self.goal:
            reward += 1.0
            status = "goal"
            self.position = (new_x, new_y)
            return StepResult(next_state=self.position, reward=reward, done=True, info={"status": status})
        elif (new_x, new_y) in self.traps:
            reward -= 1.0
            status = "trap"
            self.position = (new_x, new_y)
            return StepResult(next_state=self.position, reward=reward, done=True, info={"status": status})

        self.position = (new_x, new_y)
        return StepResult(next_state=self.position, reward=reward, done=False, info={"status": status})

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _is_wall(self, x: int, y: int) -> bool:
        return self.grid[y][x] == 1

    def get_observation(self) -> Grid:
        """Return a copy of the grid with the agent drawn as 8 (display only)."""
        obs = [row[:] for row in self.grid]
        ax, ay = self.position
        obs[ay][ax] = 8
        return obs

    def get_local_patch(self, size: int = 5) -> Grid:
        """Return a square patch centered on the agent; out-of-bounds treated as walls."""
        if size % 2 == 0:
            raise ValueError("Patch size must be odd.")
        half = size // 2
        patch = [[1 for _ in range(size)] for _ in range(size)]
        ax, ay = self.position
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                gx, gy = ax + dx, ay + dy
                if self._in_bounds(gx, gy):
                    patch[dy + half][dx + half] = self.grid[gy][gx]
        patch[half][half] = 8  # mark agent at center
        return patch

    def shortest_path_length(self, from_pos: Position | None = None, avoid_traps: bool = True) -> int | None:
        """BFS shortest path length to goal; returns None if unreachable."""
        start = from_pos or self.position
        if start == self.goal:
            return 0
        queue: deque[tuple[Position, int]] = deque()
        queue.append((start, 0))
        visited = set([start])
        while queue:
            (x, y), dist = queue.popleft()
            for nx, ny in self._neighbors(x, y):
                if not self._in_bounds(nx, ny):
                    continue
                if (nx, ny) in visited:
                    continue
                if self._is_blocked(nx, ny, treat_traps=avoid_traps):
                    continue
                if (nx, ny) == self.goal:
                    return dist + 1
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))
        return None

    @staticmethod
    def _path_exists(
        grid: Grid,
        start: Position,
        goal: Position,
        traps: List[Position] | None = None,
        treat_traps_blocking: bool = True,
    ) -> bool:
        queue: deque[Position] = deque([start])
        visited = {start}
        height = len(grid)
        width = len(grid[0]) if grid else 0
        trap_set = set(traps or [])

        def in_bounds(x: int, y: int) -> bool:
            return 0 <= x < width and 0 <= y < height

        def is_blocked(x: int, y: int) -> bool:
            if grid[y][x] == 1:
                return True
            if grid[y][x] == 4 and treat_traps_blocking:
                return True
            if treat_traps_blocking and (x, y) in trap_set:
                return True
            return False

        while queue:
            x, y = queue.popleft()
            if (x, y) == goal:
                return True
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = x + dx, y + dy
                if not in_bounds(nx, ny):
                    continue
                if (nx, ny) in visited:
                    continue
                if is_blocked(nx, ny):
                    continue
                visited.add((nx, ny))
                queue.append((nx, ny))
        return False

    @staticmethod
    def _is_near_path(pos: Position, path: List[Position], radius: int) -> bool:
        if radius <= 0:
            return pos in path
        px, py = pos
        for x, y in path:
            if abs(px - x) + abs(py - y) <= radius:
                return True
        return False

    @staticmethod
    def _carve_safe_path(start: Position, goal: Position, rng: random.Random) -> List[Position]:
        """Create a monotonic-but-randomized corridor from start to goal."""
        x, y = start
        gx, gy = goal
        path = [(x, y)]
        while (x, y) != (gx, gy):
            moves = []
            if x < gx:
                moves.append((1, 0))
            if y < gy:
                moves.append((0, 1))
            if x > gx:
                moves.append((-1, 0))
            if y > gy:
                moves.append((0, -1))
            dx, dy = rng.choice(moves)
            x, y = x + dx, y + dy
            if (x, y) not in path:
                path.append((x, y))
        return path

    def _neighbors(self, x: int, y: int) -> List[Position]:
        return [(x + dx, y + dy) for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))]

    def _is_blocked(self, x: int, y: int, treat_traps: bool = True) -> bool:
        if self._is_wall(x, y):
            return True
        if treat_traps and (x, y) in self.traps:
            return True
        if treat_traps and self.grid[y][x] == 4:
            return True
        return False
