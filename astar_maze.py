from __future__ import annotations

import argparse
import heapq
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.viz import (PALETTE, card_figure, caption, save_animation,
                        use_headless_if_saving)

# ---- tunables -------------------------------------------------------------
GRID_N = 31            # maze is GRID_N x GRID_N cells (odd looks best)
USE_HEURISTIC = True   # True = A*, False = Dijkstra (blind flood)
CELLS_PER_FRAME = 4    # search cells revealed per animation frame
HOLD_FRAMES = 25       # frames to hold the finished path before looping


def make_maze(n: int) -> np.ndarray:
    """Randomized DFS (recursive backtracker). 0 = open, 1 = wall."""
    g = np.ones((n, n), dtype=np.uint8)
    stack = [(1, 1)]
    g[1, 1] = 0
    while stack:
        y, x = stack[-1]
        nbrs = []
        for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            ny, nx = y + dy, x + dx
            if 1 <= ny < n - 1 and 1 <= nx < n - 1 and g[ny, nx] == 1:
                nbrs.append((ny, nx, dy, dx))
        if not nbrs:
            stack.pop()
            continue
        ny, nx, dy, dx = nbrs[np.random.randint(len(nbrs))]
        g[y + dy // 2, x + dx // 2] = 0   # knock down the wall between
        g[ny, nx] = 0
        stack.append((ny, nx))
    return g


def astar(grid, start, goal, use_heuristic):
    """Return (visit_order, path). visit_order is cells in expansion order."""
    n = grid.shape[0]

    def h(c):
        return abs(c[0] - goal[0]) + abs(c[1] - goal[1]) if use_heuristic else 0

    openh = [(h(start), 0, start)]
    came, gscore, visit_order, seen = {}, {start: 0}, [], set()
    while openh:
        _, g, cur = heapq.heappop(openh)
        if cur in seen:
            continue
        seen.add(cur)
        visit_order.append(cur)
        if cur == goal:
            break
        y, x = cur
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < n and 0 <= nx < n and grid[ny, nx] == 0:
                ng = g + 1
                if ng < gscore.get((ny, nx), 1e18):
                    gscore[(ny, nx)] = ng
                    came[(ny, nx)] = cur
                    heapq.heappush(openh, (ng + h((ny, nx)), ng, (ny, nx)))
    path, c = [], goal
    while c in came:
        path.append(c)
        c = came[c]
    path.append(start)
    path.reverse()
    return visit_order, path


def hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", metavar="OUT.gif")
    args = ap.parse_args()
    use_headless_if_saving(args.save)

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    grid = make_maze(GRID_N)
    start, goal = (1, 1), (GRID_N - 2, GRID_N - 2)
    visit_order, path = astar(grid, start, goal, USE_HEURISTIC)

    bg = hex_to_rgb(PALETTE["bg"])
    wall = hex_to_rgb(PALETTE["grid"])
    cool = hex_to_rgb(PALETTE["cool"])
    cyan = hex_to_rgb(PALETTE["cyan"])
    pink = hex_to_rgb(PALETTE["pink"])
    hot = hex_to_rgb(PALETTE["hot"])

    base = np.where(grid[..., None] == 1, wall, bg).astype(np.float64)

    n_search = len(visit_order)
    search_frames = (n_search + CELLS_PER_FRAME - 1) // CELLS_PER_FRAME
    path_frames = len(path)
    total = search_frames + path_frames + HOLD_FRAMES

    fig, ax = card_figure()
    img = ax.imshow(base / 255.0, interpolation="nearest")
    caption(ax, "A*  pathfinding" if USE_HEURISTIC else "dijkstra  flood")

    def render(frame):
        canvas = base.copy()
        revealed = min(frame * CELLS_PER_FRAME, n_search)
        # frontier: gradient cool -> cyan by visit order
        for i in range(revealed):
            y, x = visit_order[i]
            t = i / max(1, n_search - 1)
            canvas[y, x] = cool * (1 - t) + cyan * t
        # path draws on once search is done
        if frame >= search_frames:
            steps = min(frame - search_frames + 1, path_frames)
            for i in range(steps):
                y, x = path[i]
                t = i / max(1, path_frames - 1)
                canvas[y, x] = pink * (1 - t) + hot * t
        # start / goal always visible
        canvas[start] = hot
        canvas[goal] = pink
        img.set_array(np.clip(canvas / 255.0, 0.0, 1.0))
        return (img,)

    anim = FuncAnimation(fig, render, frames=total, interval=40, blit=True)

    if args.save:
        save_animation(anim, args.save, fps=25)
    else:
        plt.show()


if __name__ == "__main__":
    main()
