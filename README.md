# A* Pathfinding

Watch A* search a maze, expanding its frontier toward the goal and then tracing the shortest path back.

Part of my portfolio of small, from-scratch visualisations of computer-science ideas. Built on numpy and matplotlib, so every moving part is visible.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python astar_maze.py                  # live animated window
python astar_maze.py --save out.gif   # export a looping GIF
python astar_maze.py --save out.mp4   # smaller file, best for the web (needs ffmpeg)
```
