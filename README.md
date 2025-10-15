# Informed Search Strategies: Path Planning in Dynamic City Grids

## Overview

This repository implements and evaluates informed search algorithms for pathfinding in a simulated 2D city grid environment. Designed for the CS 272: Artificial Intelligence course, the project explores **Greedy Best-First Search (GBFS)** for multi-goal scenarios, **Weighted A\*** with tunable heuristic weighting, and **Bidirectional A\*** for efficient large-scale navigation. The algorithms use heuristics (Manhattan, Euclidean, non-admissible) to guide robots through obstacle-dense urban layouts, mimicking real-world applications like autonomous delivery systems.

Key focus: Balancing speed, optimality, and adaptability in static/dynamic grids with 70-85% obstacle density.

- **Language**: Python 3.8+
- **Dependencies**: `numpy`, `matplotlib`, `heapq` (standard library)
- **Environment**: 4-connected grid (no diagonals) for realistic urban movement

## Features

- **CityGrid Simulator**: Generates navigable city-like maps with roads, blocks, and dynamic obstacle modifications.
- **Multi-Goal GBFS**: Sequentially routes to nearest unvisited goals using dynamic heuristics.
- **Weighted A\***: Tunable α parameter for greedier searches (α > 1 trades optimality for speed).
- **Bidirectional A\***: Parallel forward/backward searches to halve exploration in symmetric grids.
- **Heuristics**: Manhattan (grid-optimal), Euclidean (straight-line), Non-Admissible (1.5x Manhattan for faster but suboptimal paths).
- **Visualization**: ASCII grids, path displays, and matplotlib plots for nodes expanded, cost, and runtime.
- **Evaluation**: Comparative metrics across heuristics and α values.

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/hamzaraheem06/Informed-Search-Strategies.git
   cd Informed-Search-Strategies
   ```

2. Install dependencies (optional, as most are standard):

   ```
   pip install numpy matplotlib
   ```

3. Run the project:
   ```
   python main.py
   ```

## Usage

### Core Modules

- `grid.py`: CityGrid class for environment setup and visualization.
- `heuristics.py`: Heuristic functions (e.g., `manhattan(start, goal)`).
- `search.py`: Algorithm implementations (e.g., `multi_goal_gbfs(start, goals, grid, heuristic)`).
- `visualize.py`: Basic visualization.
- `visualize_enhanced.py`: Advanced plotting.
- `main.py`: Experimental runner with plots.

### Running Experiments

Execute `main.py` to generate comparative plots:

```bash
python main.py
```

This runs algorithms across heuristics and α values, outputting:

- Bar charts: Nodes expanded, path cost, execution time.
- Console logs: e.g., "WA\* (α=1.5), Manhattan: cost=30, nodes=38, time=0.002 sec"

## Results Highlights

- **GBFS**: Fastest (0.002s) but suboptimal (cost +10-15%).
- **Weighted A\* (α=1.5)**: 40% fewer nodes than A\* (α=1), minor cost increase.
- **Bidirectional A\***: 50% reduction in expansions for static grids.
- Heuristic Impact: Manhattan best for 4-connected; Non-Admissible speeds up but risks dead-ends.

See `report.pdf` (or rendered images) for full visualizations.

## Structure

```
Informed-Search-Strategies/
├── grid.py              # CityGrid environment
├── heuristics.py        # Heuristic functions
├── search.py            # Search algorithms
├── visualize.py         # Basic visualization
├── visualize_enhanced.py # Advanced plotting
├── main.py              # Experiment runner
├── report.pdf           # Assignment report (optional)
└── README.md            # This file
```

## Contributing

Contributions welcome! Fork the repo, create a feature branch, and submit a PR. Focus on:

- New heuristics (e.g., Chebyshev).
- 8-connected movement.
- Real-time dynamic replanning (e.g., D\* Lite).

Report issues via GitHub Issues.

## Acknowledgments

- Inspired by _Artificial Intelligence: A Modern Approach_ (Russell & Norvig, 2021).
- Course: CS 272 – Artificial Intelligence, NUST.
- Thanks to GeeksforGeeks and Medium for algorithm insights.

---

_Built with ❤️ for efficient AI pathfinding. Questions? Open an issue!_

---
