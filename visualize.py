import time
import matplotlib.pyplot as plt
from grid import CityGrid
from search import greedy_best_first_search, weighted_a_star, bidirectional_a_star, dijkstra
from heuristics import manhattan, euclidean, non_admissible, chebyshev, weighted_euclidean, zero_heuristic

def main():
    # Set up the grid environment (same as main.py for consistency)
    goals = [(39, 50), (27, 53), (24, 30), (21, 33), (0, 32)]

    grid_env = CityGrid(width=55, height=40, seed=69, obstacle_density= 0.85, goals=goals)
    start = grid_env.start
    # Choose a single goal for analysis (using the second goal for consistency)
    goal = goals[1]  # (14, 19)

    # Define heuristics to test (two admissible, one non-admissible)
    heuristics = [
        ("Manhattan", manhattan),
        ("Euclidean", euclidean),
        ("Non-Admissible (1.5x Manhattan)", non_admissible),
        ("Chebyshev", chebyshev),
        ("Weighted Euclidean", weighted_euclidean),
        ("Zero Heuristic", zero_heuristic)
    ]

    # Define algorithms and parameters (including multiple weighted A* alphas)
    algorithms = [
        ("GBFS", greedy_best_first_search, None),
        ("WA* (α=1.0)", weighted_a_star, 1.0),
        ("WA* (α=1.5)", weighted_a_star, 1.5),
        ("WA* (α=2.0)", weighted_a_star, 2.0),
        ("Bi-A*", bidirectional_a_star, None),
        ("Dijkstra", dijkstra, None)
    ]

    # Prepare a structure to store results
    # results[algorithm_name][heuristic_name] = {"nodes": ..., "cost": ..., "time": ...}
    results = {alg[0]: {} for alg in algorithms}

    # Run each algorithm with each heuristic and record metrics
    for alg_name, alg_func, alpha in algorithms:
        for heur_name, heur_func in heuristics:
            start_time = time.perf_counter()
            if alg_name == "GBFS":
                path, nodes, _ = alg_func(start, goal, grid_env.grid, heur_func)
                cost = len(path) - 1  # cost = number of moves (4-connected, cost=1 each)
            elif alg_name.startswith("WA*"):
                path, cost, nodes = alg_func(start, goal, grid_env.grid, heur_func, alpha=alpha)
            elif alg_name == "Bi-A*":
                path, cost, nodes = alg_func(start, goal, grid_env.grid, heur_func)
            elif alg_name == "Dijkstra":
                path, cost, expanded = alg_func(start, goal, grid_env.grid, heur_func)
            else:
                path, cost, nodes = None, float('inf'), 0
            end_time = time.perf_counter()
            elapsed = end_time - start_time

            results[alg_name][heur_name] = {"nodes": nodes, "cost": cost, "time": elapsed}
            print(f"{alg_name}, {heur_name}: cost = {cost}, nodes = {nodes}, time = {elapsed:.6f} sec")

    # Visualization: subplots for each metric (rows) and heuristic (columns)
    metrics = ["nodes", "cost", "time"]
    metric_labels = {"nodes": "Nodes Expanded", "cost": "Path Cost", "time": "Execution Time (sec)"}
    fig, axes = plt.subplots(len(metrics), len(heuristics), figsize=(12, 9))

    # Define a consistent algorithm order and colors
    alg_order = [alg[0] for alg in algorithms]
    colors = plt.get_cmap('tab10').colors
    alg_colors = {alg: colors[i % len(colors)] for i, (alg, _, _) in enumerate(algorithms)}

    # Fill in subplots
    for i, metric in enumerate(metrics):
        for j, (heur_name, _) in enumerate(heuristics):
            ax = axes[i][j]
            # Collect metric values for each algorithm
            values = [results[alg_name][heur_name][metric] for alg_name in alg_order]
            # Create bar chart for this subplot
            bars = ax.bar(alg_order, values, color=[alg_colors[name] for name in alg_order])
            # Set column title for heuristic (top row)
            if i == 0:
                ax.set_title(heur_name)
            # Set row label for metric (first column)
            if j == 0:
                ax.set_ylabel(metric_labels[metric])
            # Rotate x-axis labels for readability
            ax.set_xticks(range(len(alg_order)))
            ax.set_xticklabels(alg_order, rotation=45, ha='right')
            # Annotate bars with their value
            for bar in bars:
                height = bar.get_height()
                label = f"{height:.2f}" if metric == "time" else f"{int(height)}"
                ax.annotate(label,
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
