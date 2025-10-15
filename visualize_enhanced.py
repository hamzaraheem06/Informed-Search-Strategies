import time
import math
import random
import matplotlib.pyplot as plt
from search import weighted_a_star, bidirectional_a_star
from grid import CityGrid
from heuristics import manhattan, euclidean, non_admissible


# ---------- Run Experiments ----------
def run_experiment(grid_env, heuristic, alpha_values):
    """
    Runs Weighted A* for multiple α values and Bidirectional A* once,
    collecting runtime, cost, and node expansions.
    """
    start, goal = grid_env.start, grid_env.goals[0]
    results = {"Weighted A*": {}, "Bidirectional A*": {}}

    # Weighted A* with different alpha
    for alpha in alpha_values:
        t0 = time.perf_counter()
        try:
            path, cost, nodes = weighted_a_star(start, goal, grid_env.grid, heuristic, alpha)
        except Exception:
            path, cost, nodes = None, math.nan, math.nan
        t1 = time.perf_counter()

        results["Weighted A*"][alpha] = {
            "path": path,
            "cost": cost,
            "nodes": nodes,
            "time": t1 - t0
        }

    # Bidirectional A*
    t0 = time.perf_counter()
    try:
        path, cost, nodes = bidirectional_a_star(start, goal, grid_env.grid, heuristic)
    except Exception:
        path, cost, nodes = None, math.nan, math.nan
    t1 = time.perf_counter()

    results["Bidirectional A*"] = {
        "path": path,
        "cost": cost,
        "nodes": nodes,
        "time": t1 - t0
    }

    return results


# ---------- Modify Grid Dynamically ----------
def modify_grid(grid_env, num_changes=50, retries=10):
    """
    Randomly modifies obstacles while ensuring start and goal are reachable.
    """
    for _ in range(retries):
        new_grid = [row[:] for row in grid_env.grid]
        height, width = grid_env.height, grid_env.width

        for _ in range(num_changes):
            y, x = random.randint(0, height - 1), random.randint(0, width - 1)
            if (y, x) != grid_env.start and (y, x) not in grid_env.goals:
                new_grid[y][x] = 1 - new_grid[y][x]  # flip obstacle

        # Ensure start and goal are free
        s, g = grid_env.start, grid_env.goals[0]
        if new_grid[s[0]][s[1]] == 0 and new_grid[g[0]][g[1]] == 0:
            grid_env.grid = new_grid
            return True

    print("[!] Warning: Could not ensure valid dynamic grid.")
    return False


# ---------- Visualization ----------
def plot_comparison(results, heuristic_name, grid_type):
    """
    Plots Weighted A* vs Bidirectional A* comparison for nodes, cost, and time.
    """
    alpha_values = list(results["Weighted A*"].keys())
    metrics = ["nodes", "cost", "time"]
    titles = {"nodes": "Nodes Expanded", "cost": "Path Cost", "time": "Execution Time (s)"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Weighted A* vs Bidirectional A* ({heuristic_name} - {grid_type} Grid)", fontsize=14)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        wa_values = [results["Weighted A*"][a][metric] for a in alpha_values]
        bi_value = results["Bidirectional A*"][metric]

        ax.plot(alpha_values, wa_values, marker='o', label="Weighted A*")
        if not math.isnan(bi_value):
            ax.axhline(y=bi_value, color='r', linestyle='--', label="Bidirectional A*")

        ax.set_xlabel("α (Weight Factor)")
        ax.set_ylabel(titles[metric])
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()


# ---------- Experiment Runner ----------
def compare_algorithms():
    width, height = 25, 25
    alpha_values = [0.5, 1.0, 1.5, 2.0]
    heuristics = [
        ("Manhattan", manhattan),
        ("Euclidean", euclidean),
        ("Non-Admissible (1.5×Manhattan)", non_admissible)
    ]

    for grid_type in ["Static", "Dynamic"]:
        print(f"\n{'='*10} {grid_type.upper()} GRID {'='*10}")

        # Initialize city environment
        city = CityGrid(width, height, obstacle_density=0.55, seed=42)

        if grid_type == "Dynamic":
            modify_grid(city, num_changes=15)

        for hname, hfunc in heuristics:
            print(f"\n--- Heuristic: {hname} ---")
            results = run_experiment(city, hfunc, alpha_values)

            # Print summary table
            print("\nWeighted A* Results:")
            for a in alpha_values:
                r = results["Weighted A*"][a]
                print(f" α={a:<3} | Cost={r['cost']:<6} | Nodes={r['nodes']:<6} | Time={r['time']:.6f}s")

            rb = results["Bidirectional A*"]
            print(f"\nBidirectional A* | Cost={rb['cost']} | Nodes={rb['nodes']} | Time={rb['time']:.6f}s")

            # Plot results
            plot_comparison(results, hname, grid_type)


# ---------- Run ----------
if __name__ == "__main__":
    compare_algorithms()
