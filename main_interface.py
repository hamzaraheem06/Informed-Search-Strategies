#!/usr/bin/env python3
"""
main_interface.py

Unified terminal interface to run the different demos / experiments
in your pathfinding project. Uses Option 1: imports existing modules
and calls their entry functions where available. Executes
islamabad_search.py as a script so it preserves its interactive prompts.

Author: Generated for you
"""

import sys
import runpy
import importlib
import traceback
from textwrap import dedent

# ANSI colors (should work on most terminals)
CSI = "\033["
RESET = CSI + "0m"
BOLD = CSI + "1m"
GREEN = CSI + "32m"
YELLOW = CSI + "33m"
RED = CSI + "31m"
CYAN = CSI + "36m"

def cprint(msg, color=""):
    print(f"{color}{msg}{RESET}" if color else msg)

def try_import(module_name):
    try:
        mod = importlib.import_module(module_name)
        return mod
    except Exception as e:
        return None

def check_optional_deps():
    deps = ["osmnx", "folium", "geopy", "matplotlib"]
    missing = []
    for d in deps:
        try:
            importlib.import_module(d)
        except Exception:
            missing.append(d)
    return missing

def run_random_grid_demo():
    """
    Calls the existing main.py demo (if available).
    Falls back to running the module as a script.
    """
    cprint("\n[Run] Random CityGrid demo (main.py)", CYAN)
    mod = try_import("main")
    if mod and hasattr(mod, "main"):
        try:
            mod.main()
        except Exception:
            cprint("Error while running main.main():", RED)
            traceback.print_exc()
    else:
        # fallback: execute as script (runs its __main__ block)
        try:
            runpy.run_module("main", run_name="__main__")
        except Exception:
            cprint("Failed to run main.py as a module.", RED)
            traceback.print_exc()

def run_visualize():
    cprint("\n[Run] Visualization (visualize.py): multi-algorithm comparisons", CYAN)
    mod = try_import("visualize")
    if mod and hasattr(mod, "main"):
        try:
            mod.main()
        except Exception:
            cprint("Error while running visualize.main():", RED)
            traceback.print_exc()
    else:
        try:
            runpy.run_module("visualize", run_name="__main__")
        except Exception:
            cprint("Failed to run visualize.py as a module.", RED)
            traceback.print_exc()

def run_visualize_enhanced():
    cprint("\n[Run] Enhanced experiments (visualize_enhanced.py): Weighted A* vs Bi-A*", CYAN)
    mod = try_import("visualize_enhanced")
    if mod and hasattr(mod, "compare_algorithms"):
        try:
            mod.compare_algorithms()
        except Exception:
            cprint("Error while running visualize_enhanced.compare_algorithms():", RED)
            traceback.print_exc()
    else:
        try:
            runpy.run_module("visualize_enhanced", run_name="__main__")
        except Exception:
            cprint("Failed to run visualize_enhanced.py as a module.", RED)
            traceback.print_exc()

def run_islamabad_search():
    """
    Execute islamabad_search as a script so it behaves exactly as before,
    prompting the user for start/goal.
    """
    cprint("\n[Run] Islamabad real-world pathfinding (islamabad_search.py)", CYAN)
    # runpy.run_module will execute the top-level code as __main__ (same as `python islamabad_search.py`)
    try:
        runpy.run_module("islamabad_search", run_name="__main__")
    except Exception:
        cprint("Error while running islamabad_search.py (see traceback):", RED)
        traceback.print_exc()

def run_quick_benchmark():
    """
    Quick benchmark summary:
    - Creates a small random CityGrid
    - Runs A*, Weighted A*(α=1.5), Bi-A*, GBFS, Dijkstra
    - Prints a compact table with cost, nodes expanded (and time if available)
    This is implemented here so it doesn't require changing existing files.
    """
    cprint("\n[Run] Quick benchmark (silent runs, summary output)", CYAN)
    # Import required pieces
    try:
        CityGrid = importlib.import_module("grid").CityGrid
        search = importlib.import_module("search")
        heur = importlib.import_module("heuristics")
    except Exception as e:
        cprint("Missing required modules for quick benchmark (grid/search/heuristics).", RED)
        cprint("Exception: " + str(e), YELLOW)
        return

    import time

    # Build small grid for quick runs
    grid_env = CityGrid(width=30, height=30, obstacle_density=0.4, seed=123)
    start = grid_env.start
    goal = grid_env.goals[0]

    # Algorithms to run
    algs = [
        ("A*", search.a_star, heur.manhattan, {}),
        ("WA* (α=1.5)", search.weighted_a_star, heur.manhattan, {"alpha": 1.5}),
        ("Bi-A*", search.bidirectional_a_star, heur.manhattan, {}),
        ("GBFS", search.greedy_best_first_search, heur.manhattan, {}),
        ("Dijkstra", search.dijkstra, heur.zero_heuristic, {})
    ]

    results = []
    for name, func, hfunc, extra in algs:
        try:
            t0 = time.perf_counter()
            if name == "GBFS":
                path, nodes, _ = func(start, goal, grid_env.grid, hfunc)
                cost = len(path) - 1 if path else float("inf")
            else:
                # other APIs return (path, cost, nodes)
                path, cost, nodes = func(start, goal, grid_env.grid, hfunc, **extra) if extra else func(start, goal, grid_env.grid, hfunc)
            t1 = time.perf_counter()
            elapsed = t1 - t0
        except TypeError:
            # Some function signatures may differ - try without extras
            try:
                t0 = time.perf_counter()
                path, cost, nodes = func(start, goal, grid_env.grid, hfunc)
                t1 = time.perf_counter()
                elapsed = t1 - t0
            except Exception as e:
                cprint(f"  [!] {name} failed: {e}", YELLOW)
                path, cost, nodes, elapsed = None, float("inf"), -1, float('nan')
        except Exception as e:
            cprint(f"  [!] {name} failed: {e}", YELLOW)
            path, cost, nodes, elapsed = None, float("inf"), -1, float('nan')

        results.append((name, cost, nodes, elapsed, bool(path)))

    # Print compact table
    print("\nQuick Benchmark Summary (grid: 30x30, seed=123):")
    header = f"{'Algorithm':20s} | {'Found':5s} | {'Cost':8s} | {'Nodes':8s} | {'Time (s)':9s}"
    print(header)
    print("-" * len(header))
    for name, cost, nodes, elapsed, found in results:
        found_s = "Yes" if found else "No"
        cost_s = f"{cost:.2f}" if cost != float("inf") else "Inf"
        nodes_s = f"{nodes}" if nodes >= 0 else "Err"
        time_s = f"{elapsed:.4f}" if not (elapsed is None) else "N/A"
        print(f"{name:20s} | {found_s:5s} | {cost_s:8s} | {nodes_s:8s} | {time_s:9s}")

    print()

def show_header():
    cprint(dedent(f"""
    {BOLD}{CYAN}=== AI Pathfinding Toolkit ==={RESET}
    A unified launcher for your project files.
    Running from: {GREEN}{sys.argv[0]}{RESET}
    """))

def show_menu():
    print(dedent(f"""
    {BOLD}Select an action:{RESET}
      1) Run Random CityGrid demo (main.py)
      2) Run visualization (visualize.py) - multi-algo comparison (plots)
      3) Run enhanced experiments (visualize_enhanced.py) - Weighted A* vs Bi-A*
      4) Run Islamabad real-world pathfinding (islamabad_search.py) [interactive prompts]
      5) Quick benchmark summary (silent runs, prints table)
      6) Check missing optional dependencies
      0) Exit
    """))

def main_loop():
    show_header()
    while True:
        try:
            show_menu()
            choice = input("Enter choice [0-6]: ").strip()
            if choice == "1":
                run_random_grid_demo()
            elif choice == "2":
                run_visualize()
            elif choice == "3":
                run_visualize_enhanced()
            elif choice == "4":
                run_islamabad_search()
            elif choice == "5":
                run_quick_benchmark()
            elif choice == "6":
                miss = check_optional_deps()
                if miss:
                    cprint("Missing optional dependencies (recommended):", YELLOW)
                    for d in miss:
                        print("  -", d)
                    cprint("Install with: pip install " + " ".join(miss), YELLOW)
                else:
                    cprint("All optional dependencies appear installed.", GREEN)
            elif choice == "0":
                cprint("\nGoodbye!", GREEN)
                break
            else:
                cprint("Invalid choice. Please enter a number from the menu.", YELLOW)

            input("\nPress Enter to return to menu...")
        except KeyboardInterrupt:
            cprint("\nInterrupted. Exiting.", RED)
            break
        except Exception:
            cprint("Unexpected error (traceback):", RED)
            traceback.print_exc()
            input("\nPress Enter to return to menu...")

if __name__ == "__main__":
    main_loop()
