from grid import CityGrid
from search import a_star, weighted_a_star, bidirectional_a_star, multi_goal_gbfs
from heuristics import manhattan

def main():
    # Create grid
    goals = [(39, 50), (27, 53), (24, 30), (21, 33), (0, 32)]

    grid_env = CityGrid(width=55, height=40, seed=69, obstacle_density= 0.85, goals=goals)
    start = grid_env.start
    goal = grid_env.goals[1]

    grid_env.display(start, goals)

    # A*
    path_a, cost_a, nodes_a = a_star(start, goal, grid_env.grid, manhattan)
    # Weighted A*
    path_w, cost_w, nodes_w = weighted_a_star(start, goal, grid_env.grid, manhattan, alpha=1.5)
    # Bidirectional A*
    path_b, cost_b, nodes_b = bidirectional_a_star(start, goal, grid_env.grid, manhattan)
    # multi goal gbfs
    path_segments, path_m, nodes_m, all_frontiers = multi_goal_gbfs(start, goals, grid_env.grid, manhattan)


    print(f"A*: cost={cost_a:.2f}, nodes={nodes_a}")
    print("A* path: ")
    grid_env.display_path(path_a, start, goal)

    print()
    print()

    print(f"Weighted A* (α=1.5): cost={cost_w:.2f}, nodes={nodes_w}")
    print("Weighted A* path: ")
    grid_env.display_path(path_w, start, goal)

    print()
    print()

    print(f"Bidirectional A*: cost={cost_b:.2f}, nodes={nodes_b}")
    print("Bidirectional A* path: ")
    grid_env.display_path(path_b, start, goal)

    print()
    print()

    print(f"M-GBFS*: nodes={nodes_m}")
    print("M-GBFS path: ")
    grid_env.display_multi_goal_path(path_segments, start, goals)

if __name__ == "__main__":
    main()


