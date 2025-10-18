import heapq
import math
import time


def a_star(start, goal, grid, heuristic):
    """
    A* Search Algorithm
    -------------------
    start: (y, x) tuple - starting position
    goal: (y, x) tuple - goal position
    grid: 2D list where 0 = free cell, 1 = obstacle
    heuristic: function(node, goal) -> float (e.g., Manhattan or Euclidean)

    Returns:
        path (list of (y, x)), total_cost (float), nodes_expanded (int)
    """

    height, width = len(grid), len(grid[0])

    # 8-connected grid → allows diagonal movement
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),  # up, down, left, right
    ]

    def in_bounds(y, x):
        return 0 <= y < height and 0 <= x < width

    def cost(a, b):
        # Cost = 1 for straight moves, sqrt(2) for diagonal
        return math.sqrt(2) if a[0] != b[0] and a[1] != b[1] else 1

    open_set = []
    heapq.heappush(open_set, (0, start))  # (f, node)

    came_from = {}
    g_score = {start: 0}
    nodes_expanded = 0

    while open_set:
        f, current = heapq.heappop(open_set)
        nodes_expanded += 1

        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, g_score[goal], nodes_expanded

        cy, cx = current

        for dy, dx in directions:
            ny, nx = cy + dy, cx + dx

            if not in_bounds(ny, nx) or grid[ny][nx] == 1:
                continue  # skip obstacles or out-of-bounds

            neighbor = (ny, nx)
            tentative_g = g_score[current] + cost(current, neighbor)

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))

    return None, float('inf'), nodes_expanded  # no path found


def weighted_a_star(start, goal, grid, heuristic, alpha=1.0):
    """
    Weighted A* Search Algorithm
    ----------------------------
    start: (y, x)
    goal: (y, x)
    grid: 2D list
    heuristic: function(node, goal)
    alpha: weight for heuristic (α > 1 = greedier)

    Returns:
        path, total_cost, nodes_expanded
    """
    height, width = len(grid), len(grid[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def in_bounds(y, x):
        return 0 <= y < height and 0 <= x < width

    def cost(a, b):
        return 1  # 4-connected

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    nodes_expanded = 0

    while open_set:
        f, current = heapq.heappop(open_set)
        nodes_expanded += 1

        if current == goal:
            # reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, g_score[goal], nodes_expanded

        cy, cx = current
        for dy, dx in directions:
            ny, nx = cy + dy, cx + dx
            if not in_bounds(ny, nx) or grid[ny][nx] == 1:
                continue

            neighbor = (ny, nx)
            tentative_g = g_score[current] + cost(current, neighbor)

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + alpha * heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))

    return None, float('inf'), nodes_expanded


def bidirectional_a_star(start, goal, grid, heuristic):
    """
    Bidirectional A* Search Algorithm
    ---------------------------------
    Runs A* from both start and goal until frontiers meet.

    Returns:
        path, total_cost, nodes_expanded
    """
    height, width = len(grid), len(grid[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def in_bounds(y, x):
        return 0 <= y < height and 0 <= x < width

    def cost(a, b):
        return 1

    # open sets for both searches
    open_fwd = [(0, start)]
    open_bwd = [(0, goal)]

    # cost and parents for both directions
    g_fwd = {start: 0}
    g_bwd = {goal: 0}
    came_from_fwd = {}
    came_from_bwd = {}

    visited_fwd = set()
    visited_bwd = set()

    nodes_expanded = 0
    meeting_node = None

    while open_fwd and open_bwd:
        # Expand from start side
        _, current_fwd = heapq.heappop(open_fwd)
        visited_fwd.add(current_fwd)
        nodes_expanded += 1

        if current_fwd in visited_bwd:
            meeting_node = current_fwd
            break

        cy, cx = current_fwd
        for dy, dx in directions:
            ny, nx = cy + dy, cx + dx
            if not in_bounds(ny, nx) or grid[ny][nx] == 1:
                continue
            neighbor = (ny, nx)
            tentative_g = g_fwd[current_fwd] + cost(current_fwd, neighbor)
            if tentative_g < g_fwd.get(neighbor, float('inf')):
                came_from_fwd[neighbor] = current_fwd
                g_fwd[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_fwd, (f_score, neighbor))

        # Expand from goal side
        _, current_bwd = heapq.heappop(open_bwd)
        visited_bwd.add(current_bwd)
        nodes_expanded += 1

        if current_bwd in visited_fwd:
            meeting_node = current_bwd
            break

        cy, cx = current_bwd
        for dy, dx in directions:
            ny, nx = cy + dy, cx + dx
            if not in_bounds(ny, nx) or grid[ny][nx] == 1:
                continue
            neighbor = (ny, nx)
            tentative_g = g_bwd[current_bwd] + cost(current_bwd, neighbor)
            if tentative_g < g_bwd.get(neighbor, float('inf')):
                came_from_bwd[neighbor] = current_bwd
                g_bwd[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, start)
                heapq.heappush(open_bwd, (f_score, neighbor))

    if meeting_node is None:
        return None, float('inf'), nodes_expanded

    # reconstruct path
    path_fwd = []
    node = meeting_node
    while node in came_from_fwd:
        path_fwd.append(node)
        node = came_from_fwd[node]
    path_fwd.append(start)
    path_fwd.reverse()

    path_bwd = []
    node = meeting_node
    while node in came_from_bwd:
        path_bwd.append(node)
        node = came_from_bwd[node]
    # skip meeting node to avoid duplication
    path_bwd = path_bwd[1:] if path_bwd else []

    full_path = path_fwd + path_bwd
    total_cost = g_fwd[meeting_node] + g_bwd[meeting_node]
    return full_path, total_cost, nodes_expanded


def greedy_best_first_search(start, goal, grid, heuristic):
    """
    Greedy Best-First Search (GBFS)
    Uses only heuristic to guide the search.
    Returns:
        path, nodes_expanded, frontier_history
    """
    height, width = len(grid), len(grid[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def in_bounds(y, x):
        return 0 <= y < height and 0 <= x < width

    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), start))
    came_from = {}
    visited = set()
    nodes_expanded = 0

    # To store frontier snapshots for visualization
    frontier_history = []

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        nodes_expanded += 1

        # Record current frontier (for visualization)
        frontier_snapshot = [node for (_, node) in open_set]
        frontier_history.append(frontier_snapshot)

        if current == goal:
            # reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, nodes_expanded, frontier_history

        cy, cx = current
        for dy, dx in directions:
            ny, nx = cy + dy, cx + dx
            if not in_bounds(ny, nx) or grid[ny][nx] == 1:
                continue
            neighbor = (ny, nx)
            if neighbor not in visited:
                came_from[neighbor] = current
                heapq.heappush(open_set, (heuristic(neighbor, goal), neighbor))

    return None, nodes_expanded, frontier_history


def multi_goal_gbfs(start, goals, grid, heuristic):
    """
    Multi-Goal Greedy Best-First Search
    -----------------------------------
    Visit all goals one by one using GBFS.
    Always move to the nearest unvisited goal.
    """
    current_start = start
    remaining_goals = goals[:]
    full_path = []
    total_nodes = 0
    path_segments = []
    all_frontiers = []

    while remaining_goals:
        # Find the nearest goal (by heuristic)
        nearest_goal = min(remaining_goals, key=lambda g: heuristic(current_start, g))

        # Run GBFS to that goal
        path, nodes, frontier_history = greedy_best_first_search(current_start, nearest_goal, grid, heuristic)

        if path is None:
            print(f"❌ No path found to goal {nearest_goal}")
            break

        # Append path (avoid duplicating the starting cell)
        path_segments.append(path)
        if full_path:
            full_path.extend(path[1:])
        else:
            full_path.extend(path)

        total_nodes += nodes
        all_frontiers.extend(frontier_history)

        # Move to next goal
        current_start = nearest_goal
        remaining_goals.remove(nearest_goal)

    return path_segments, full_path, total_nodes, all_frontiers


# Helper function to get the found path
def reconstruct_path(came_from, start, goal):
    """Reconstruct path from start to goal using parent pointers."""
    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def dijkstra(start, goal, grid, heuristic):
    """
    Dijkstra's Algorithm
    --------------------
    Uniform cost search (heuristic is ignored, uses g_score only).

    Returns:
        path, total_cost, nodes_expanded
    """
    height, width = len(grid), len(grid[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connected

    def in_bounds(y, x):
        return 0 <= y < height and 0 <= x < width

    def cost(a, b):
        return 1  # Uniform cost for grid cells

    open_set = []
    heapq.heappush(open_set, (0, start))  # (g, node)
    came_from = {}
    g_score = {start: 0}
    nodes_expanded = 0

    while open_set:
        _, current = heapq.heappop(open_set)
        nodes_expanded += 1

        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, g_score[goal], nodes_expanded

        cy, cx = current
        for dy, dx in directions:
            ny, nx = cy + dy, cx + dx
            if not in_bounds(ny, nx) or grid[ny][nx] == 1:
                continue

            neighbor = (ny, nx)
            tentative_g = g_score[current] + cost(current, neighbor)

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                # f_score = g + 0 (since heuristic=0 for Dijkstra)
                heapq.heappush(open_set, (tentative_g, neighbor))

    return None, float('inf'), nodes_expanded