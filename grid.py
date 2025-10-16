import random
from collections import deque

# CityGrid Environment
# This class represents a simulated city map for an autonomous robot.
# Each cell can be free (0) or obstacle (1).
# The grid supports:
#   - 8-connected movement (diagonals allowed)
#   - Random obstacle generation (city-like layout)
#   - Dynamic updates (add/remove obstacles)
#   - Movement costs (diagonals slightly higher)

class CityGrid:
    def __init__(self, width=None, height=None, obstacle_density=.70, seed=None, goals=None, external_grid=None):
        """
        Initialize the city grid environment.

        :param width: number of columns in the grid (required unless external_grid provided)
        :param height: number of rows in the grid (required unless external_grid provided)
        :param obstacle_density: approximate proportion of obstacle cells
        :param seed: random seed for reproducibility
        :param goals: initialize grid with goals at initial stage.
        :param external_grid: optional pre-built grid (list of lists); if provided, skips random generation and uses its dimensions.
        """
        import random
        self.random = random.Random(seed) if seed is not None else random.Random()

        if external_grid is not None:
            # Use provided external grid
            self.height = len(external_grid)
            self.width = len(external_grid[0]) if self.height > 0 else 0
            self.grid = [row[:] for row in external_grid]  # Deep copy
            total_cells = self.height * self.width
            self.obstacle_density = sum(
                sum(1 for cell in row if cell == 1) for row in self.grid) / total_cells if total_cells > 0 else 0
            self.start = (0, 0)
            self.goals = goals if goals else [(self.height - 1, self.width - 1)]
            # Ensure start and goals are free
            if self.is_valid(*self.start):
                self.grid[self.start[0]][self.start[1]] = 0
            for gy, gx in self.goals:
                if self.is_valid(gy, gx):
                    self.grid[gy][gx] = 0
            return  # Skip random generation

        # Original random generation path (require width/height)
        if width is None or height is None:
            raise ValueError("width and height must be provided when external_grid is not specified.")

        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
        self.obstacle_density = obstacle_density
        self.start = (0, 0)
        self.goals = goals if goals else [(height - 1, width - 1)]  # default: bottom-right goal
        self.generate_obstacles(obstacle_density)

    # Why choose 4-connected instead of 8-connected?
    # -----------------------------------------------
    # In many real-world navigation setups (like warehouse robots,
    # street-following delivery bots, or indoor robots),
    # movement is restricted to four cardinal directions —
    # north, south, east, west.
    #
    # This ensures motion adheres to "grid-aligned" movement,
    # and prevents corner-cutting through obstacles.
    #
    # Each move has a uniform cost = 1.

    def get_neighbors(self, node):
        """Return valid 4-connected neighbors for a given cell (y, x)."""
        y, x = node
        directions = [
            (-1, 0),  # North
            (1, 0),   # South
            (0, -1),  # West
            (0, 1)    # East
        ]

        neighbors = []
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.height and 0 <= nx < self.width:
                if self.grid[ny][nx] == 0:  # free cell
                    neighbors.append(((ny, nx), 1))  # uniform cost
        return neighbors

    def generate_obstacles(self, density=0.70):
        """
        Generate obstacles intelligently to simulate a city environment.
        Roads and blocks alternate in a grid pattern with random noise.

        :param density: approximate fraction of blocked cells
        """
        max_attempts = 100
        attempts = 0

        while attempts < max_attempts:
            attempts += 1

            # Start with an empty grid
            for y in range(self.height):
                for x in range(self.width):
                    self.grid[y][x] = 0

            # Generate obstacle layout with dynamic road spacing
            for y in range(self.height):
                for x in range(self.width):
                    # Dynamically determine road spacing based on grid size
                    road_spacing = max(3, min(self.width, self.height) // 10)

                    # Structured city pattern: main roads every few cells
                    if (y % road_spacing == 0 or x % road_spacing == 0) and not (y == 0 and x == 0):
                        # Roads are mostly open with a bit of randomness
                        self.grid[y][x] = 0 if self.random.random() > (1 - self.obstacle_density) else 1
                    else:
                        # Inner blocks get obstacles based on density
                        if self.random.random() < density:
                            self.grid[y][x] = 1

            # Randomly clear some cells to add alternate routes
            for _ in range(int(self.width * self.height * 0.05)):
                ry = self.random.randint(0, self.height - 1)
                rx = self.random.randint(0, self.width - 1)
                self.grid[ry][rx] = 0

            # Ensure start and goals are open
            self.grid[self.start[0]][self.start[1]] = 0
            for gy, gx in self.goals:
                self.grid[gy][gx] = 0

            # ✅ Check if all goals are reachable from the start
            if all(self.is_reachable(self.start, g) for g in self.goals):
                return  # Success — keep this layout!

        print("⚠️ Warning: Failed to generate a valid reachable map after several attempts.")

    def is_valid(self, y, x):
        """Check if a cell is inside the grid and not an obstacle."""
        return 0 <= y < self.height and 0 <= x < self.width and self.grid[y][x] == 0

    def display(self, start=None, goals=None):
        """
        Print a simple ASCII map of the grid.
        'S' = Start, 'G' = Goal, '█' = obstacle, '.' = free cell.
        """
        for y in range(self.height):
            for x in range(self.width):
                if start and (y, x) == start:
                    print("S", end=" ")
                elif goals and (y, x) in goals:
                    print("G", end=" ")
                elif self.grid[y][x] == 1:
                    print("█", end=" ")
                else:
                    print(".", end=" ")
            print()
        print()

    def display_path(self, path, start, goal):
        """
        Displays the grid in a simple text format.
        Uses:
          S - start
          G - goal
          █ - obstacle
          . - empty road
          o - path
        """
        path_set = set(path) if path else set()

        print(f"\nDisplaying path from start ({start[0]}, {start[1]}) to goal ({goal[0]}, {goal[1]}):")
        for y in range(self.height):
            for x in range(self.width):
                if (y, x) == start:
                    print("S", end=" ")
                elif (y, x) == goal:
                    print("G", end=" ")
                elif (y, x) in path_set:
                    print("o", end=" ")
                elif self.grid[y][x] == 1:
                    print("█", end=" ")
                else:
                    print(".", end=" ")
            print()
        print()

    def display_multi_goal_path(self, path_segments, start, goals):
        """
        Displays the grid showing paths to multiple goals.
        Each segment of the path is shown in sequence.
        Uses:
          S - start
          A/B/C/... - goals
          █ - obstacle
          . - empty road
          o - path to each goal (different symbols per goal)
        """
        # Assign unique symbols for each goal path
        symbols = ['o', '*', '+', 'x', '~', '^', 'Δ']
        goal_marks = [chr(65 + i) for i in range(len(goals))]  # A, B, C, etc.

        # Convert goals list to a dict for labeling
        goal_labels = {tuple(goals[i]): goal_marks[i] for i in range(len(goals))}

        # Combine all path segments
        full_path = set()
        for segment in path_segments:
            full_path.update(segment)

        print("\nDisplaying Multi-Goal GBFS Path:")
        print(f"Start: {start}")
        print(f"Goals: {', '.join([f'{goal_marks[i]}{goals[i]}' for i in range(len(goals))])}\n")

        for y in range(self.height):
            for x in range(self.width):
                if (y, x) == start:
                    print("S", end=" ")
                elif (y, x) in goal_labels:
                    print(goal_labels[(y, x)], end=" ")
                elif (y, x) in full_path:
                    # Choose symbol based on which goal segment it belongs to
                    for i, segment in enumerate(path_segments):
                        if (y, x) in segment:
                            print(symbols[i % len(symbols)], end=" ")
                            break
                elif self.grid[y][x] == 1:
                    print("█", end=" ")
                else:
                    print(".", end=" ")
            print()
        print()

    def modify_obstacle(self, cell, add=True):
        """
        Dynamically add or remove an obstacle.

        :param cell: tuple (y, x)
        :param add: if True, add an obstacle; else remove it
        """
        y, x = cell
        if 0 <= y < self.height and 0 <= x < self.width:
            self.grid[y][x] = 1 if add else 0

    def is_reachable(self, start, goal):
        """
        Use BFS to check if there is any valid path between start and goal.
        Ensures generated maps are navigable.
        """
        if not self.is_valid(*start) or not self.is_valid(*goal):
            return False

        queue = deque([start])
        visited = {start}

        while queue:
            y, x = queue.popleft()
            if (y, x) == goal:
                return True
            for (ny, nx), _ in self.get_neighbors((y, x)):
                if (ny, nx) not in visited:
                    visited.add((ny, nx))
                    queue.append((ny, nx))
        return False

    def display_path_cropped(self, path, alg_name, margin=10):
        """
        Displays a cropped section of the grid around the path for better readability.
        :param path: list of (y, x) tuples
        :param alg_name: name of the algorithm (for title)
        :param margin: cells to expand around path bounds
        """
        if not path:
            print("No path to display.")
            return

        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        min_y, max_y = min(ys), max(ys)
        min_x, max_x = min(xs), max(xs)

        # Crop bounds with margin
        min_y = max(0, min_y - margin)
        max_y = min(self.height - 1, max_y + margin)
        min_x = max(0, min_x - margin)
        max_x = min(self.width - 1, max_x + margin)

        path_set = set(path)
        start = self.start
        goal = self.goals[0] if self.goals else None

        print(f"Cropped {alg_name} Path Display (rows {min_y}-{max_y}, cols {min_x}-{max_x}):")
        print("Legend: S=start, G=goal, o=path, █=obstacle/building, .=road")
        print("-" * ((max_x - min_x + 1) * 2))

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if (y, x) == start:
                    print("S", end=" ")
                elif goal and (y, x) == goal:
                    print("G", end=" ")
                elif (y, x) in path_set:
                    print("o", end=" ")
                elif self.grid[y][x] == 1:
                    print("█", end=" ")
                else:
                    print(".", end=" ")
            print()
        print("-" * ((max_x - min_x + 1) * 2))
        print()