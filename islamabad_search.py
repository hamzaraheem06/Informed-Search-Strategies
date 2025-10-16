import os
import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
import math
import random
from collections import deque
import folium  # For interactive maps

# Assuming grid.py and search.py are in the same directory
from grid import CityGrid
from search import *
from heuristics import *


# ============================
# Helper: find nearest road cell (4-connected)
# ============================
def find_nearest_road_cell(grid, point, search_radius=20):
    """
    If the start/goal lies on an obstacle, search nearby for a drivable cell using BFS-like expansion.
    Uses 4-connected for consistency.
    """
    y, x = point
    h, w = grid.shape
    print(f"   Checking cell ({y}, {x}): value = {grid[y, x] if 0 <= y < h and 0 <= x < w else 'OOB'}")
    if 0 <= y < h and 0 <= x < w and grid[y, x] == 0:
        return point
    queue = deque([(y, x, 0)])
    visited = set([(y, x)])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connected

    while queue:
        cy, cx, dist = queue.popleft()
        if 0 <= cy < h and 0 <= cx < w and grid[cy, cx] == 0:
            print(f"   Snapped to nearest road at ({cy}, {cx}) (dist {dist})")
            return (cy, cx)
        if dist >= search_radius:
            break
        for dy, dx in directions:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in visited:
                visited.add((ny, nx))
                queue.append((ny, nx, dist + 1))
    print(f"   Warning: No free cell found within radius {search_radius}; using original {point}")
    return point


# ============================
# Convert OSM Graph to Grid (with 4-neighbor dilation)
# ============================
def graph_to_grid(G, resolution=0.0003):  # ~30m/cell for better detail
    """
    Convert OSM street network graph into a binary grid.
    Marks nodes and interpolated edge points.
    Applies 4-neighbor dilation to connect gaps without diagonals.
    """
    nodes = np.array([[data['y'], data['x']] for _, data in G.nodes(data=True)])
    if len(nodes) == 0:
        raise ValueError("Empty graph provided.")
    min_lat, min_lon = nodes[:, 0].min(), nodes[:, 1].min()
    max_lat, max_lon = nodes[:, 0].max(), nodes[:, 1].max()

    height = int((max_lat - min_lat) / resolution) + 1
    width = int((max_lon - min_lon) / resolution) + 1
    grid = np.ones((height, width), dtype=int)

    # Mark nodes as free
    for _, data in G.nodes(data=True):
        y_idx = int((data['y'] - min_lat) / resolution)
        x_idx = int((data['x'] - min_lon) / resolution)
        if 0 <= y_idx < height and 0 <= x_idx < width:
            grid[y_idx, x_idx] = 0

    # Mark edges with more interpolation points
    for u, v, _ in G.edges(data=True):
        lat1, lon1 = G.nodes[u]['y'], G.nodes[u]['x']
        lat2, lon2 = G.nodes[v]['y'], G.nodes[v]['x']
        # Approximate geodesic distance
        deg_dist = np.hypot(lat2 - lat1, (lon2 - lon1) * np.cos(np.radians((lat1 + lat2) / 2)))
        num_points = max(5, int(deg_dist / resolution) + 1)
        for t in np.linspace(0, 1, num_points):
            lat = lat1 + t * (lat2 - lat1)
            lon = lon1 + t * (lon2 - lon1)
            y_idx = int((lat - min_lat) / resolution)
            x_idx = int((lon - min_lon) / resolution)
            if 0 <= y_idx < height and 0 <= x_idx < width:
                grid[y_idx, x_idx] = 0

    # 4-neighbor dilation: expand roads to connect adjacent gaps
    new_grid = grid.copy()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for y in range(height):
        for x in range(width):
            if grid[y, x] == 0:  # Road cell
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        new_grid[ny, nx] = 0
    grid = new_grid

    road_count = np.sum(grid == 0)
    road_pct = 100 * road_count / (height * width)
    print(f"   Grid: {height}x{width}, roads: {road_count} ({road_pct:.1f}%)")
    return grid, (min_lat, min_lon, max_lat, max_lon)


# ============================
# Lat/Lon ↔ Grid Index
# ============================
def latlon_to_grid(lat, lon, bounds, resolution=0.0003):
    min_lat, min_lon, _, _ = bounds
    y_idx = int((lat - min_lat) / resolution)
    x_idx = int((lon - min_lon) / resolution)
    return (y_idx, x_idx)


def grid_to_latlon(y, x, bounds, resolution=0.0003):
    min_lat, min_lon, _, _ = bounds
    lat = min_lat + y * resolution
    lon = min_lon + x * resolution
    return (lat, lon)


# ============================
# Display Interactive Map with Folium (Google Maps-like)
# ============================
def display_interactive_map(G, path, bounds, resolution=0.0003, title="Path on Islamabad Map", start_input="",
                            goal_input=""):
    """
    Create an interactive Folium map with path overlay, markers, and street labels.
    Saves to 'route_map.html' for browser viewing.
    """
    print("\n🗺️ Generating interactive map (Google Maps-style) – open 'route_map.html' in your browser!")

    # Convert path to lat/lon
    latlon_path = [grid_to_latlon(y, x, bounds, resolution) for y, x in path]
    latitudes = [p[0] for p in latlon_path]
    longitudes = [p[1] for p in latlon_path]

    # Center map on path midpoint
    center_lat = np.mean(latitudes)
    center_lon = np.mean(longitudes)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15,
                   tiles='OpenStreetMap')  # OSM tiles with street labels

    # Add path as red polyline
    folium.PolyLine(
        locations=latlon_path,
        color='red',
        weight=5,
        opacity=0.8,
        popup=f'Route: {start_input} to {goal_input}'
    ).add_to(m)

    # Add start marker (green)
    folium.Marker(
        location=latlon_path[0],
        popup=f'Start: {start_input}',
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)

    # Add goal marker (blue)
    folium.Marker(
        location=latlon_path[-1],
        popup=f'Goal: {goal_input}',
        icon=folium.Icon(color='blue', icon='stop')
    ).add_to(m)

    # Fit bounds to path with margin
    south, west = min(latitudes) - 0.002, min(longitudes) - 0.002
    north, east = max(latitudes) + 0.002, max(longitudes) + 0.002
    m.fit_bounds([[south, west], [north, east]])

    # Save to HTML
    output_file = 'route_map.html'
    m.save(output_file)
    print(f"   Saved interactive map to '{output_file}' – zoom/pan to explore streets and route!")


# ============================
# Run Algorithm (with debug)
# ============================
def run_algorithm(start_idx, goal_idx, grid, algorithm_name, heuristic):
    """
    Run the specified algorithm and return results.
    """
    sy, sx = start_idx
    gy, gx = goal_idx
    h, w = grid.shape
    print(f"   Start cell ({sy}, {sx}): {grid[sy, sx] if 0 <= sy < h and 0 <= sx < w else 'OOB'}")
    print(f"   Goal cell ({gy}, {gx}): {grid[gy, gx] if 0 <= gy < h and 0 <= gx < w else 'OOB'}")
    if algorithm_name == "Bi-A*":
        path, cost, expanded = bidirectional_a_star(start_idx, goal_idx, grid, heuristic)
    # elif algorithm_name == "GBFS":
    #     path, expanded, _ = greedy_best_first_search(start_idx, goal_idx, grid, heuristic)
    #     cost = len(path) - 1 if path else float('inf')
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    return path, cost, expanded


# ============================
# MAIN
# ============================
if __name__ == "__main__":
    print("=== Islamabad Real-World Pathfinding Demo ===")
    print("Using OpenStreetMap data for Islamabad roads (4-connected).\n")

    # Cache paths - Delete islamabad_grid.npz to regenerate with new settings
    graph_path = "islamabad_graph.graphml"
    grid_path = "islamabad_grid.npz"
    resolution = 0.0003  # Finer resolution for connectivity

    # Load/Download graph
    if not os.path.exists(graph_path):
        print("🌐 Downloading Islamabad OSM data...")
        geolocator = Nominatim(user_agent="islamabad_pathfinder")
        try:
            islamabad = geolocator.geocode("Islamabad, Pakistan")
            bbox = islamabad.raw['boundingbox']
            south, north, west, east = map(float, bbox)
            margin = 0.02
            north += margin
            south -= margin
            east += margin
            west -= margin
            G = ox.graph_from_bbox((west, south, east, north), network_type="drive")
            ox.save_graphml(G, graph_path)
            print("✅ Graph saved.")
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"❌ Geocoding error: {e}. Using approximate bbox.")
            north = 33.8538118 + 0.02
            south = 33.5338118 - 0.02
            east = 73.2251511 + 0.02
            west = 72.9051511 - 0.02
            G = ox.graph_from_bbox((west, south, east, north), network_type="drive")
            ox.save_graphml(G, graph_path)
    else:
        print("📂 Loading cached graph...")
        G = ox.load_graphml(graph_path)

    # Load/Convert to grid
    if not os.path.exists(grid_path):
        print("🧭 Converting to grid...")
        grid, bounds = graph_to_grid(G, resolution)
        np.savez(grid_path, grid=grid, bounds=bounds, resolution=resolution)
        print("✅ Grid saved.")
    else:
        print("📂 Loading cached grid...")
        data = np.load(grid_path)
        grid = data['grid']
        bounds = data['bounds']
        resolution = data.get('resolution', 0.0003)

    print(f"Grid size: {grid.shape[0]} x {grid.shape[1]} cells")

    # User inputs
    geolocator = Nominatim(user_agent="islamabad_pathfinder")
    max_retries = 3
    start = goal = None
    start_input = goal_input = ""

    for attempt in range(max_retries):
        start_input = input("\nEnter start location (e.g., 'F-8 Markaz, Islamabad'): ").strip()
        goal_input = input("Enter goal location (e.g., 'Blue Area, Islamabad'): ").strip()
        try:
            start = geolocator.geocode(start_input)
            goal = geolocator.geocode(goal_input)
            if start and goal:
                break
            print("⚠️ Could not geocode. Try again.")
        except (GeocoderTimedOut, GeocoderServiceError):
            print("⚠️ Timeout. Try again.")
            time.sleep(1)

    if not start or not goal:
        print("❌ Failed to geocode. Exiting.")
        exit()

    print(f"\n📍 Start: {start.address}")
    print(f"📍 Goal: {goal.address}")
    print(f"   Start coords: {start.latitude:.4f}, {start.longitude:.4f}")
    print(f"   Goal coords: {goal.latitude:.4f}, {goal.longitude:.4f}")

    # Convert to grid indices
    start_idx = latlon_to_grid(start.latitude, start.longitude, bounds, resolution)
    goal_idx = latlon_to_grid(goal.latitude, goal.longitude, bounds, resolution)

    h, w = grid.shape
    start_idx = (max(0, min(h - 1, start_idx[0])), max(0, min(w - 1, start_idx[1])))
    goal_idx = (max(0, min(h - 1, goal_idx[0])), max(0, min(w - 1, goal_idx[1])))

    # Adjust if needed
    start_idx = find_nearest_road_cell(grid, start_idx)
    goal_idx = find_nearest_road_cell(grid, goal_idx)

    print(f"\nStart grid index: {start_idx}")
    print(f"Goal grid index: {goal_idx}")

    # Create CityGrid
    env = CityGrid(external_grid=grid.tolist())
    env.start = start_idx
    env.goals = [goal_idx]

    # Algorithms (using Euclidean from heuristics)
    algorithms = [
        ("Bi-A*", manhattan)
    ]

    for alg_name, heuristic in algorithms:
        print(f"\n🚀 Running {alg_name} Search...")
        path, cost, expanded = run_algorithm(start_idx, goal_idx, grid, alg_name, heuristic)

        if path:
            print(f"✅ {alg_name} Path found! Length = {len(path) - 1}, Cost ≈ {cost:.2f}, Nodes expanded = {expanded}")
            # Use cropped display
            env.display_path_cropped(path, alg_name, margin=15)
            display_interactive_map(G, path, bounds, resolution, f"{alg_name} Path", start_input, goal_input)
        else:
            print(f"❌ {alg_name}: No path found.")

    print("\n=== Demo Complete ===")
    print("Note: If no path, delete 'islamabad_grid.npz' to regenerate with finer grid.")