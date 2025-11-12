import heapq
import numpy as np

class Node:
    """A search node: level, g (value so far), weight, solution, h (heuristic), f = g + h."""
    def __init__(self, level, g, weight, solution):
        self.level = level
        self.g = g
        self.weight = weight
        self.solution = solution
        self.h = 0.0
        self.f = 0.0

    def __lt__(self, other):
        """Max-heap: return True if self.f > other.f (higher f = higher priority)."""
        # Max-heap (prioritize nodes with higher f value)
        return self.f > other.f


def calculate_heuristic(node, weights, values, capacity, n_items):
    """Calculate h(x): heuristic estimate of remaining value using fractional knapsack."""
    if node.weight >= capacity:
        return 0.0  # Cannot add anything more

    h = 0.0
    total_weight = node.weight
    idx = node.level + 1

    # Greedily add complete items if they fit
    while idx < n_items and total_weight + weights[idx] <= capacity:
        total_weight += weights[idx]
        h += values[idx]
        idx += 1

    # If space remains, add fractional value of next item
    if idx < n_items and weights[idx] > 0:
        remain = capacity - total_weight
        h += values[idx] * (remain / weights[idx])

    return h


def a_star_search(context, heuristic_func=None):
    """A* search for 0/1 Knapsack using fractional knapsack heuristic."""
    weights = context['weights']
    values = context['values']
    capacity = context['capacity']
    n_items = len(weights)

    # Handle edge cases
    if n_items == 0 or capacity <= 0:
        return np.zeros(0, dtype=int), 0

    # Sort items by value/weight ratio (greedy heuristic)
    items = sorted([(v / w, v, w, i) for i, (v, w) in enumerate(zip(values, weights)) if w > 0], reverse=True)
    sorted_values = [v for _, v, _, _ in items]
    sorted_weights = [w for _, _, w, _ in items]
    original_idx = [i for _, _, _, i in items]
    n_sorted = len(sorted_weights)

    # If no valid items (all weights == 0), return empty solution
    if n_sorted == 0:
        return np.zeros(n_items, dtype=int), 0

    # Initialize
    best_value = 0
    best_solution = [0] * n_items

    pq = []
    counter = 0
    max_nodes = 100_000

    # Create root node: calculate h then f = g + h
    root = Node(level=-1, g=0, weight=0, solution=[0] * n_items)
    root.h = calculate_heuristic(root, sorted_weights, sorted_values, capacity, n_sorted)
    root.f = root.g + root.h
    heapq.heappush(pq, (-root.f, counter, root))
    counter += 1

    nodes_explored = 0

    # A* search loop
    while pq and nodes_explored < max_nodes:
        _, _, current = heapq.heappop(pq)
        nodes_explored += 1

        # Prune if f(x) <= best_value
        if current.f <= best_value:
            continue
        next_level = current.level + 1
        # Stop if we've considered all items
        if next_level >= n_sorted:
            continue

        item_weight = sorted_weights[next_level]
        item_value = sorted_values[next_level]
        orig_idx = original_idx[next_level]

        # Branch 1: Include item
        if current.weight + item_weight <= capacity:
            new_solution = current.solution.copy()
            new_solution[orig_idx] = 1

            child = Node(next_level, current.g + item_value, current.weight + item_weight, new_solution)
            child.h = calculate_heuristic(child, sorted_weights, sorted_values, capacity, n_sorted)
            child.f = child.g + child.h

            # Update best solution
            if child.g > best_value:
                best_value = child.g
                best_solution = new_solution.copy()

            # Add to queue if promising
            if child.f > best_value:
                heapq.heappush(pq, (-child.f, counter, child))
                counter += 1

        # Branch 2: Exclude item
        new_solution = current.solution.copy()
        new_solution[orig_idx] = 0

        child = Node(next_level, current.g, current.weight, new_solution)
        child.h = calculate_heuristic(child, sorted_weights, sorted_values, capacity, n_sorted)
        child.f = child.g + child.h

        if child.g > best_value:
            best_value = child.g
            best_solution = new_solution.copy()

        if child.f > best_value:
            heapq.heappush(pq, (-child.f, counter, child))
            counter += 1

    return np.array(best_solution, dtype=int), best_value
