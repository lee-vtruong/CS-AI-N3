import numpy as np
import heapq

class Node:
    """Node for A* search tree."""
    
    def __init__(self, level, value, weight, solution, bound):
        """
        Parameters:
        -----------
        level : int
            Current item index being considered
        value : float
            Current total value
        weight : float
            Current total weight
        solution : list
            Current solution (binary decisions)
        bound : float
            Upper bound on maximum value achievable from this node
        """
        self.level = level
        self.value = value
        self.weight = weight
        self.solution = solution
        self.bound = bound
    
    def __lt__(self, other):
        # For max-heap behavior in min-heap (negate for priority)
        # We want to explore nodes with higher bound first
        return self.bound > other.bound

def calculate_bound(node, n_items, capacity, weights, values):
    """
    Calculate upper bound on maximum value achievable from this node.
    Uses fractional knapsack relaxation.
    """
    if node.weight >= capacity:
        return 0
    
    bound = node.value
    total_weight = node.weight
    level = node.level + 1
    
    # Calculate value-to-weight ratios for remaining items
    remaining_items = []
    for i in range(level, n_items):
        if weights[i] > 0:
            ratio = values[i] / weights[i]
            remaining_items.append((ratio, values[i], weights[i], i))
    
    # Sort by ratio (descending)
    remaining_items.sort(reverse=True)
    
    # Add items greedily (fractional allowed for bound)
    for ratio, value, weight, idx in remaining_items:
        if total_weight + weight <= capacity:
            total_weight += weight
            bound += value
        else:
            # Add fraction of item
            remaining_capacity = capacity - total_weight
            bound += ratio * remaining_capacity
            break
    
    return bound

def a_star_search(context, heuristic_func=None):
    """
    A* Search for Knapsack problem.
    
    Parameters:
    -----------
    context : dict
        Problem context with 'weights', 'values', 'capacity'
    heuristic_func : function (optional)
        Custom heuristic function (not used in current implementation,
        using fractional knapsack bound instead)
    
    Returns:
    --------
    best_solution : ndarray
        Best solution found (binary vector)
    best_fitness : float
        Best fitness value (total value)
    """
    weights = context['weights']
    values = context['values']
    capacity = context['capacity']
    n_items = len(weights)
    
    # Initialize best solution
    best_solution = np.zeros(n_items, dtype=int)
    best_value = 0
    
    # Priority queue (min-heap, but we'll negate priorities for max behavior)
    pq = []
    
    # Create root node
    root = Node(level=-1, value=0, weight=0, solution=[], 
                bound=calculate_bound(Node(-1, 0, 0, [], 0), n_items, capacity, weights, values))
    
    heapq.heappush(pq, root)
    
    nodes_explored = 0
    max_nodes = 100000  # Limit to prevent infinite loops
    
    # A* search
    while pq and nodes_explored < max_nodes:
        current = heapq.heappop(pq)
        nodes_explored += 1
        
        # If bound is less than current best, skip this branch
        if current.bound <= best_value:
            continue
        
        # If we've processed all items
        if current.level == n_items - 1:
            continue
        
        next_level = current.level + 1
        
        # --- Branch 1: Include next item ---
        if current.weight + weights[next_level] <= capacity:
            solution_include = current.solution + [1]
            value_include = current.value + values[next_level]
            weight_include = current.weight + weights[next_level]
            
            # Update best if this is a complete solution
            if next_level == n_items - 1:
                if value_include > best_value:
                    best_value = value_include
                    best_solution = np.array(solution_include + [0] * (n_items - len(solution_include)), dtype=int)
            else:
                # Create new node
                node_include = Node(next_level, value_include, weight_include, 
                                  solution_include, 0)
                node_include.bound = calculate_bound(node_include, n_items, capacity, 
                                                     weights, values)
                
                # Add to queue if promising
                if node_include.bound > best_value:
                    heapq.heappush(pq, node_include)
                
                # Update best if better
                if value_include > best_value:
                    best_value = value_include
                    temp_solution = solution_include + [0] * (n_items - len(solution_include))
                    best_solution = np.array(temp_solution, dtype=int)
        
        # --- Branch 2: Exclude next item ---
        solution_exclude = current.solution + [0]
        value_exclude = current.value
        weight_exclude = current.weight
        
        # Update best if this is a complete solution
        if next_level == n_items - 1:
            if value_exclude > best_value:
                best_value = value_exclude
                best_solution = np.array(solution_exclude + [0] * (n_items - len(solution_exclude)), dtype=int)
        else:
            # Create new node
            node_exclude = Node(next_level, value_exclude, weight_exclude, 
                              solution_exclude, 0)
            node_exclude.bound = calculate_bound(node_exclude, n_items, capacity, 
                                                 weights, values)
            
            # Add to queue if promising
            if node_exclude.bound > best_value:
                heapq.heappush(pq, node_exclude)
    
    return best_solution, best_value

