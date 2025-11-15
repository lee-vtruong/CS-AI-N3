import numpy as np

def generate_knapsack_data(n_items, seed=42):
    """Generate random data for the Knapsack problem."""
    np.random.seed(seed)
    weights = np.random.randint(1, 20, n_items)
    values = np.random.randint(10, 100, n_items)
    # Set capacity to 40% of total weight
    capacity = int(np.sum(weights) * 0.4)
    return weights, values, capacity, n_items

def knapsack_fitness(solution, context):
    """
    Fitness function for Knapsack problem.
    'solution' is a binary vector (n_items,).
    'context' is a dict {'weights', 'values', 'capacity'}.
    """
    weights = context['weights']
    values = context['values']
    capacity = context['capacity']
    
    total_weight = np.dot(solution, weights)
    total_value = np.dot(solution, values)
    
    # Apply penalty if capacity is exceeded
    if total_weight > capacity:
        return 0  # Invalid solution, fitness = 0 (penalty for minimization)
    
    return -total_value

