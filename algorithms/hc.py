import numpy as np

def hill_climbing_continuous(obj_func, bounds, n_dim, max_iter, step_size=0.1, **kwargs):
    """
    Hill Climbing algorithm for continuous optimization.
    
    Parameters:
    -----------
    obj_func : function
        Objective function to minimize
    bounds : array-like
        Bounds for each dimension [[min, max], ...]
    n_dim : int
        Number of dimensions
    max_iter : int
        Maximum number of iterations
    step_size : float
        Step size for generating neighbors
    
    Returns:
    --------
    best_solution : ndarray
        Best solution found (real-valued vector)
    best_fitness : float
        Best fitness value
    history : list
        History of best fitness values
    """
    min_b, max_b = np.asarray(bounds).T
    
    # Initialize random solution
    current_solution = min_b + (max_b - min_b) * np.random.rand(n_dim)
    current_fitness = obj_func(current_solution)
    
    # Track best solution
    best_solution = current_solution.copy()
    best_fitness = current_fitness
    
    # History tracking
    history = [best_fitness]
    
    # Hill climbing main loop
    for iteration in range(max_iter):
        # Generate neighbor by adding random perturbation
        perturbation = np.random.randn(n_dim) * step_size * (max_b - min_b)
        neighbor_solution = current_solution + perturbation
        
        # Apply bounds
        neighbor_solution = np.clip(neighbor_solution, min_b, max_b)
        
        # Evaluate neighbor
        neighbor_fitness = obj_func(neighbor_solution)
        
        # Accept if better (for minimization)
        if neighbor_fitness < current_fitness:
            current_solution = neighbor_solution
            current_fitness = neighbor_fitness
            
            # Update best
            if current_fitness < best_fitness:
                best_solution = current_solution.copy()
                best_fitness = current_fitness
        
        # Record history
        history.append(best_fitness)
    
    return best_solution, best_fitness, history


def hill_climbing_discrete(obj_func, context, n_dim, max_iter, **kwargs):
    """
    Hill Climbing algorithm for discrete optimization.
    
    Parameters:
    -----------
    obj_func : function
        Objective function to maximize (will be negated internally for minimization)
    context : dict
        Context dict with problem data
    n_dim : int
        Number of dimensions
    max_iter : int
        Maximum number of iterations
    
    Returns:
    --------
    best_solution : ndarray
        Best solution found (binary vector)
    best_fitness : float
        Best fitness value (maximization, not negated)
    history : list
        History of best fitness values
    """
    # # Initialize zeros solution
    # current_solution = np.zeros(n_dim)

    # Initialize random binary solution until valid
    while True:
        current_solution = np.random.randint(0, 2, n_dim)
        total_weight = np.dot(current_solution, context['weights'])
        if total_weight <= context['capacity']:
            break
    
    # Negate fitness for maximization (Knapsack is maximization, but we minimize)
    current_fitness = -obj_func(current_solution, context)
    
    # Track best solution
    best_solution = current_solution.copy()
    best_fitness = current_fitness
    
    # History tracking
    history = [best_fitness]
    
    # Hill climbing main loop
    for iteration in range(max_iter):
        # Generate neighbor by flipping one random bit
        neighbor_solution = current_solution.copy()
        bit_to_flip = np.random.randint(0, n_dim)
        neighbor_solution[bit_to_flip] = 1 - neighbor_solution[bit_to_flip]
        
        # Evaluate neighbor
        # Negate fitness for maximization (Knapsack is maximization, but we minimize)
        neighbor_fitness = -obj_func(neighbor_solution, context)
        
        # Accept if better (for minimization)
        if neighbor_fitness < current_fitness:
            current_solution = neighbor_solution
            current_fitness = neighbor_fitness
            
            # Update best
            if current_fitness < best_fitness:
                best_solution = current_solution.copy()
                best_fitness = current_fitness
        
        # Record history
        history.append(best_fitness)
    
    # Return negated fitness (convert back to maximization)
    best_fitness = -best_fitness
    # Convert history to maximization (all values should be positive)
    history = [-h for h in history]
    
    return best_solution, best_fitness, history
