import numpy as np

def hill_climbing(obj_func, context_or_bounds, n_dim, pop_size, max_iter, problem_type='continuous', step_size=0.1, **kwargs):
    """
    Hill Climbing algorithm for continuous and discrete optimization.
    
    Parameters:
    -----------
    obj_func : function
        Objective function to optimize
    context_or_bounds : array-like or dict
        - If 'continuous': bounds for each dimension [[min, max], ...]
        - If 'discrete': context dict with problem data
    n_dim : int
        Number of dimensions
    pop_size : int
        Population size (for consistency, not used in HC)
    max_iter : int
        Maximum number of iterations
    problem_type : str
        'continuous' (default) or 'discrete'
    step_size : float
        Step size for generating neighbors (continuous only)
    
    Returns:
    --------
    best_solution : ndarray
        Best solution found
    best_fitness : float
        Best fitness value
    history : list
        History of best fitness values
    """
    # Initialize based on problem type
    if problem_type == 'continuous':
        bounds = context_or_bounds
        min_b, max_b = np.asarray(bounds).T
        
        # Initialize random solution
        current_solution = min_b + (max_b - min_b) * np.random.rand(n_dim)
        current_fitness = obj_func(current_solution)
    else:  # 'discrete'
        context = context_or_bounds
        
        # Initialize random binary solution
        current_solution = np.random.randint(0, 2, n_dim)
        # Negate fitness for maximization (Knapsack is maximization, but we minimize)
        current_fitness = -obj_func(current_solution, context)
    
    # Track best solution
    best_solution = current_solution.copy()
    best_fitness = current_fitness
    
    # History tracking
    history = [best_fitness]
    
    # Hill climbing main loop
    for iteration in range(max_iter):
        if problem_type == 'continuous':
            # Generate neighbor by adding random perturbation
            perturbation = np.random.randn(n_dim) * step_size * (max_b - min_b)
            neighbor_solution = current_solution + perturbation
            
            # Apply bounds
            neighbor_solution = np.clip(neighbor_solution, min_b, max_b)
            
            # Evaluate neighbor
            neighbor_fitness = obj_func(neighbor_solution)
        else:  # 'discrete'
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
        if problem_type == 'discrete':
            # Store negated fitness in history (for consistency)
            history.append(best_fitness)
        else:
            history.append(best_fitness)
    
    # Return negated fitness for discrete (convert back to maximization)
    if problem_type == 'discrete':
        best_fitness = -best_fitness
    
    return best_solution, best_fitness, history

