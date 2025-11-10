import numpy as np

def levy_flight(n_dim, beta=1.5):
    """
    Generate step using Levy flight distribution.
    """
    sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
             (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
    
    u = np.random.randn(n_dim) * sigma
    v = np.random.randn(n_dim)
    
    step = u / np.abs(v)**(1 / beta)
    return step

def cuckoo_search(obj_func, context_or_bounds, n_dim, pop_size, max_iter, problem_type='continuous', pa=0.25, beta=1.5, **kwargs):
    """
    Cuckoo Search (CS) algorithm.
    
    Parameters:
    -----------
    obj_func : function
        Objective function to minimize
    context_or_bounds : array-like or dict
        - If 'continuous': bounds for each dimension [[min, max], ...]
        - If 'discrete': context dict with problem data
    n_dim : int
        Number of dimensions
    pop_size : int
        Population size (number of nests)
    max_iter : int
        Maximum number of iterations
    problem_type : str
        'continuous' (default) or 'discrete'
    pa : float
        Probability of discovering alien eggs (0 to 1)
    beta : float
        Parameter for Levy flight distribution
    
    Returns:
    --------
    best_solution : ndarray
        Best solution found
    best_fitness : float
        Best fitness value
    history : list
        History of best fitness values
    """
    # Initialize bounds (for continuous) or context (for discrete)
    if problem_type == 'continuous':
        bounds = context_or_bounds
        min_b, max_b = np.asarray(bounds).T
    else:
        context = context_or_bounds
        # For discrete, we'll use arbitrary bounds for initialization
        min_b = np.full(n_dim, -5.0)
        max_b = np.full(n_dim, 5.0)
    
    # Initialize nests (solutions)
    nests = min_b + (max_b - min_b) * np.random.rand(pop_size, n_dim)
    
    # Evaluate fitness
    if problem_type == 'discrete':
        # Evaluate initial fitness with binarization
        # Negate fitness for maximization (Knapsack is maximization, but we minimize)
        fitness = np.array([
            -obj_func((1 / (1 + np.exp(-nest)) > 0.5).astype(int), context) 
            for nest in nests
        ])
    else:
        fitness = np.array([obj_func(nest) for nest in nests])
    
    # Find best nest
    best_idx = np.argmin(fitness)
    best_solution = nests[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    # History tracking
    history = [best_fitness]
    
    # CS main loop
    for iteration in range(max_iter):
        # Generate new solutions using Levy flights
        for i in range(pop_size):
            # Generate a cuckoo via Levy flights
            step_size = 0.01 * levy_flight(n_dim, beta)
            new_nest = nests[i] + step_size * (nests[i] - best_solution)
            
            # Evaluate fitness based on problem type
            if problem_type == 'discrete':
                # Apply sigmoid to convert continuous to binary
                probabilities = 1 / (1 + np.exp(-new_nest))
                binary_solution = (probabilities > 0.5).astype(int)
                # Negate fitness for maximization (Knapsack is maximization, but we minimize)
                new_fitness = -obj_func(binary_solution, context)
            else:  # 'continuous'
                # Apply bounds
                new_nest = np.clip(new_nest, min_b, max_b)
                new_fitness = obj_func(new_nest)
            
            # Choose a random nest j
            j = np.random.randint(pop_size)
            
            # Replace nest j if new nest is better
            if new_fitness < fitness[j]:
                nests[j] = new_nest
                fitness[j] = new_fitness
                
                # Update global best
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_solution = new_nest.copy()
        
        # Abandon a fraction (pa) of worst nests
        n_abandon = int(pa * pop_size)
        
        # Get indices of worst nests
        worst_indices = np.argsort(fitness)[-n_abandon:]
        
        # Replace worst nests with new random solutions
        for idx in worst_indices:
            # Generate new random solution
            nests[idx] = min_b + (max_b - min_b) * np.random.rand(n_dim)
            if problem_type == 'discrete':
                # Evaluate with binarization
                probabilities = 1 / (1 + np.exp(-nests[idx]))
                binary_solution = (probabilities > 0.5).astype(int)
                # Negate fitness for maximization (Knapsack is maximization, but we minimize)
                fitness[idx] = -obj_func(binary_solution, context)
            else:
                fitness[idx] = obj_func(nests[idx])
            
            # Update global best if necessary
            if fitness[idx] < best_fitness:
                best_fitness = fitness[idx]
                best_solution = nests[idx].copy()
        
        # Record history
        history.append(best_fitness)
    
    # Return binary solution for discrete problems, continuous for continuous
    if problem_type == 'discrete':
        probabilities = 1 / (1 + np.exp(-best_solution))
        best_solution = (probabilities > 0.5).astype(int)
        # Return negated fitness (convert back to maximization)
        best_fitness = -best_fitness
    
    return best_solution, best_fitness, history

