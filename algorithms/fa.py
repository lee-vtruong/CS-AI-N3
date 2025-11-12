import numpy as np

def firefly_algorithm_continuous(obj_func, bounds, n_dim, pop_size, max_iter, alpha=0.5, beta0=1.0, gamma=0.95, **kwargs):
    """
    Firefly Algorithm (FA) for continuous optimization.
    
    Parameters:
    -----------
    obj_func : function
        Objective function to minimize
    bounds : array-like
        Bounds for each dimension [[min, max], ...]
    n_dim : int
        Number of dimensions
    pop_size : int
        Population size (number of fireflies)
    max_iter : int
        Maximum number of iterations
    alpha : float
        Randomization parameter (step size)
    beta0 : float
        Attractiveness at r=0
    gamma : float
        Light absorption coefficient
    
    Returns:
    --------
    best_solution : ndarray
        Best solution found (real-valued vector)
    best_fitness : float
        Best fitness value
    history : list
        History of best fitness values
    """
    # Initialize bounds
    min_b, max_b = np.asarray(bounds).T
    
    # Initialize firefly positions
    fireflies = min_b + (max_b - min_b) * np.random.rand(pop_size, n_dim)
    
    # Evaluate fitness (light intensity)
    fitness = np.array([obj_func(f) for f in fireflies])
    
    # Find best firefly
    best_idx = np.argmin(fitness)
    best_solution = fireflies[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    # History tracking
    history = [best_fitness]
    
    # FA main loop
    for iteration in range(max_iter):
        # Update alpha (decreasing randomization over time)
        alpha_t = alpha * (0.95 ** iteration)
        
        # Move fireflies towards brighter ones
        for i in range(pop_size):
            for j in range(pop_size):
                # If firefly j is brighter than firefly i
                if fitness[j] < fitness[i]:
                    # Calculate distance
                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    
                    # Calculate attractiveness (beta decreases with distance)
                    beta = beta0 * np.exp(-gamma * r**2)
                    
                    # Move firefly i towards firefly j
                    fireflies[i] = fireflies[i] + beta * (fireflies[j] - fireflies[i]) + \
                                   alpha_t * (np.random.rand(n_dim) - 0.5)
                    
                    # Apply bounds
                    fireflies[i] = np.clip(fireflies[i], min_b, max_b)
                    fitness[i] = obj_func(fireflies[i])
                    
                    # Update global best
                    if fitness[i] < best_fitness:
                        best_fitness = fitness[i]
                        best_solution = fireflies[i].copy()
        
        # Record history
        history.append(best_fitness)
    
    return best_solution, best_fitness, history


def firefly_algorithm_discrete(obj_func, context, n_dim, pop_size, max_iter, alpha=0.5, beta0=1.0, gamma=0.95, **kwargs):
    """
    Firefly Algorithm (FA) for discrete optimization.
    
    Parameters:
    -----------
    obj_func : function
        Objective function to maximize (will be negated internally for minimization)
    context : dict
        Context dict with problem data
    n_dim : int
        Number of dimensions
    pop_size : int
        Population size (number of fireflies)
    max_iter : int
        Maximum number of iterations
    alpha : float
        Randomization parameter (step size)
    beta0 : float
        Attractiveness at r=0
    gamma : float
        Light absorption coefficient
    
    Returns:
    --------
    best_solution : ndarray
        Best solution found (binary vector)
    best_fitness : float
        Best fitness value (maximization, not negated)
    history : list
        History of best fitness values
    """
    # For discrete, we'll use arbitrary bounds for initialization
    min_b = np.full(n_dim, -5.0)
    max_b = np.full(n_dim, 5.0)
    
    # Initialize firefly positions
    fireflies = min_b + (max_b - min_b) * np.random.rand(pop_size, n_dim)
    
    # Evaluate fitness (light intensity) with binarization (numerically stable sigmoid)
    # Negate fitness for maximization (Knapsack is maximization, but we minimize)
    def stable_sigmoid_binarize(x):
        # Clip x to prevent overflow in exp
        x_clipped = np.clip(x, -500, 500)
        probs = 1 / (1 + np.exp(-x_clipped))
        return (probs > 0.5).astype(int)
    
    fitness = np.array([
        -obj_func(stable_sigmoid_binarize(f), context) 
        for f in fireflies
    ])
    
    # Find best firefly
    best_idx = np.argmin(fitness)
    best_solution = fireflies[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    # History tracking
    history = [best_fitness]
    
    # FA main loop
    for iteration in range(max_iter):
        # Update alpha (decreasing randomization over time)
        alpha_t = alpha * (0.95 ** iteration)
        
        # Move fireflies towards brighter ones
        for i in range(pop_size):
            for j in range(pop_size):
                # If firefly j is brighter than firefly i
                if fitness[j] < fitness[i]:
                    # Calculate distance
                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    
                    # Calculate attractiveness (beta decreases with distance)
                    beta = beta0 * np.exp(-gamma * r**2)
                    
                    # Move firefly i towards firefly j
                    fireflies[i] = fireflies[i] + beta * (fireflies[j] - fireflies[i]) + \
                                   alpha_t * (np.random.rand(n_dim) - 0.5)
                    
                    # Apply sigmoid to convert continuous to binary (numerically stable)
                    # Clip x to prevent overflow in exp
                    x = fireflies[i]
                    x_clipped = np.clip(x, -500, 500)
                    probabilities = 1 / (1 + np.exp(-x_clipped))
                    binary_solution = (probabilities > 0.5).astype(int)
                    # Negate fitness for maximization (Knapsack is maximization, but we minimize)
                    fitness[i] = -obj_func(binary_solution, context)
                    
                    # Update global best
                    if fitness[i] < best_fitness:
                        best_fitness = fitness[i]
                        best_solution = fireflies[i].copy()
        
        # Record history
        history.append(best_fitness)
    
    # Return binary solution (numerically stable sigmoid)
    # Clip x to prevent overflow in exp
    x = best_solution
    x_clipped = np.clip(x, -500, 500)
    probabilities = 1 / (1 + np.exp(-x_clipped))
    best_solution = (probabilities > 0.5).astype(int)
    # Return negated fitness (convert back to maximization)
    best_fitness = -best_fitness
    
    return best_solution, best_fitness, history
