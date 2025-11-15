import numpy as np
import math

def levy_flight(n_dim, beta=1.5):
    """
    Generate step using Levy flight distribution.
    """
    sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
             (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
    
    u = np.random.randn(n_dim) * sigma
    v = np.random.randn(n_dim)
    
    step = u / np.abs(v)**(1 / beta)
    return step

def cuckoo_search_continuous(obj_func, bounds, n_dim, pop_size, max_iter, pa=0.25, beta=1.5, **kwargs):
    """
    Cuckoo Search (CS) algorithm for continuous optimization.
    
    Parameters:
    -----------
    obj_func : function
        Objective function to minimize
    bounds : array-like
        Bounds for each dimension [[min, max], ...]
    n_dim : int
        Number of dimensions
    pop_size : int
        Population size (number of nests)
    max_iter : int
        Maximum number of iterations
    pa : float
        Probability of discovering alien eggs (0 to 1)
    beta : float
        Parameter for Levy flight distribution
    
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
    
    # Initialize nests (solutions)
    nests = min_b + (max_b - min_b) * np.random.rand(pop_size, n_dim)
    
    # Evaluate fitness
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
            fitness[idx] = obj_func(nests[idx])
            
            # Update global best if necessary
            if fitness[idx] < best_fitness:
                best_fitness = fitness[idx]
                best_solution = nests[idx].copy()
        
        # Record history
        history.append(best_fitness)
    
    return best_solution, best_fitness, history


def cuckoo_search_discrete(obj_func, context, n_dim, pop_size, max_iter, pa=0.25, beta=1.5, **kwargs):
    """
    Cuckoo Search (CS) algorithm for discrete optimization.
    
    Parameters:
    -----------
    obj_func : function
        Objective function to minimize
    context : dict
        Context dict with problem data
    n_dim : int
        Number of dimensions
    pop_size : int
        Population size (number of nests)
    max_iter : int
        Maximum number of iterations
    pa : float
        Probability of discovering alien eggs (0 to 1)
    beta : float
        Parameter for Levy flight distribution
    
    Returns:
    --------
    best_solution : ndarray
        Best solution found (binary vector)
    best_fitness : float
        Best fitness value (minimization)
    history : list
        History of best fitness values
    """
    # For discrete, we'll use arbitrary bounds for initialization
    min_b = np.full(n_dim, -5.0)
    max_b = np.full(n_dim, 5.0)
    
    # Initialize nests (solutions)
    nests = min_b + (max_b - min_b) * np.random.rand(pop_size, n_dim)
    
    # Evaluate fitness with binarization (numerically stable sigmoid)
    def stable_sigmoid_binarize(x):
        # Clip x to prevent overflow in exp
        x_clipped = np.clip(x, -500, 500)
        probs = 1 / (1 + np.exp(-x_clipped))
        return (probs > 0.5).astype(int)
    
    fitness = np.array([
        obj_func(stable_sigmoid_binarize(nest), context) 
        for nest in nests
    ])
    
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
            
            # Add np.clip to ensure consistency
            new_nest = np.clip(new_nest, min_b, max_b)
            
            # Apply sigmoid to convert continuous to binary (numerically stable)
            # Clip x to prevent overflow in exp
            x = new_nest
            x_clipped = np.clip(x, -500, 500)
            probabilities = 1 / (1 + np.exp(-x_clipped))
            binary_solution = (probabilities > 0.5).astype(int)
            new_fitness = obj_func(binary_solution, context)
            
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
            # Evaluate with binarization (numerically stable sigmoid)
            # Clip x to prevent overflow in exp
            x = nests[idx]
            x_clipped = np.clip(x, -500, 500)
            probabilities = 1 / (1 + np.exp(-x_clipped))
            binary_solution = (probabilities > 0.5).astype(int)
            fitness[idx] = obj_func(binary_solution, context)
            
            # Update global best if necessary
            if fitness[idx] < best_fitness:
                best_fitness = fitness[idx]
                best_solution = nests[idx].copy()
        
        # Record history
        history.append(best_fitness)
    
    # Return binary solution (numerically stable sigmoid)
    # Clip x to prevent overflow in exp
    x = best_solution
    x_clipped = np.clip(x, -500, 500)
    probabilities = 1 / (1 + np.exp(-x_clipped))
    best_solution = (probabilities > 0.5).astype(int)
    
    return best_solution, best_fitness, history