import numpy as np

def abc_algorithm(obj_func, bounds, n_dim, pop_size, max_iter, limit=10, **kwargs):
    """
    Artificial Bee Colony (ABC) algorithm.
    
    Parameters:
    -----------
    obj_func : function
        Objective function to minimize
    bounds : array-like
        Bounds for each dimension [[min, max], ...]
    n_dim : int
        Number of dimensions
    pop_size : int
        Population size (number of food sources)
    max_iter : int
        Maximum number of iterations
    limit : int
        Abandonment limit (number of trials before abandoning a source)
    
    Returns:
    --------
    best_solution : ndarray
        Best solution found
    best_fitness : float
        Best fitness value
    history : list
        History of best fitness values
    """
    # Initialize bounds
    min_b, max_b = np.asarray(bounds).T
    
    # Initialize food sources (solutions)
    food_sources = min_b + (max_b - min_b) * np.random.rand(pop_size, n_dim)
    fitness = np.array([obj_func(sol) for sol in food_sources])
    
    # Trial counters for each food source
    trial = np.zeros(pop_size)
    
    # Find best solution
    best_idx = np.argmin(fitness)
    best_solution = food_sources[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    # History tracking
    history = [best_fitness]
    
    # ABC main loop
    for iteration in range(max_iter):
        # --- Employed Bee Phase ---
        for i in range(pop_size):
            # Select a random dimension
            j = np.random.randint(n_dim)
            
            # Select a random neighbor (different from i)
            k = np.random.choice([x for x in range(pop_size) if x != i])
            
            # Generate new solution
            phi = np.random.uniform(-1, 1)
            new_solution = food_sources[i].copy()
            new_solution[j] = food_sources[i][j] + phi * (food_sources[i][j] - food_sources[k][j])
            
            # Apply bounds
            new_solution = np.clip(new_solution, min_b, max_b)
            
            # Evaluate new solution
            new_fitness = obj_func(new_solution)
            
            # Greedy selection
            if new_fitness < fitness[i]:
                food_sources[i] = new_solution
                fitness[i] = new_fitness
                trial[i] = 0
                
                # Update global best
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_solution = new_solution.copy()
            else:
                trial[i] += 1
        
        # --- Onlooker Bee Phase ---
        # Calculate selection probabilities (fitness proportionate)
        # For minimization, convert to maximization-like probabilities
        max_fitness = np.max(fitness)
        fitness_normalized = max_fitness - fitness + 1e-10
        probabilities = fitness_normalized / np.sum(fitness_normalized)
        
        for _ in range(pop_size):
            # Select a food source based on probability
            i = np.random.choice(pop_size, p=probabilities)
            
            # Select a random dimension
            j = np.random.randint(n_dim)
            
            # Select a random neighbor
            k = np.random.choice([x for x in range(pop_size) if x != i])
            
            # Generate new solution
            phi = np.random.uniform(-1, 1)
            new_solution = food_sources[i].copy()
            new_solution[j] = food_sources[i][j] + phi * (food_sources[i][j] - food_sources[k][j])
            
            # Apply bounds
            new_solution = np.clip(new_solution, min_b, max_b)
            
            # Evaluate new solution
            new_fitness = obj_func(new_solution)
            
            # Greedy selection
            if new_fitness < fitness[i]:
                food_sources[i] = new_solution
                fitness[i] = new_fitness
                trial[i] = 0
                
                # Update global best
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_solution = new_solution.copy()
            else:
                trial[i] += 1
        
        # --- Scout Bee Phase ---
        # Abandon exhausted food sources
        for i in range(pop_size):
            if trial[i] > limit:
                # Generate new random solution
                food_sources[i] = min_b + (max_b - min_b) * np.random.rand(n_dim)
                fitness[i] = obj_func(food_sources[i])
                trial[i] = 0
                
                # Update global best if necessary
                if fitness[i] < best_fitness:
                    best_fitness = fitness[i]
                    best_solution = food_sources[i].copy()
        
        # Record history
        history.append(best_fitness)
    
    return best_solution, best_fitness, history

