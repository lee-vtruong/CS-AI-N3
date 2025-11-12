import numpy as np

def abc_algorithm_continuous(obj_func, bounds, n_dim, pop_size, max_iter, limit=10, **kwargs):
    """
    Artificial Bee Colony (ABC) algorithm for continuous optimization.
    
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
        Best solution found (real-valued vector)
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


def abc_algorithm_discrete(obj_func, context, n_dim, pop_size, max_iter, limit=10, **kwargs):
    """
    Artificial Bee Colony (ABC) algorithm for discrete optimization.
    
    Parameters:
    -----------
    obj_func : function
        Objective function to maximize (will be negated internally for minimization)
    context : dict
        Context dict with problem data
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
        Best solution found (binary vector)
    best_fitness : float
        Best fitness value (maximization, not negated)
    history : list
        History of best fitness values
    """
    # For discrete, we'll use arbitrary bounds for initialization
    min_b = np.full(n_dim, -5.0)
    max_b = np.full(n_dim, 5.0)
    
    # Initialize food sources (solutions)
    food_sources = min_b + (max_b - min_b) * np.random.rand(pop_size, n_dim)
    # Evaluate initial fitness with binarization
    # Negate fitness for maximization (Knapsack is maximization, but we minimize)
    fitness = np.array([
        -obj_func((1 / (1 + np.exp(-sol)) > 0.5).astype(int), context) 
        for sol in food_sources
    ])
    
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
            
            # Apply sigmoid to convert continuous to binary
            probabilities = 1 / (1 + np.exp(-new_solution))
            binary_solution = (probabilities > 0.5).astype(int)
            # Negate fitness for maximization (Knapsack is maximization, but we minimize)
            new_fitness = -obj_func(binary_solution, context)
            
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
            
            # Apply sigmoid to convert continuous to binary
            probabilities = 1 / (1 + np.exp(-new_solution))
            binary_solution = (probabilities > 0.5).astype(int)
            # Negate fitness for maximization (Knapsack is maximization, but we minimize)
            new_fitness = -obj_func(binary_solution, context)
            
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
                # Evaluate with binarization
                probabilities = 1 / (1 + np.exp(-food_sources[i]))
                binary_solution = (probabilities > 0.5).astype(int)
                # Negate fitness for maximization (Knapsack is maximization, but we minimize)
                fitness[i] = -obj_func(binary_solution, context)
                trial[i] = 0
                
                # Update global best if necessary
                if fitness[i] < best_fitness:
                    best_fitness = fitness[i]
                    best_solution = food_sources[i].copy()
        
        # Record history
        history.append(best_fitness)
    
    # Return binary solution
    probabilities = 1 / (1 + np.exp(-best_solution))
    best_solution = (probabilities > 0.5).astype(int)
    # Return negated fitness (convert back to maximization)
    best_fitness = -best_fitness
    
    return best_solution, best_fitness, history
