import numpy as np

# --- Helper Functions ---

def tournament_selection(population, fitness, k=3, maximize=False):
    """
    Tournament selection.
    """
    selected = []
    pop_size = len(population)
    
    for _ in range(pop_size):
        # Randomly select k individuals
        indices = np.random.choice(pop_size, k, replace=False)
        
        # Find the best among them
        if maximize:
            best_idx = indices[np.argmax(fitness[indices])]
        else:
            best_idx = indices[np.argmin(fitness[indices])]
        
        selected.append(population[best_idx].copy())
    
    return np.array(selected)

def crossover_continuous(p1, p2, crossover_rate, bounds):
    """
    Simulated Binary Crossover (SBX) for continuous problems.
    """
    n_dim = len(p1)
    offspring1 = p1.copy()
    offspring2 = p2.copy()
    
    if np.random.rand() < crossover_rate:
        for i in range(n_dim):
            if np.random.rand() < 0.5:
                # Perform SBX
                if abs(p1[i] - p2[i]) > 1e-10:
                    eta = 20  # Distribution index
                    
                    if p1[i] < p2[i]:
                        y1 = p1[i]
                        y2 = p2[i]
                    else:
                        y1 = p2[i]
                        y2 = p1[i]
                    
                    min_b, max_b = bounds[i]
                    
                    beta = 1.0 + (2.0 * (y1 - min_b) / (y2 - y1))
                    alpha = 2.0 - beta ** -(eta + 1.0)
                    
                    rand = np.random.rand()
                    if rand <= 1.0 / alpha:
                        betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                    
                    c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                    
                    beta = 1.0 + (2.0 * (max_b - y2) / (y2 - y1))
                    alpha = 2.0 - beta ** -(eta + 1.0)
                    
                    if rand <= 1.0 / alpha:
                        betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                    
                    c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
                    
                    offspring1[i] = np.clip(c1, min_b, max_b)
                    offspring2[i] = np.clip(c2, min_b, max_b)
    
    return offspring1, offspring2

def crossover_discrete(p1, p2, crossover_rate):
    """
    One-point crossover for discrete (binary) problems.
    """
    n_dim = len(p1)
    offspring1 = p1.copy()
    offspring2 = p2.copy()
    
    if np.random.rand() < crossover_rate:
        # Choose crossover point
        point = np.random.randint(1, n_dim)
        
        # Perform crossover
        offspring1 = np.concatenate([p1[:point], p2[point:]])
        offspring2 = np.concatenate([p2[:point], p1[point:]])
    
    return offspring1, offspring2

def mutate_continuous(solution, mutation_rate, bounds):
    """
    Polynomial mutation for continuous problems.
    """
    n_dim = len(solution)
    mutated = solution.copy()
    
    for i in range(n_dim):
        if np.random.rand() < mutation_rate:
            min_b, max_b = bounds[i]
            delta = max_b - min_b
            
            # Polynomial mutation
            eta = 20  # Distribution index
            rand = np.random.rand()
            
            if rand < 0.5:
                delta_q = (2.0 * rand) ** (1.0 / (eta + 1.0)) - 1.0
            else:
                delta_q = 1.0 - (2.0 * (1.0 - rand)) ** (1.0 / (eta + 1.0))
            
            mutated[i] = solution[i] + delta_q * delta
            mutated[i] = np.clip(mutated[i], min_b, max_b)
    
    return mutated

def mutate_discrete(solution, mutation_rate):
    """
    Bit-flip mutation for discrete (binary) problems.
    """
    mutated = solution.copy()
    n_dim = len(solution)
    
    for i in range(n_dim):
        if np.random.rand() < mutation_rate:
            mutated[i] = 1 - mutated[i]  # Flip bit
    
    return mutated

# --- Main GA Functions ---

def genetic_continuous(obj_func, bounds, n_dim, pop_size, max_iter, 
                                 crossover_rate=0.8, mutation_rate=0.01, 
                                 **kwargs):
    """
    Genetic Algorithm (GA) for continuous optimization.
    
    Parameters:
    -----------
    obj_func : function
        Objective function to minimize
    bounds : array-like
        Bounds for each dimension [[min, max], ...]
    n_dim : int
        Number of dimensions
    pop_size : int
        Population size
    max_iter : int
        Maximum number of iterations (generations)
    crossover_rate : float
        Crossover probability
    mutation_rate : float
        Mutation probability
    
    Returns:
    --------
    best_solution : ndarray
        Best solution found (real-valued vector)
    best_fitness : float
        Best fitness value
    history : list
        History of best fitness values
    """
    bounds = np.asarray(bounds)
    min_b, max_b = bounds.T
    
    # Initialize population
    population = min_b + (max_b - min_b) * np.random.rand(pop_size, n_dim)
    fitness = np.array([obj_func(ind) for ind in population])
    
    # Find best
    best_idx = np.argmin(fitness)
    best_fitness = fitness[best_idx]
    best_solution = population[best_idx].copy()
    history = [best_fitness]
    
    # Evolution loop
    for generation in range(max_iter):
        # Selection
        parents = tournament_selection(population, fitness, k=3, maximize=False)
        
        # Crossover and Mutation
        offspring = []
        for i in range(0, pop_size, 2):
            if i + 1 < pop_size:
                p1, p2 = parents[i], parents[i + 1]
                o1, o2 = crossover_continuous(p1, p2, crossover_rate, bounds)
                o1 = mutate_continuous(o1, mutation_rate, bounds)
                o2 = mutate_continuous(o2, mutation_rate, bounds)
                offspring.append(o1)
                offspring.append(o2)
            else:
                o1 = mutate_continuous(parents[i], mutation_rate, bounds)
                offspring.append(o1)
        
        offspring = np.array(offspring[:pop_size])

        # --- THAY ĐỔI: ELITISM (Giữ lại cá thể tốt nhất) ---
        # Đảm bảo cá thể tốt nhất của thế hệ trước được truyền sang thế hệ sau
        offspring[0] = best_solution.copy()
        
        # Evaluate offspring
        offspring_fitness = np.array([obj_func(ind) for ind in offspring])
        
        # Replacement (generational)
        population = offspring
        fitness = offspring_fitness
        
        # Update best
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_solution = population[current_best_idx].copy()
        
        history.append(best_fitness)
    
    return best_solution, best_fitness, history


def genetic_discrete(obj_func, context, n_dim, pop_size, max_iter, 
                               crossover_rate=0.8, mutation_rate=0.01, 
                               **kwargs):
    """
    Genetic Algorithm (GA) for discrete optimization.
    
    Parameters:
    -----------
    obj_func : function
        Objective function to maximize
    context : dict
        Context dict with problem data
    n_dim : int
        Number of dimensions/items
    pop_size : int
        Population size
    max_iter : int
        Maximum number of iterations (generations)
    crossover_rate : float
        Crossover probability
    mutation_rate : float
        Mutation probability
    
    Returns:
    --------
    best_solution : ndarray
        Best solution found (binary vector)
    best_fitness : float
        Best fitness value
    history : list
        History of best fitness values
    """
    # Initialize population (binary)
    population = np.random.randint(0, 2, (pop_size, n_dim))
    fitness = np.array([obj_func(ind, context) for ind in population])
    
    # Find best
    best_idx = np.argmax(fitness)
    best_fitness = fitness[best_idx]
    best_solution = population[best_idx].copy()
    history = [best_fitness]
    
    # Evolution loop
    for generation in range(max_iter):
        # Selection
        parents = tournament_selection(population, fitness, k=3, maximize=True)
        
        # Crossover and Mutation
        offspring = []
        for i in range(0, pop_size, 2):
            if i + 1 < pop_size:
                p1, p2 = parents[i], parents[i + 1]
                o1, o2 = crossover_discrete(p1, p2, crossover_rate)
                o1 = mutate_discrete(o1, mutation_rate)
                o2 = mutate_discrete(o2, mutation_rate)
                offspring.append(o1)
                offspring.append(o2)
            else:
                o1 = mutate_discrete(parents[i], mutation_rate)
                offspring.append(o1)
        
        offspring = np.array(offspring[:pop_size], dtype=int)

        # --- THAY ĐỔI: ELITISM (Giữ lại cá thể tốt nhất) ---
        # Đảm bảo cá thể tốt nhất của thế hệ trước được truyền sang thế hệ sau
        offspring[0] = best_solution.copy()
        
        # Evaluate offspring
        offspring_fitness = np.array([obj_func(ind, context) for ind in offspring])
        
        # Replacement (generational)
        population = offspring
        fitness = offspring_fitness
        
        # Update best
        current_best_idx = np.argmax(fitness)
        if fitness[current_best_idx] > best_fitness:
            best_fitness = fitness[current_best_idx]
            best_solution = population[current_best_idx].copy()
        
        history.append(best_fitness)
    
    return best_solution, best_fitness, history