import numpy as np

def aco(obj_func, context, pop_size, max_iter, alpha=1.0, beta=2.0, rho=0.5, **kwargs):
    """
    Ant Colony Optimization (ACO) for discrete optimization (Knapsack problem).
    
    Parameters:
    -----------
    obj_func : function
        Fitness function (takes solution and context)
    context : dict
        Problem context with 'weights', 'values', 'capacity', etc.
    pop_size : int
        Number of ants
    max_iter : int
        Maximum number of iterations
    alpha : float
        Pheromone importance factor
    beta : float
        Heuristic importance factor
    rho : float
        Pheromone evaporation rate (0 to 1)
    
    Returns:
    --------
    best_solution : ndarray
        Best solution found (binary vector)
    best_fitness : float
        Best fitness value
    history : list
        History of best fitness values
    """
    # Extract problem data
    weights = context['weights']
    values = context['values']
    capacity = context['capacity']
    n_items = len(weights)
    
    # Initialize pheromone trails (for each item: include or not)
    pheromone = np.ones((n_items, 2))  # [item][0=exclude, 1=include]
    
    # Calculate heuristic information (value/weight ratio)
    heuristic = np.zeros((n_items, 2))
    for i in range(n_items):
        heuristic[i, 0] = 0.1  # Small value for not including
        if weights[i] > 0:
            heuristic[i, 1] = values[i] / weights[i]  # Value/weight ratio
        else:
            heuristic[i, 1] = values[i]
    
    # Track best solution
    best_solution = None
    best_fitness = 0
    
    # History tracking
    history = [best_fitness]
    
    # ACO main loop
    for iteration in range(max_iter):
        solutions = []
        fitnesses = []
        
        # Each ant constructs a solution
        for ant in range(pop_size):
            solution = np.zeros(n_items, dtype=int)
            current_weight = 0
            
            # Construct solution item by item
            available_items = list(range(n_items))
            np.random.shuffle(available_items)
            
            for item in available_items:
                # Calculate probabilities for include/exclude
                if current_weight + weights[item] <= capacity:
                    # Can potentially include this item
                    prob_exclude = (pheromone[item, 0] ** alpha) * (heuristic[item, 0] ** beta)
                    prob_include = (pheromone[item, 1] ** alpha) * (heuristic[item, 1] ** beta)
                    
                    total_prob = prob_exclude + prob_include
                    
                    if total_prob > 0:
                        prob_include = prob_include / total_prob
                        
                        # Decide whether to include item
                        if np.random.rand() < prob_include:
                            solution[item] = 1
                            current_weight += weights[item]
            
            # Evaluate solution
            fitness = obj_func(solution, context)
            solutions.append(solution)
            fitnesses.append(fitness)
            
            # Update best solution
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = solution.copy()
        
        # Evaporate pheromones
        pheromone *= (1 - rho)
        
        # Update pheromones based on solutions
        for solution, fitness in zip(solutions, fitnesses):
            if fitness > 0:  # Only update if valid solution
                # Deposit pheromone proportional to fitness
                deposit = fitness / best_fitness if best_fitness > 0 else 1.0
                
                for item in range(n_items):
                    if solution[item] == 1:
                        pheromone[item, 1] += deposit
                    else:
                        pheromone[item, 0] += deposit * 0.1  # Smaller deposit for exclusion
        
        # Extra pheromone for best solution
        if best_solution is not None:
            for item in range(n_items):
                if best_solution[item] == 1:
                    pheromone[item, 1] += 1.0
        
        # Record history
        history.append(best_fitness)
    
    if best_solution is None:
        best_solution = np.zeros(n_items, dtype=int)
    
    return best_solution, best_fitness, history

