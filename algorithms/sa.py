import numpy as np

def simulated_annealing_continuous(obj_func, bounds, n_dim, max_iter, 
                                   initial_temp=100.0, cooling_rate=0.95, 
                                   min_temp=0.01, **kwargs):
    """
    Simulated Annealing algorithm for continuous optimization.
    
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
    initial_temp : float
        Initial temperature
    cooling_rate : float
        Temperature cooling rate (0 < cooling_rate < 1)
    min_temp : float
        Minimum temperature (stopping criterion)
    
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
    
    # Initialize temperature
    temperature = initial_temp
    
    # History tracking
    history = [best_fitness]
    
    # Simulated Annealing main loop
    for iteration in range(max_iter):
        # Generate neighbor by adding random perturbation
        # Perturbation size decreases with temperature
        perturbation_scale = (max_b - min_b) * (temperature / initial_temp)
        perturbation = np.random.randn(n_dim) * perturbation_scale
        neighbor_solution = current_solution + perturbation
        
        # Apply bounds
        neighbor_solution = np.clip(neighbor_solution, min_b, max_b)
        
        # Evaluate neighbor
        neighbor_fitness = obj_func(neighbor_solution)
        
        # Calculate acceptance probability
        delta = neighbor_fitness - current_fitness
        
        # Accept if better, or accept worse with probability based on temperature
        if delta < 0:  # Better solution
            accept = True
        else:  # Worse solution
            # Acceptance probability: exp(-delta / temperature)
            if temperature > 0:
                acceptance_prob = np.exp(-delta / temperature)
                accept = np.random.rand() < acceptance_prob
            else:
                accept = False
        
        if accept:
            current_solution = neighbor_solution
            current_fitness = neighbor_fitness
            
            # Update best
            if current_fitness < best_fitness:
                best_solution = current_solution.copy()
                best_fitness = current_fitness
        
        # Cool down temperature
        temperature = max(initial_temp * (cooling_rate ** iteration), min_temp)
        
        # Record history
        history.append(best_fitness)
    
    return best_solution, best_fitness, history


def simulated_annealing_discrete(obj_func, context, n_dim, max_iter,
                                 initial_temp=100.0, cooling_rate=0.95,
                                 min_temp=0.01, **kwargs):
    """
    Simulated Annealing algorithm for discrete optimization.
    
    Parameters:
    -----------
    obj_func : function
        Objective function to minimize
    context : dict
        Context dict with problem data
    n_dim : int
        Number of dimensions
    max_iter : int
        Maximum number of iterations
    initial_temp : float
        Initial temperature
    cooling_rate : float
        Temperature cooling rate (0 < cooling_rate < 1)
    min_temp : float
        Minimum temperature (stopping criterion)
    
    Returns:
    --------
    best_solution : ndarray
        Best solution found (binary vector)
    best_fitness : float
        Best fitness value (minimization)
    history : list
        History of best fitness values
    """
    # Initialize random binary solution until valid
    while True:
        current_solution = np.random.randint(0, 2, n_dim)
        total_weight = np.dot(current_solution, context['weights'])
        if total_weight <= context['capacity']:
            break
    
    current_fitness = obj_func(current_solution, context)
    
    # Track best solution
    best_solution = current_solution.copy()
    best_fitness = current_fitness
    
    # Initialize temperature
    temperature = initial_temp
    
    # History tracking
    history = [best_fitness]
    
    # Simulated Annealing main loop
    for iteration in range(max_iter):
        # Generate neighbor by flipping one random bit
        neighbor_solution = current_solution.copy()
        bit_to_flip = np.random.randint(0, n_dim)
        neighbor_solution[bit_to_flip] = 1 - neighbor_solution[bit_to_flip]
        
        # Check if neighbor is valid (for Knapsack)
        total_weight = np.dot(neighbor_solution, context['weights'])
        if total_weight > context['capacity']:
            # Invalid solution, skip
            history.append(best_fitness)
            # Cool down temperature
            temperature = max(initial_temp * (cooling_rate ** iteration), min_temp)
            continue
        
        # Evaluate neighbor
        neighbor_fitness = obj_func(neighbor_solution, context)
        
        # Calculate acceptance probability
        delta = neighbor_fitness - current_fitness
        
        # Accept if better, or accept worse with probability based on temperature
        if delta < 0:  # Better solution
            accept = True
        else:  # Worse solution
            # Acceptance probability: exp(-delta / temperature)
            if temperature > 0:
                acceptance_prob = np.exp(-delta / temperature)
                accept = np.random.rand() < acceptance_prob
            else:
                accept = False
        
        if accept:
            current_solution = neighbor_solution
            current_fitness = neighbor_fitness
            
            # Update best
            if current_fitness < best_fitness:
                best_solution = current_solution.copy()
                best_fitness = current_fitness
        
        # Cool down temperature
        temperature = max(initial_temp * (cooling_rate ** iteration), min_temp)
        
        # Record history
        history.append(best_fitness)
    
    return best_solution, best_fitness, history

