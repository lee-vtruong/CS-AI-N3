import numpy as np

def pso(obj_func, context_or_bounds, n_dim, pop_size, max_iter, problem_type='continuous', w=0.8, c1=2.0, c2=2.0, **kwargs):
    """
    Particle Swarm Optimization (PSO) algorithm.
    
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
        Population size (number of particles)
    max_iter : int
        Maximum number of iterations
    problem_type : str
        'continuous' (default) or 'discrete'
    w : float
        Inertia weight
    c1 : float
        Cognitive coefficient
    c2 : float
        Social coefficient
    
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
        # The actual solution will be binarized before evaluation
        min_b = np.full(n_dim, -5.0)
        max_b = np.full(n_dim, 5.0)
    
    # Initialize particles positions and velocities
    particles = min_b + (max_b - min_b) * np.random.rand(pop_size, n_dim)
    velocities = np.random.randn(pop_size, n_dim)
    
    # Initialize personal best positions and fitness
    # Keep positions as continuous for velocity updates, but evaluate fitness on binary for discrete
    personal_best_positions = particles.copy()
    if problem_type == 'discrete':
        # Evaluate initial fitness with binarization
        # Negate fitness for maximization (Knapsack is maximization, but we minimize)
        personal_best_fitness = np.array([
            -obj_func((1 / (1 + np.exp(-p)) > 0.5).astype(int), context) 
            for p in particles
        ])
    else:
        personal_best_fitness = np.array([obj_func(p) for p in particles])
    
    # Initialize global best
    best_idx = np.argmin(personal_best_fitness)
    global_best_position = personal_best_positions[best_idx].copy()
    global_best_fitness = personal_best_fitness[best_idx]
    
    # History tracking
    history = [global_best_fitness]
    
    # PSO main loop
    for iteration in range(max_iter):
        for i in range(pop_size):
            # Update velocity
            r1, r2 = np.random.rand(2)
            cognitive = c1 * r1 * (personal_best_positions[i] - particles[i])
            social = c2 * r2 * (global_best_position - particles[i])
            velocities[i] = w * velocities[i] + cognitive + social
            
            # Update position
            particles[i] = particles[i] + velocities[i]
            
            # Evaluate fitness based on problem type
            if problem_type == 'discrete':
                # Apply sigmoid to convert continuous to binary
                probabilities = 1 / (1 + np.exp(-particles[i]))
                binary_solution = (probabilities > 0.5).astype(int)
                # Negate fitness for maximization (Knapsack is maximization, but we minimize)
                fitness = -obj_func(binary_solution, context)
            else:  # 'continuous'
                # Apply bounds
                particles[i] = np.clip(particles[i], min_b, max_b)
                fitness = obj_func(particles[i])
            
            # Update personal best
            if fitness < personal_best_fitness[i]:
                personal_best_fitness[i] = fitness
                # Store continuous position for velocity updates
                personal_best_positions[i] = particles[i].copy()
                
                # Update global best
                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = particles[i].copy()
        
        # Record history
        history.append(global_best_fitness)
    
    # Return binary solution for discrete problems, continuous for continuous
    if problem_type == 'discrete':
        probabilities = 1 / (1 + np.exp(-global_best_position))
        best_solution = (probabilities > 0.5).astype(int)
        # Return negated fitness (convert back to maximization)
        best_fitness = -global_best_fitness
    else:
        best_solution = global_best_position
        best_fitness = global_best_fitness
    
    return best_solution, best_fitness, history

