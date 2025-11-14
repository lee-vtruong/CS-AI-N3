import numpy as np

def pso_continuous(obj_func, bounds, n_dim, pop_size, max_iter, w=0.8, c1=2.0, c2=2.0, **kwargs):
    """
    Particle Swarm Optimization (PSO) algorithm for continuous optimization.
    
    Parameters:
    -----------
    obj_func : function
        Objective function to minimize
    bounds : array-like
        Bounds for each dimension [[min, max], ...]
    n_dim : int
        Number of dimensions
    pop_size : int
        Population size (number of particles)
    max_iter : int
        Maximum number of iterations
    w : float
        Inertia weight
    c1 : float
        Cognitive coefficient
    c2 : float
        Social coefficient
    
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
    
    # Initialize particles positions and velocities
    particles = min_b + (max_b - min_b) * np.random.rand(pop_size, n_dim)
    velocities = np.random.randn(pop_size, n_dim)
    
    # Initialize personal best positions and fitness
    personal_best_positions = particles.copy()
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
            
            # Apply bounds
            particles[i] = np.clip(particles[i], min_b, max_b)
            
            # Evaluate fitness
            fitness = obj_func(particles[i])
            
            # Update personal best
            if fitness < personal_best_fitness[i]:
                personal_best_fitness[i] = fitness
                personal_best_positions[i] = particles[i].copy()
                
                # Update global best
                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = particles[i].copy()
        
        # Record history
        history.append(global_best_fitness)
    
    return global_best_position, global_best_fitness, history


def pso_discrete(obj_func, context, n_dim, pop_size, max_iter, w=0.8, c1=2.0, c2=2.0, **kwargs):
    """
    Particle Swarm Optimization (PSO) algorithm for discrete optimization.
    
    Parameters:
    -----------
    obj_func : function
        Objective function to maximize (will be negated internally for minimization)
    context : dict
        Context dict with problem data
    n_dim : int
        Number of dimensions
    pop_size : int
        Population size (number of particles)
    max_iter : int
        Maximum number of iterations
    w : float
        Inertia weight
    c1 : float
        Cognitive coefficient
    c2 : float
        Social coefficient
    
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
    # The actual solution will be binarized before evaluation
    min_b = np.full(n_dim, -5.0)
    max_b = np.full(n_dim, 5.0)
    
    # Initialize particles positions and velocities
    particles = min_b + (max_b - min_b) * np.random.rand(pop_size, n_dim)
    velocities = np.random.randn(pop_size, n_dim)
    
    # Initialize personal best positions and fitness
    # Keep positions as continuous for velocity updates, but evaluate fitness on binary for discrete
    personal_best_positions = particles.copy()
    # Evaluate initial fitness with binarization (numerically stable sigmoid)
    # Negate fitness for maximization (Knapsack is maximization, but we minimize)
    def stable_sigmoid_binarize(x):
        # Clip x to prevent overflow in exp
        x_clipped = np.clip(x, -500, 500)
        probs = 1 / (1 + np.exp(-x_clipped))
        return (probs > 0.5).astype(int)
    
    personal_best_fitness = np.array([
        -obj_func(stable_sigmoid_binarize(p), context) 
        for p in particles
    ])
    
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
            
            # Apply sigmoid to convert continuous to binary (numerically stable)
            # Clip x to prevent overflow in exp
            x = particles[i]
            x_clipped = np.clip(x, -500, 500)
            probabilities = 1 / (1 + np.exp(-x_clipped))
            binary_solution = (probabilities > 0.5).astype(int)
            # Negate fitness for maximization (Knapsack is maximization, but we minimize)
            fitness = -obj_func(binary_solution, context)
            
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
    
    # Return binary solution (numerically stable sigmoid)
    # Clip x to prevent overflow in exp
    x = global_best_position
    x_clipped = np.clip(x, -500, 500)
    probabilities = 1 / (1 + np.exp(-x_clipped))
    best_solution = (probabilities > 0.5).astype(int)
    # Return negated fitness (convert back to maximization)
    best_fitness = -global_best_fitness
    # Convert history to maximization (all values should be positive)
    history = [-h for h in history]
    
    return best_solution, best_fitness, history
