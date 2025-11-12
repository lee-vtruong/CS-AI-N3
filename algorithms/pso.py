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
    Binary Particle Swarm Optimization (BPSO) for discrete optimization.
    
    Parameters:
    -----------
    obj_func : function
        Objective function to maximize
    context : dict
        Problem context with 'weights', 'values', 'capacity', etc.
    n_dim : int
        Number of dimensions (items)
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
        Best binary solution found (0/1 vector)
    best_fitness : float
        Best fitness value
    history : list
        History of best fitness values
    """
    # Initialize velocity space (continuous, unbounded)
    velocities = np.zeros((pop_size, n_dim))
    
    # Initialize particles as binary {0, 1}
    particles = np.random.randint(0, 2, size=(pop_size, n_dim))
    
    # Initialize personal best (binary solutions and fitness)
    personal_best_particles = particles.copy()
    personal_best_fitness = np.array([obj_func(p, context) for p in particles])
    
    # Initialize global best
    best_idx = np.argmax(personal_best_fitness)
    global_best_particle = personal_best_particles[best_idx].copy()
    global_best_fitness = personal_best_fitness[best_idx]
    
    # History tracking
    history = [global_best_fitness]
    
    # Helper function: sigmoid transfer function
    def sigmoid_transfer(v):
        """Numerically stable sigmoid S(v) = 1 / (1 + exp(-v))"""
        v_clipped = np.clip(v, -500, 500)
        return 1.0 / (1.0 + np.exp(-v_clipped))
    
    # BPSO main loop
    for iteration in range(max_iter):
        for i in range(pop_size):
            # Update velocity: v = w*v + c1*rand()*(pbest-x) + c2*rand()*(gbest-x)
            r1 = np.random.rand(n_dim)
            r2 = np.random.rand(n_dim)
            cognitive = c1 * r1 * (personal_best_particles[i].astype(float) - particles[i].astype(float))
            social = c2 * r2 * (global_best_particle.astype(float) - particles[i].astype(float))
            velocities[i] = w * velocities[i] + cognitive + social
            
            # Apply sigmoid transfer function and stochastic binarization
            # x_i(t+1) = 1 if rand() < S(v_i(t+1)), else 0
            transfer_probs = sigmoid_transfer(velocities[i])
            particles[i] = (np.random.rand(n_dim) < transfer_probs).astype(int)
            
            # Evaluate fitness
            fitness = obj_func(particles[i], context)
            
            # Update personal best
            if fitness > personal_best_fitness[i]:
                personal_best_fitness[i] = fitness
                personal_best_particles[i] = particles[i].copy()
                
                # Update global best
                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best_particle = particles[i].copy()
        
        # Record history
        history.append(global_best_fitness)
    
    return global_best_particle, global_best_fitness, history
