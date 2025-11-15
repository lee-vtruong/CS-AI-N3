# algorithms/aco.py 
import numpy as np


def aco_discrete(obj_func, context, pop_size, max_iter, alpha=1.0, beta=2.0, rho=0.5, Q = 1.0, **kwargs):
    """
    Ant Colony Optimization (ACO) for discrete optimization (Knapsack problem).
    
    Parameters:
    -----------
    obj_func : function
        Objective function to minimize (takes solution and context)
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
        Best fitness value (minimization)
    history : list
        History of best fitness values
    """

    weights = np.array(context['weights'])
    values = np.array(context['values'])
    capacity = context['capacity']
    n_items = len(weights)

    # Initialize pheromone and heuristic
    pheromone = np.ones(n_items)*0.1
    eta = values / (weights + 1e-8)  # value/weight ratio

    history = []
    best_fitness = np.inf
    best_solution = None

    for _ in range(max_iter):
        solutions = []
        fitnesses = []

        for _ in range(pop_size):
            sol = np.zeros(n_items, dtype=int)
            available_items = np.ones(n_items, dtype=bool)
            current_weight = 0.0
            # Construct solution
            while np.any(available_items):
                # Find items that can be added (without exceeding capacity)
                feasible_items = np.where((available_items) & (current_weight + weights <= capacity))[0]
                
                # If no items can be added, stop
                if  len(feasible_items) == 0:
                    break

                prob = (pheromone[feasible_items] ** alpha) * (eta[feasible_items] ** beta)
                if prob.sum() == 0:
                    prob = np.ones(len(feasible_items)) / len(feasible_items)
                else:
                    prob /= prob.sum()
                selected_item = np.random.choice(feasible_items, p=prob)
                sol[selected_item] = 1
                current_weight += weights[selected_item]
                available_items[selected_item] = False

            fit = obj_func(sol, context)
            solutions.append(sol)
            fitnesses.append(fit)

            # Update global best
            if fit < best_fitness:
                best_fitness = fit
                best_solution = sol.copy()

        history.append(best_fitness)

        # Pheromone update: evaporation + elite deposit
        # For minimization, better solutions have less negative fitness (closer to 0)
        # We use -fit to convert to positive reward (larger for better solutions)
        pheromone *= (1 - rho)
        for sol, fit in zip(solutions, fitnesses):
            # Skip penalty cases (fit >= 0 indicates invalid solution)
            if fit < 0:
                pheromone += sol * (Q * (-fit))
        
    return best_solution, best_fitness, history

def aco_continuous(obj_func, bounds, n_dim, archive_size, pop_size, max_iter, q=0.1, xi=0.85, **kwargs):
    """
    Ant Colony Optimization for continuous domains (ACOr) based on the paper:
    "Ant Colony Optimization for Continuous Domains" by Socha and Dorigo (2008).
    
    This implementation assumes minimization of the objective function.
    Handles unconstrained problems, but clips solutions to bounds if provided.
    
    Parameters:
    -----------
    obj_func : function
        Objective function to minimize (takes a solution vector as input).
    bounds : list of tuples
        List of (min, max) for each dimension (n_dim elements).
    n_dim : int
        Number of dimensions (variables).
    archive_size : int
        Size of the solution archive (k in the paper, recommended k >= n_dim).
    pop_size : int
        Number of ants (m in the paper).
    max_iter : int
        Maximum number of iterations.
    q : float
        Parameter controlling the weight distribution (small q prefers best solutions).
    xi : float
        Parameter for standard deviation (similar to evaporation rate, higher xi -> slower convergence).
    
    Returns:
    --------
    best_solution : ndarray
        Best solution found.
    best_fitness : float
        Best fitness value.
    history : list
        History of best fitness values per iteration.
    """
    bounds = np.array(bounds)
    lb, ub = bounds[:, 0], bounds[:, 1]

    # Initialize archive
    archive = np.random.uniform(lb, ub, size=(archive_size, n_dim))
    fitness = np.array([obj_func(sol) for sol in archive])
    sorted_idx = np.argsort(fitness)
    archive = archive[sorted_idx]
    fitness = fitness[sorted_idx]

    history = []
    best_fitness = fitness[0]
    best_solution = archive[0].copy()

    # Calculate weights once (similar to MATLAB implementation)
    ranks = np.arange(1, archive_size + 1)
    w = (1 / (q * archive_size * np.sqrt(2 * np.pi))) * np.exp(- (ranks - 1)**2 / (2 * q**2 * archive_size**2))
    p = w / w.sum()  # Probability of selecting solution l

    for iter in range(max_iter):
        new_solutions = []
        new_fitnesses = []

        for _ in range(pop_size):
            # Select a solution l from archive according to p
            l = np.random.choice(archive_size, p=p)

            sol = np.zeros(n_dim)
            for i in range(n_dim):
                mu = archive[l, i]

                # Calculate sigma_i^l = xi * avg |s_i^e - s_i^l| for all e ≠ l
                if archive_size > 1:
                    distances = np.abs(archive[:, i] - mu)
                    sigma = xi * np.sum(distances) / (archive_size - 1)
                else:
                    sigma = xi * (ub[i] - lb[i])  # fallback

                sol[i] = np.random.normal(mu, sigma)
            
            sol = np.clip(sol, lb, ub)
            fit = obj_func(sol)
            new_solutions.append(sol)
            new_fitnesses.append(fit)

            if fit < best_fitness:
                best_fitness = fit
                best_solution = sol.copy()

        # Merge archive + new solutions → sort → keep top k
        combined = np.vstack((archive, new_solutions))
        combined_fit = np.concatenate((fitness, new_fitnesses))
        sorted_idx = np.argsort(combined_fit)
        archive = combined[sorted_idx[:archive_size]]
        fitness = combined_fit[sorted_idx[:archive_size]]

        history.append(best_fitness)

        # Update weights and p (because archive has changed)
        ranks = np.arange(1, archive_size + 1)
        w = (1 / (q * archive_size * np.sqrt(2 * np.pi))) * np.exp(- (ranks - 1)**2 / (2 * q**2 * archive_size**2))
        p = w / w.sum()

    return best_solution, best_fitness, history