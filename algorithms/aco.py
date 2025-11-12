# algorithms/aco.py 
import numpy as np


def aco_discrete(obj_func, context, pop_size, max_iter, alpha=1.0, beta=2.0, rho=0.5, Q = 1.0, **kwargs):
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

    weights = context['weights']
    values = context['values']
    capacity = context['capacity']
    n_items = len(weights)

    # Initialize pheromone and heuristic
    pheromone = np.ones(n_items)*0.1
    eta = values / (weights + 1e-8)  # value/weight ratio

    history = []
    best_fitness = -np.inf
    best_solution = None

    for _ in range(max_iter):
        solutions = []
        fitnesses = []

        for _ in range(pop_size):
            sol = np.zeros(n_items, dtype=int)
            available_items = np.ones(n_items, dtype=bool)
            current_weight = 0.0
            # Construct solution by greedy selection
            while available_items.any():
                # Find feasible items (can be added without exceeding capacity)
                feasible_items = np.where((available_items) & (current_weight + weights <= capacity))[0]
                
                # If no feasible items remain, stop
                if not feasible_items.any():
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
            if fit > best_fitness:
                best_fitness = fit
                best_solution = sol.copy()

        history.append(best_fitness)

        # Pheromone update: evaporation + elite deposit
        pheromone *= (1 - rho)
        for sol, fit in zip(solutions, fitnesses):
            if fit > 0:
                pheromone += sol * (Q * fit)
        
    return best_solution, best_fitness, history


def acor_continuous(
    fitness_func,
    bounds,
    n_dim,
    pop_size,
    max_iter,
    Q=1.0,
    k_neighbors=3,
    xi=0.85,
    seed=None
):
    """
    Ant Colony Optimization for Continuous Domains (ACOr).

    Implements the standard ACOr algorithm (Socha & Dorigo, 2008).
    Maintains an archive of solutions and samples new solutions using
    Gaussian kernels centered at archive members, with adaptive standard
    deviation based on neighbor distances.

    Parameters
    ----------
    fitness_func : callable
        Objective function with signature: fitness_func(solution)
        where solution is a 1D np.ndarray of length n_dim.
    bounds : np.ndarray
        2D array of shape (n_dim, 2) defining lower and upper bounds:
        [[lb1, ub1], [lb2, ub2], ...].
    n_dim : int
        Dimensionality of the problem.
    pop_size : int
        Size of the solution archive (number of ants).
    max_iter : int
        Maximum number of iterations.
    Q : float, default=1.0
        Controls selection probability: weight = Q / rank.
    k_neighbors : int, default=3
        Number of nearest neighbors used to compute adaptive sigma.
    xi : float, default=0.85
        Scaling factor for standard deviation (xi ∈ (0,1)).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    best_solution : np.ndarray
        Best solution found in the search space.
    best_fitness : float
        Fitness of the best solution.
    history : list of float
        Best fitness value at each iteration.

    References
    ----------
    Socha, K., & Dorigo, M. (2008). Ant colony optimization for continuous domains.
    European Journal of Operational Research, 185(3), 1155–1173.

    Notes
    -----
    - Archive is updated by combining old and new solutions and keeping top `pop_size`.
    - Sigma is computed as: xi * mean(distance to k nearest neighbors).
    - Highly effective for multimodal continuous functions like Rastrigin.
    """
    if seed is not None:
        np.random.seed(seed)

    bounds = np.array(bounds)
    lb, ub = bounds[:, 0], bounds[:, 1]

    # Initialize solution archive
    archive = np.random.uniform(lb, ub, size=(pop_size, n_dim))
    archive_fits = np.array([fitness_func(sol) for sol in archive])

    best_fitness = np.max(archive_fits)
    best_solution = archive[np.argmax(archive_fits)].copy()
    history = [best_fitness]

    for _ in range(max_iter):
        new_solutions = []
        new_fitnesses = []

        for _ in range(pop_size):
            # Select solution from archive with probability proportional to 1/rank
            ranks = np.arange(1, pop_size + 1)
            weights = Q / ranks
            weights /= weights.sum()
            idx = np.random.choice(pop_size, p=weights)
            mu = archive[idx]

            # Compute adaptive standard deviation (sigma)
            dists = np.linalg.norm(archive - mu, axis=1)
            dists[idx] = np.inf  # exclude self
            nearest_dists = np.partition(dists, k_neighbors - 1)[:k_neighbors]
            sigma = xi * np.mean(nearest_dists)
            sigma = max(sigma, 1e-8)

            # Sample new solution from Gaussian
            sol = np.clip(np.random.normal(mu, sigma), lb, ub)
            fit = fitness_func(sol)

            new_solutions.append(sol)
            new_fitnesses.append(fit)

            # Update global best
            if fit > best_fitness:
                best_fitness = fit
                best_solution = sol.copy()

        history.append(best_fitness)
        new_solutions = np.array(new_solutions)
        new_fitnesses = np.array(new_fitnesses)

        # Update archive: keep top `pop_size` from old + new
        candidates = np.vstack((archive, new_solutions))
        cand_fits = np.hstack((archive_fits, new_fitnesses))
        order = np.argsort(cand_fits)[::-1]
        archive = candidates[order[:pop_size]]
        archive_fits = cand_fits[order[:pop_size]]

    return best_solution, best_fitness, history