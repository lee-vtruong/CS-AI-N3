import numpy as np
import time
import csv
import sys
import os
import tracemalloc

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problems.knapsack import generate_knapsack_data, knapsack_fitness
from algorithms import aco, ga, a_star, pso, abc, fa, cs, hc

# --- Định nghĩa thông số chung ---
N_RUNS = 20  # Số lần chạy để lấy trung bình
N_ITEMS_LIST = [20, 50]  # Thử nghiệm với số lượng items khác nhau
POP_SIZE = 50
MAX_ITER = 1000

# Tham số riêng
ALGO_PARAMS = {
    'pso': {'w': 0.8, 'c1': 2.0, 'c2': 2.0},
    'abc': {'limit': 10},
    'fa': {'alpha': 0.5, 'beta0': 1.0, 'gamma': 0.95},
    'cs': {'pa': 0.25, 'beta': 1.5},
    'ga': {'crossover_rate': 0.8, 'mutation_rate': 0.05},
    'hc': {'step_size': 0.1},
    'aco': {'alpha': 1.0, 'beta': 2.0, 'rho': 0.5}
}

ALGOS = {
    'PSO': pso.pso_discrete,
    'ABC': abc.abc_algorithm_discrete,
    'FA': fa.firefly_algorithm_discrete,
    'CS': cs.cuckoo_search_discrete,
    'GA': ga.genetic_algorithm_discrete,
    'HC': hc.hill_climbing_discrete,
    'ACO': aco.aco,
    'A*': a_star.a_star_search
}

# --- Script chạy thí nghiệm ---
results_summary = []  # Để lưu kết quả tổng hợp
convergence_data = {}  # Để lưu lịch sử hội tụ
raw_fitness_data = {}  # Để lưu dữ liệu thô cho box plot

print("=" * 60)
print("KNAPSACK PROBLEM OPTIMIZATION EXPERIMENTS")
print("=" * 60)
print(f"Settings: N_RUNS={N_RUNS}, POP_SIZE={POP_SIZE}, MAX_ITER={MAX_ITER}")
print("=" * 60)

for n_items in N_ITEMS_LIST:
    print(f"\n{'='*60}")
    print(f"Running experiments for N_ITEMS={n_items}")
    print(f"{'='*60}")

    # Generate knapsack problem instance
    weights, values, capacity, _ = generate_knapsack_data(n_items, seed=42)
    context = {
        'weights': weights,
        'values': values,
        'capacity': capacity
    }

    print(f"Problem: {n_items} items, capacity={capacity}")
    print(f"Total weight: {np.sum(weights)}, Total value: {np.sum(values)}")

    convergence_data[n_items] = {}
    raw_fitness_data[n_items] = {}

    for algo_name, algo_func in ALGOS.items():
        print(f"\n[{algo_name}] Starting {N_RUNS} runs...")
        run_histories = []
        run_fitnesses = []
        run_times = []
        run_memories = []

        for r in range(N_RUNS):
            # Start memory tracing
            tracemalloc.start()
            start_time = time.time()

            # A* is different - it's deterministic and doesn't have iterations
            if algo_name == 'A*':
                sol, fit = algo_func(context)
                hist = [fit]  # A* doesn't have history, just final result
            elif algo_name == 'ACO':
                # ACO doesn't need n_dim
                sol, fit, hist = algo_func(knapsack_fitness, context, POP_SIZE, MAX_ITER,
                                           **ALGO_PARAMS[algo_name.lower()])
            elif algo_name == 'HC':
                # HC doesn't need pop_size parameter
                sol, fit, hist = algo_func(knapsack_fitness, context, n_items, MAX_ITER,
                                           **ALGO_PARAMS[algo_name.lower()])
            else:
                # All other algorithms (PSO, ABC, FA, CS, GA) use _discrete functions
                sol, fit, hist = algo_func(knapsack_fitness, context, n_items, POP_SIZE, MAX_ITER,
                                           **ALGO_PARAMS[algo_name.lower()])

            elapsed = time.time() - start_time
            # Get peak memory usage (in MB)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mem_mb = peak / 1024 / 1024  # Convert bytes to MB

            run_times.append(elapsed)
            run_fitnesses.append(fit)
            run_histories.append(hist)
            run_memories.append(peak_mem_mb)

            if (r + 1) % 5 == 0:
                print(
                    f"  Run {r+1}/{N_RUNS} completed. Best fitness: {fit:.2f}")

        # Tính toán thống kê
        avg_fit = np.mean(run_fitnesses)
        std_fit = np.std(run_fitnesses)
        min_fit = np.min(run_fitnesses)
        max_fit = np.max(run_fitnesses)
        avg_time = np.mean(run_times)
        avg_peak_mem = np.mean(run_memories)

        print(f"[{algo_name}] Completed!")
        print(f"  Avg Fitness: {avg_fit:.2f} ± {std_fit:.2f}")
        print(f"  Min Fitness: {min_fit:.2f}")
        print(f"  Max Fitness: {max_fit:.2f}")
        print(f"  Avg Time: {avg_time:.4f}s")
        print(f"  Avg Peak Memory: {avg_peak_mem:.4f} MB")

        results_summary.append(
            [algo_name, n_items, avg_fit, std_fit, min_fit, max_fit, avg_time, avg_peak_mem])

        # Lưu lịch sử hội tụ trung bình
        if algo_name == 'A*':
            # For A*, create a flat history
            avg_convergence = [max_fit] * (MAX_ITER + 1)
        else:
            # Pad histories to same length if needed
            max_len = max(len(h) for h in run_histories)
            padded_histories = []
            for h in run_histories:
                if len(h) < max_len:
                    # Pad with last value
                    padded = list(h) + [h[-1]] * (max_len - len(h))
                else:
                    padded = h
                padded_histories.append(padded)

            avg_convergence = np.mean(np.array(padded_histories), axis=0)

        convergence_data[n_items][algo_name] = avg_convergence

        # Save raw fitness data for box plots
        raw_fitness_data[n_items][algo_name] = run_fitnesses

# --- Lưu kết quả ra file CSV ---
results_dir = os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(results_dir, exist_ok=True)

print(f"\n{'='*60}")
print("Saving results to CSV files...")
print(f"{'='*60}")

# Summary file
summary_file = os.path.join(results_dir, 'knapsack_summary.csv')
with open(summary_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Algorithm', 'N_Items', 'Avg_Fitness', 'Std_Dev_Fitness',
                     'Min_Fitness', 'Max_Fitness', 'Avg_Time', 'Avg_Peak_Mem_MB'])
    writer.writerows(results_summary)
print(f"✓ Summary saved to: {summary_file}")

# Convergence file
convergence_file = os.path.join(results_dir, 'knapsack_convergence.csv')
with open(convergence_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # Viết header
    header = ['Iteration'] + \
        [f"{algo}_N{n}" for n in N_ITEMS_LIST for algo in ALGOS.keys()]
    writer.writerow(header)
    # Viết data
    max_iters = max(len(convergence_data[n][algo])
                    for n in N_ITEMS_LIST for algo in ALGOS.keys())
    for i in range(max_iters):
        row = [i]
        for n in N_ITEMS_LIST:
            for algo in ALGOS.keys():
                if i < len(convergence_data[n][algo]):
                    row.append(convergence_data[n][algo][i])
                else:
                    row.append(convergence_data[n][algo][-1])
        writer.writerow(row)
print(f"✓ Convergence data saved to: {convergence_file}")

# Raw fitness data for box plots
rawdata_file = os.path.join(results_dir, 'knapsack_raw_fitness.csv')
with open(rawdata_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # Header
    header = ['Run'] + \
        [f"{algo}_N{n}" for n in N_ITEMS_LIST for algo in ALGOS.keys()]
    writer.writerow(header)
    # Data
    for run in range(N_RUNS):
        row = [run + 1]
        for n in N_ITEMS_LIST:
            for algo in ALGOS.keys():
                row.append(raw_fitness_data[n][algo][run])
        writer.writerow(row)
print(f"✓ Raw fitness data saved to: {rawdata_file}")

print(f"\n{'='*60}")
print("EXPERIMENTS COMPLETED SUCCESSFULLY!")
print(f"{'='*60}")
print(f"Results saved in: {results_dir}")
