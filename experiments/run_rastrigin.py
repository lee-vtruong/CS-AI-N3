import numpy as np
import time
import csv
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms import pso, abc, fa, cs, ga, hc
from problems.rastrigin import rastrigin

# --- Định nghĩa thông số chung ---
N_RUNS = 20  # Số lần chạy để lấy trung bình
DIMENSIONS = [10, 30]  # Thử nghiệm độ co giãn (Scalability)
POP_SIZE = 50
MAX_ITER = 1000

# Tham số riêng
ALGO_PARAMS = {
    'pso': {'w': 0.8, 'c1': 2.0, 'c2': 2.0},
    'abc': {'limit': 10},
    'fa': {'alpha': 0.5, 'beta0': 1.0, 'gamma': 0.95},
    'cs': {'pa': 0.25, 'beta': 1.5},
    'ga': {'problem_type': 'continuous', 'crossover_rate': 0.8,     'mutation_rate': 0.02},
    'hc': {'step_size': 0.1}
}

ALGOS = {
    'PSO': pso.pso,
    'ABC': abc.abc_algorithm,
    'FA': fa.firefly_algorithm,
    'CS': cs.cuckoo_search,
    'GA': ga.genetic_algorithm,
    'HC': hc.hill_climbing
}

# --- Script chạy thí nghiệm ---
results_summary = []  # Để lưu kết quả tổng hợp
convergence_data = {}  # Để lưu lịch sử hội tụ
raw_fitness_data = {}  # Để lưu dữ liệu thô cho box plot

print("=" * 60)
print("RASTRIGIN FUNCTION OPTIMIZATION EXPERIMENTS")
print("=" * 60)
print(f"Settings: N_RUNS={N_RUNS}, POP_SIZE={POP_SIZE}, MAX_ITER={MAX_ITER}")
print("=" * 60)

for D in DIMENSIONS:
    print(f"\n{'='*60}")
    print(f"Running experiments for D={D}")
    print(f"{'='*60}")
    
    bounds = np.array([[-5.12, 5.12]] * D)
    convergence_data[D] = {}
    raw_fitness_data[D] = {}
    
    for algo_name, algo_func in ALGOS.items():
        print(f"\n[{algo_name}] Starting {N_RUNS} runs...")
        run_histories = []
        run_fitnesses = []
        run_times = []
        
        for r in range(N_RUNS):
            start_time = time.time()
            
            # Xử lý input khác nhau cho HC
            if algo_name == 'HC':
                sol, fit, hist = algo_func(rastrigin, bounds, D, MAX_ITER, **ALGO_PARAMS['hc'])
            else:
                sol, fit, hist = algo_func(rastrigin, bounds, D, POP_SIZE, MAX_ITER, 
                                          **ALGO_PARAMS[algo_name.lower()])
            
            elapsed = time.time() - start_time
            run_times.append(elapsed)
            run_fitnesses.append(fit)
            run_histories.append(hist)
            
            if (r + 1) % 5 == 0:
                print(f"  Run {r+1}/{N_RUNS} completed. Best fitness: {fit:.4f}")
        
        # Tính toán thống kê
        avg_fit = np.mean(run_fitnesses)
        std_fit = np.std(run_fitnesses)
        min_fit = np.min(run_fitnesses)
        max_fit = np.max(run_fitnesses)
        avg_time = np.mean(run_times)
        
        print(f"[{algo_name}] Completed!")
        print(f"  Avg Fitness: {avg_fit:.4f} ± {std_fit:.4f}")
        print(f"  Min Fitness: {min_fit:.4f}")
        print(f"  Max Fitness: {max_fit:.4f}")
        print(f"  Avg Time: {avg_time:.4f}s")
        
        results_summary.append([algo_name, D, avg_fit, std_fit, min_fit, max_fit, avg_time])
        
        # Lưu lịch sử hội tụ trung bình
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
        convergence_data[D][algo_name] = avg_convergence
        
        # Save raw fitness data for box plots
        raw_fitness_data[D][algo_name] = run_fitnesses

# --- Lưu kết quả ra file CSV ---
results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(results_dir, exist_ok=True)

print(f"\n{'='*60}")
print("Saving results to CSV files...")
print(f"{'='*60}")

# Summary file
summary_file = os.path.join(results_dir, 'rastrigin_summary.csv')
with open(summary_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Algorithm', 'Dimension', 'Avg_Fitness', 'Std_Dev_Fitness', 
                     'Min_Fitness', 'Max_Fitness', 'Avg_Time'])
    writer.writerows(results_summary)
print(f"✓ Summary saved to: {summary_file}")

# Convergence file
convergence_file = os.path.join(results_dir, 'rastrigin_convergence.csv')
with open(convergence_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # Viết header
    header = ['Iteration'] + [f"{algo}_D{D}" for D in DIMENSIONS for algo in ALGOS.keys()]
    writer.writerow(header)
    # Viết data
    max_iters = max(len(convergence_data[D][algo]) for D in DIMENSIONS for algo in ALGOS.keys())
    for i in range(max_iters):
        row = [i]
        for D in DIMENSIONS:
            for algo in ALGOS.keys():
                if i < len(convergence_data[D][algo]):
                    row.append(convergence_data[D][algo][i])
                else:
                    row.append(convergence_data[D][algo][-1])
        writer.writerow(row)
print(f"✓ Convergence data saved to: {convergence_file}")

# Raw fitness data for box plots
rawdata_file = os.path.join(results_dir, 'rastrigin_raw_fitness.csv')
with open(rawdata_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # Header
    header = ['Run'] + [f"{algo}_D{D}" for D in DIMENSIONS for algo in ALGOS.keys()]
    writer.writerow(header)
    # Data
    for run in range(N_RUNS):
        row = [run + 1]
        for D in DIMENSIONS:
            for algo in ALGOS.keys():
                row.append(raw_fitness_data[D][algo][run])
        writer.writerow(row)
print(f"✓ Raw fitness data saved to: {rawdata_file}")

print(f"\n{'='*60}")
print("EXPERIMENTS COMPLETED SUCCESSFULLY!")
print(f"{'='*60}")
print(f"Results saved in: {results_dir}")

