import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot_rastrigin_convergence():
    """Plot convergence curves for Rastrigin experiments."""

    results_dir = os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'results')
    convergence_file = os.path.join(results_dir, 'rastrigin_convergence.csv')

    if not os.path.exists(convergence_file):
        print(f"Error: {convergence_file} not found!")
        print("Please run experiments/run_rastrigin.py first.")
        return

    # Read convergence data
    data = {}
    with open(convergence_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            iteration = int(row['Iteration'])
            for key, value in row.items():
                if key != 'Iteration':
                    if key not in data:
                        data[key] = []
                    data[key].append(float(value))

    # Extract dimensions
    dimensions = [10, 30]
    algorithms = ['PSO', 'ABC', 'FA', 'CS', 'GA', 'HC']

    # Color scheme
    colors = {
        'PSO': '#1f77b4',
        'ABC': '#ff7f0e',
        'FA': '#2ca02c',
        'CS': '#d62728',
        'GA': '#9467bd',
        'HC': '#8c564b'
    }

    # Create plots for each dimension
    for D in dimensions:
        plt.figure(figsize=(12, 7))

        for algo in algorithms:
            key = f"{algo}_D{D}"
            if key in data:
                plt.plot(data[key], label=algo,
                         linewidth=2, color=colors[algo])

        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Best Fitness Value', fontsize=12)
        plt.title(
            f'Convergence Curves - Rastrigin Function (D={D})', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for better visualization

        # Save figure to results/ directory
        results_output_dir = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), 'results')
        output_file = os.path.join(
            results_output_dir, f'convergence_rastrigin_D{D}.png')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()


def plot_knapsack_convergence():
    """Plot convergence curves for Knapsack experiments."""

    results_dir = os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'results')
    convergence_file = os.path.join(results_dir, 'knapsack_convergence.csv')

    if not os.path.exists(convergence_file):
        print(f"Error: {convergence_file} not found!")
        print("Please run experiments/run_knapsack.py first.")
        return

    # Read convergence data
    data = {}
    with open(convergence_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            iteration = int(row['Iteration'])
            for key, value in row.items():
                if key != 'Iteration':
                    if key not in data:
                        data[key] = []
                    data[key].append(float(value))

    # Extract problem sizes
    n_items_list = [20, 50]
    algorithms = ['PSO', 'ABC', 'FA', 'CS', 'GA', 'HC', 'ACO', 'A*']

    # Color scheme
    colors = {
        'PSO': '#1f77b4',
        'ABC': '#ff7f0e',
        'FA': '#2ca02c',
        'CS': '#d62728',
        'GA': '#9467bd',
        'HC': '#8c564b',
        'ACO': '#e377c2',
        'A*': '#7f7f7f'
    }

    # Create plots for each problem size
    for n in n_items_list:
        plt.figure(figsize=(12, 7))

        for algo in algorithms:
            key = f"{algo}_N{n}"
            if key in data:
                plt.plot(data[key], label=algo,
                         linewidth=2, color=colors[algo])

        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Best Fitness Value (Total Value)', fontsize=12)
        plt.title(
            f'Convergence Curves - Knapsack Problem (N={n} items)', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)

        # Save figure to results/ directory
        results_output_dir = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), 'results')
        output_file = os.path.join(
            results_output_dir, f'convergence_knapsack_N{n}.png')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()


if __name__ == '__main__':
    print("=" * 60)
    print("PLOTTING CONVERGENCE CURVES")
    print("=" * 60)

    print("\n[1] Plotting Rastrigin convergence...")
    plot_rastrigin_convergence()

    print("\n[2] Plotting Knapsack convergence...")
    plot_knapsack_convergence()

    print("\n" + "=" * 60)
    print("CONVERGENCE PLOTS COMPLETED!")
    print("=" * 60)
