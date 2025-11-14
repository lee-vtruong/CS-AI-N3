import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot_rastrigin_single_algo_convergence():
    """Plot convergence curves for each algorithm separately for Rastrigin experiments.
    Each algorithm gets a figure with 2 subplots (D=10 and D=30)."""

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

    # Extract dimensions and algorithms
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

    # Create plots for each algorithm
    for algo in algorithms:
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Subplot 1: D=10
        key_d10 = f"{algo}_D{dimensions[0]}"
        if key_d10 in data:
            ax1.plot(data[key_d10], linewidth=2, color=colors[algo])
        
        ax1.set_xlabel('Iteration', fontsize=11)
        ax1.set_ylabel('Best Fitness Value', fontsize=11)
        ax1.set_title(f'{algo} - Rastrigin Function (D={dimensions[0]})', 
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Subplot 2: D=30
        key_d30 = f"{algo}_D{dimensions[1]}"
        if key_d30 in data:
            ax2.plot(data[key_d30], linewidth=2, color=colors[algo])
        
        ax2.set_xlabel('Iteration', fontsize=11)
        ax2.set_ylabel('Best Fitness Value', fontsize=11)
        ax2.set_title(f'{algo} - Rastrigin Function (D={dimensions[1]})', 
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Add overall title
        fig.suptitle(f'Convergence Curves - {algo} Algorithm', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # Save figure to results/ directory
        results_output_dir = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), 'results/single_algo_convergence')
        output_file = os.path.join(
            results_output_dir, f'single_algo_convergence_{algo}_rastrigin.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()


def plot_knapsack_single_algo_convergence():
    """Plot convergence curves for each algorithm separately for Knapsack experiments.
    Each algorithm gets a figure with 2 subplots (N=20 and N=50)."""

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

    # Extract problem sizes and algorithms
    n_items_list = [20, 50]
    algorithms = ['PSO', 'ABC', 'FA', 'CS', 'GA', 'HC', 'ACO', 'SA']

    # Color scheme
    colors = {
        'PSO': '#1f77b4',
        'ABC': '#ff7f0e',
        'FA': '#2ca02c',
        'CS': '#d62728',
        'GA': '#9467bd',
        'HC': '#8c564b',
        'ACO': '#e377c2',
        'SA': '#7f7f7f'
    }

    # Create plots for each algorithm
    for algo in algorithms:
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Subplot 1: N=20
        key_n20 = f"{algo}_N{n_items_list[0]}"
        if key_n20 in data:
            ax1.plot(data[key_n20], linewidth=2, color=colors[algo])
        
        ax1.set_xlabel('Iteration', fontsize=11)
        ax1.set_ylabel('Best Fitness Value (Total Value)', fontsize=11)
        ax1.set_title(f'{algo} - Knapsack Problem (N={n_items_list[0]} items)', 
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: N=50
        key_n50 = f"{algo}_N{n_items_list[1]}"
        if key_n50 in data:
            ax2.plot(data[key_n50], linewidth=2, color=colors[algo])
        
        ax2.set_xlabel('Iteration', fontsize=11)
        ax2.set_ylabel('Best Fitness Value (Total Value)', fontsize=11)
        ax2.set_title(f'{algo} - Knapsack Problem (N={n_items_list[1]} items)', 
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add overall title
        fig.suptitle(f'Convergence Curves - {algo} Algorithm', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # Save figure to results/ directory
        results_output_dir = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), 'results/single_algo_convergence')
        output_file = os.path.join(
            results_output_dir, f'single_algo_convergence_{algo}_knapsack.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()


if __name__ == '__main__':
    print("=" * 60)
    print("PLOTTING SINGLE ALGORITHM CONVERGENCE CURVES")
    print("=" * 60)

    print("\n[1] Plotting Rastrigin single algorithm convergence...")
    plot_rastrigin_single_algo_convergence()

    print("\n[2] Plotting Knapsack single algorithm convergence...")
    plot_knapsack_single_algo_convergence()

    print("\n" + "=" * 60)
    print("SINGLE ALGORITHM CONVERGENCE PLOTS COMPLETED!")
    print("=" * 60)

