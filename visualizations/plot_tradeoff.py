import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot_rastrigin_tradeoff():
    """Plot scatter plot showing trade-off between time and fitness for Rastrigin experiments."""

    results_dir = os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'results')
    summary_file = os.path.join(results_dir, 'rastrigin_summary.csv')

    if not os.path.exists(summary_file):
        print(f"Error: {summary_file} not found!")
        print("Please run experiments/run_rastrigin.py first.")
        return

    # Read summary data
    data = {}
    with open(summary_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            algo = row['Algorithm']
            dim = int(row['Dimension'])
            avg_fitness = float(row['Avg_Fitness'])
            avg_time = float(row['Avg_Time'])
            
            key = f"{algo}_D{dim}"
            if key not in data:
                data[key] = {'algo': algo, 'dim': dim, 'fitness': avg_fitness, 'time': avg_time}

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

    # Create plots for each dimension
    for D in dimensions:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot each algorithm as a point
        for algo in algorithms:
            key = f"{algo}_D{D}"
            if key in data:
                x = data[key]['time']
                y = data[key]['fitness']
                ax.scatter(x, y, s=200, color=colors[algo], alpha=0.7,
                          edgecolors='black', linewidth=2, label=algo, zorder=3)
                # Add algorithm label near the point
                ax.annotate(algo, (x, y), xytext=(5, 5), textcoords='offset points',
                           fontsize=10, fontweight='bold', color=colors[algo])

        ax.set_xlabel('Avg Time (seconds)', fontsize=12)
        ax.set_ylabel('Avg Fitness Value', fontsize=12)
        ax.set_title(f'Time vs Fitness Trade-off - Rastrigin Function (D={D})',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')  # Log scale for time (better visualization)
        ax.set_yscale('log')  # Log scale for fitness (better visualization)

        # Add legend
        ax.legend(loc='best', fontsize=10, framealpha=0.9)

        # Add note about optimization direction
        ax.text(0.02, 0.98, 'Note: Lower fitness is better (minimization)',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save figure to results/ directory
        results_output_dir = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), 'results')
        output_file = os.path.join(
            results_output_dir, f'tradeoff_rastrigin_D{D}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()


def plot_knapsack_tradeoff():
    """Plot scatter plot showing trade-off between time and fitness for Knapsack experiments."""

    results_dir = os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'results')
    summary_file = os.path.join(results_dir, 'knapsack_summary.csv')

    if not os.path.exists(summary_file):
        print(f"Error: {summary_file} not found!")
        print("Please run experiments/run_knapsack.py first.")
        return

    # Read summary data
    data = {}
    with open(summary_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            algo = row['Algorithm']
            n_items = int(row['N_Items'])
            avg_fitness = float(row['Avg_Fitness'])
            avg_time = float(row['Avg_Time'])
            
            key = f"{algo}_N{n_items}"
            if key not in data:
                data[key] = {'algo': algo, 'n_items': n_items, 'fitness': avg_fitness, 'time': avg_time}

    # Extract problem sizes and algorithms
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
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot each algorithm as a point
        for algo in algorithms:
            key = f"{algo}_N{n}"
            if key in data:
                x = data[key]['time']
                y = data[key]['fitness']
                ax.scatter(x, y, s=200, color=colors[algo], alpha=0.7,
                          edgecolors='black', linewidth=2, label=algo, zorder=3)
                # Add algorithm label near the point
                ax.annotate(algo, (x, y), xytext=(5, 5), textcoords='offset points',
                           fontsize=10, fontweight='bold', color=colors[algo])

        ax.set_xlabel('Avg Time (seconds)', fontsize=12)
        ax.set_ylabel('Avg Fitness Value (Total Value)', fontsize=12)
        ax.set_title(f'Time vs Fitness Trade-off - Knapsack Problem (N={n} items)',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')  # Log scale for time (better visualization)

        # Add legend
        ax.legend(loc='best', fontsize=10, framealpha=0.9)

        # Add note about optimization direction
        ax.text(0.02, 0.98, 'Note: Higher fitness is better (maximization)',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save figure to results/ directory
        results_output_dir = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), 'results')
        output_file = os.path.join(
            results_output_dir, f'tradeoff_knapsack_N{n}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()


if __name__ == '__main__':
    print("=" * 60)
    print("PLOTTING TIME vs FITNESS TRADE-OFF ANALYSIS")
    print("=" * 60)

    print("\n[1] Plotting Rastrigin trade-off...")
    plot_rastrigin_tradeoff()

    print("\n[2] Plotting Knapsack trade-off...")
    plot_knapsack_tradeoff()

    print("\n" + "=" * 60)
    print("TRADE-OFF PLOTS COMPLETED!")
    print("=" * 60)
    print("\nThese plots show the trade-off between computational time")
    print("and solution quality for each algorithm.")

