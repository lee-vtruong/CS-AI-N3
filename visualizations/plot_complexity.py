import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot_rastrigin_complexity():
    """Plot bar charts for time and memory complexity of Rastrigin experiments."""

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
            avg_time = float(row['Avg_Time'])
            avg_mem = float(row['Avg_Peak_Mem_MB'])
            
            key = f"{algo}_D{dim}"
            if key not in data:
                data[key] = {'algo': algo, 'dim': dim, 'time': avg_time, 'mem': avg_mem}

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
        # Prepare data for bar charts
        algo_names = []
        times = []
        memories = []
        bar_colors = []

        for algo in algorithms:
            key = f"{algo}_D{D}"
            if key in data:
                algo_names.append(algo)
                times.append(data[key]['time'])
                memories.append(data[key]['mem'])
                bar_colors.append(colors[algo])

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # --- Time Complexity Bar Chart ---
        bars1 = ax1.bar(algo_names, times, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.2)
        ax1.set_xlabel('Algorithm', fontsize=12)
        ax1.set_ylabel('Avg Time (seconds)', fontsize=12)
        ax1.set_title(f'Time Complexity - Rastrigin Function (D={D})',
                      fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_yscale('log')  # Log scale for better visualization

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}s',
                    ha='center', va='bottom', fontsize=9)

        # --- Memory Complexity Bar Chart ---
        bars2 = ax2.bar(algo_names, memories, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.2)
        ax2.set_xlabel('Algorithm', fontsize=12)
        ax2.set_ylabel('Avg Peak Memory (MB)', fontsize=12)
        ax2.set_title(f'Memory Complexity - Rastrigin Function (D={D})',
                      fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f} MB',
                    ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # Save figure to results/ directory
        results_output_dir = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), 'results/complexity')
        output_file = os.path.join(
            results_output_dir, f'complexity_rastrigin_D{D}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()


def plot_knapsack_complexity():
    """Plot bar charts for time and memory complexity of Knapsack experiments."""

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
            avg_time = float(row['Avg_Time'])
            avg_mem = float(row['Avg_Peak_Mem_MB'])
            
            key = f"{algo}_N{n_items}"
            if key not in data:
                data[key] = {'algo': algo, 'n_items': n_items, 'time': avg_time, 'mem': avg_mem}

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

    # Create plots for each problem size
    for n in n_items_list:
        # Prepare data for bar charts
        algo_names = []
        times = []
        memories = []
        bar_colors = []

        for algo in algorithms:
            key = f"{algo}_N{n}"
            if key in data:
                algo_names.append(algo)
                times.append(data[key]['time'])
                memories.append(data[key]['mem'])
                bar_colors.append(colors[algo])

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # --- Time Complexity Bar Chart ---
        bars1 = ax1.bar(algo_names, times, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.2)
        ax1.set_xlabel('Algorithm', fontsize=12)
        ax1.set_ylabel('Avg Time (seconds)', fontsize=12)
        ax1.set_title(f'Time Complexity - Knapsack Problem (N={n} items)',
                      fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_yscale('log')  # Log scale for better visualization

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}s',
                    ha='center', va='bottom', fontsize=9)

        # --- Memory Complexity Bar Chart ---
        bars2 = ax2.bar(algo_names, memories, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.2)
        ax2.set_xlabel('Algorithm', fontsize=12)
        ax2.set_ylabel('Avg Peak Memory (MB)', fontsize=12)
        ax2.set_title(f'Memory Complexity - Knapsack Problem (N={n} items)',
                      fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f} MB',
                    ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # Save figure to results/ directory
        results_output_dir = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), 'results/complexity')
        output_file = os.path.join(
            results_output_dir, f'complexity_knapsack_N{n}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()


if __name__ == '__main__':
    print("=" * 60)
    print("PLOTTING COMPUTATIONAL COMPLEXITY ANALYSIS")
    print("=" * 60)

    print("\n[1] Plotting Rastrigin complexity (Time & Memory)...")
    plot_rastrigin_complexity()

    print("\n[2] Plotting Knapsack complexity (Time & Memory)...")
    plot_knapsack_complexity()

    print("\n" + "=" * 60)
    print("COMPLEXITY PLOTS COMPLETED!")
    print("=" * 60)

