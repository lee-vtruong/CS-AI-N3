import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot_rastrigin_robustness():
    """Plot box plots for Rastrigin experiments to show robustness."""

    results_dir = os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'results')
    rawdata_file = os.path.join(results_dir, 'rastrigin_raw_fitness.csv')

    if not os.path.exists(rawdata_file):
        print(f"Error: {rawdata_file} not found!")
        print("Please run experiments/run_rastrigin.py first.")
        return

    # Read raw fitness data
    data = {}
    with open(rawdata_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key != 'Run':
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

    # Create box plots for each dimension
    for D in dimensions:
        fig, ax = plt.subplots(figsize=(12, 7))

        # Prepare data for box plot
        plot_data = []
        labels = []
        box_colors = []

        for algo in algorithms:
            key = f"{algo}_D{D}"
            if key in data:
                plot_data.append(data[key])
                labels.append(algo)
                box_colors.append(colors[algo])

        # Create box plot
        bp = ax.boxplot(plot_data, patch_artist=True,
                        showmeans=True, meanline=True,
                        medianprops=dict(color='red', linewidth=2),
                        meanprops=dict(color='blue', linewidth=2, linestyle='--'))
        # Set tick labels (compatible with older matplotlib versions)
        ax.set_xticklabels(labels)

        # Color the boxes
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('Best Fitness Value', fontsize=12)
        ax.set_title(f'Robustness Comparison - Rastrigin Function (D={D})',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')  # Log scale for better visualization

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=2, label='Median'),
            Line2D([0], [0], color='blue', linewidth=2,
                   linestyle='--', label='Mean')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        # Save figure to results/ directory
        results_output_dir = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), 'results')
        output_file = os.path.join(
            results_output_dir, f'robustness_rastrigin_D{D}.png')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()


def plot_knapsack_robustness():
    """Plot box plots for Knapsack experiments to show robustness."""

    results_dir = os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'results')
    rawdata_file = os.path.join(results_dir, 'knapsack_raw_fitness.csv')

    if not os.path.exists(rawdata_file):
        print(f"Error: {rawdata_file} not found!")
        print("Please run experiments/run_knapsack.py first.")
        return

    # Read raw fitness data
    data = {}
    with open(rawdata_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key != 'Run':
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

    # Create box plots for each problem size
    for n in n_items_list:
        fig, ax = plt.subplots(figsize=(14, 7))

        # Prepare data for box plot
        plot_data = []
        labels = []
        box_colors = []

        for algo in algorithms:
            key = f"{algo}_N{n}"
            if key in data:
                plot_data.append(data[key])
                labels.append(algo)
                box_colors.append(colors[algo])

        # Create box plot
        bp = ax.boxplot(plot_data, patch_artist=True,
                        showmeans=True, meanline=True,
                        medianprops=dict(color='red', linewidth=2),
                        meanprops=dict(color='blue', linewidth=2, linestyle='--'))
        # Set tick labels (compatible with older matplotlib versions)
        ax.set_xticklabels(labels)

        # Color the boxes
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('Best Fitness Value (Total Value)', fontsize=12)
        ax.set_title(f'Robustness Comparison - Knapsack Problem (N={n} items)',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=2, label='Median'),
            Line2D([0], [0], color='blue', linewidth=2,
                   linestyle='--', label='Mean')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

        # Save figure to results/ directory
        results_output_dir = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), 'results')
        output_file = os.path.join(
            results_output_dir, f'robustness_knapsack_N{n}.png')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()


if __name__ == '__main__':
    print("=" * 60)
    print("PLOTTING ROBUSTNESS ANALYSIS (BOX PLOTS)")
    print("=" * 60)

    print("\n[1] Plotting Rastrigin robustness...")
    plot_rastrigin_robustness()

    print("\n[2] Plotting Knapsack robustness...")
    plot_knapsack_robustness()

    print("\n" + "=" * 60)
    print("ROBUSTNESS PLOTS COMPLETED!")
    print("=" * 60)
