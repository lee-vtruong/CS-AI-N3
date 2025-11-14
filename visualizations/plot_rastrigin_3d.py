import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Explicitly register 3D projection
try:
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import projections
    # Force register 3D projection if not already registered
    if '3d' not in projections.get_projection_names():
        projections.register_projection(Axes3D)
except ImportError:
    print("Warning: Could not import 3D plotting capabilities")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problems.rastrigin import rastrigin

def plot_rastrigin_3d_surface():
    """
    Plot contour and heatmap of Rastrigin function with 2 dimensions
    to illustrate the complexity of the problem.
    Note: 3D surface plot disabled due to matplotlib compatibility issues.
    """

    print("Generating contour and heatmap plots for Rastrigin function...")

    # Create a grid of points
    x = np.linspace(-5.12, 5.12, 200)
    y = np.linspace(-5.12, 5.12, 200)
    X, Y = np.meshgrid(x, y)

    # Calculate Z values (Rastrigin function)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = rastrigin(np.array([X[i, j], Y[i, j]]))

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Heatmap plot
    im = ax1.imshow(Z, extent=[-5.12, 5.12, -5.12, 5.12], origin='lower',
                    cmap='viridis', aspect='auto', interpolation='bilinear')
    ax1.set_xlabel('X₁', fontsize=12)
    ax1.set_ylabel('X₂', fontsize=12)
    ax1.set_title('Rastrigin Function - Heatmap',
                  fontsize=14, fontweight='bold')
    ax1.plot(0, 0, 'r*', markersize=15, label='Global Minimum (0, 0)')
    ax1.legend(fontsize=10, loc='upper right')
    fig.colorbar(im, ax=ax1, shrink=0.8)

    # Contour plot
    contour = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
    ax2.contour(X, Y, Z, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    ax2.set_xlabel('X₁', fontsize=12)
    ax2.set_ylabel('X₂', fontsize=12)
    ax2.set_title('Rastrigin Function - Contour Plot',
                  fontsize=14, fontweight='bold')
    ax2.plot(0, 0, 'r*', markersize=15, label='Global Minimum (0, 0)')
    ax2.legend(fontsize=10)
    fig.colorbar(contour, ax=ax2, shrink=0.8)

    # Save figure to results/ directory
    results_output_dir = os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'results')
    output_file = os.path.join(results_output_dir, 'rastrigin_3d_surface.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_rastrigin_3d_surface_plot():
    """
    Plot 3D surface plot of Rastrigin function with improved visualization.
    """
    print("Generating improved 3D surface plot for Rastrigin function...")
    
    try:
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Warning: Could not import 3D plotting capabilities. Skipping 3D surface plot.")
        return
    
    # Create a finer grid for smoother surface
    x = np.linspace(-5.12, 5.12, 250)  # Increased resolution
    y = np.linspace(-5.12, 5.12, 250)
    X, Y = np.meshgrid(x, y)
    
    # Calculate Z values (Rastrigin function)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = rastrigin(np.array([X[i, j], Y[i, j]]))
    
    # Create 3D figure with better size
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface with improved settings
    surf = ax.plot_surface(X, Y, Z, 
                          cmap='rainbow',  # Rainbow colormap like the reference
                          alpha=0.95,      # Slightly transparent
                          linewidth=0,     # No grid lines on surface
                          antialiased=True,
                          shade=True,
                          edgecolor='none',
                          rcount=200,      # Number of rows in surface grid
                          ccount=200,      # Number of columns in surface grid
                          vmin=0,          # Minimum value for colormap
                          vmax=80)         # Maximum value for colormap
    
    # Add contour lines at the base
    contours = ax.contour(X, Y, Z, 
                         levels=15,
                         cmap='rainbow',
                         linewidths=1.5,
                         alpha=0.7,
                         offset=0)  # Draw at z=0
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.1)
    cbar.set_label('Objective Function Value', fontsize=12, labelpad=15)
    
    # Set labels with better formatting
    ax.set_xlabel('X', fontsize=14, labelpad=15, fontweight='bold')
    ax.set_ylabel('Y', fontsize=14, labelpad=15, fontweight='bold')
    ax.set_zlabel('Objective Function Value', fontsize=12, labelpad=15)
    
    # Set title
    ax.set_title('Rastrigin Function - 3D Surface Plot', 
                fontsize=16, fontweight='bold', pad=25)
    
    # Mark global minimum with a larger red star
    ax.scatter([0], [0], [rastrigin(np.array([0, 0]))], 
              color='red', s=200, marker='*', 
              edgecolors='darkred', linewidths=2,
              label='Global Minimum (0, 0)', zorder=10)
    
    # Set viewing angle for better visualization (similar to reference)
    ax.view_init(elev=25, azim=225)  # Adjusted angle
    
    # Set axis limits
    ax.set_xlim([-5.12, 5.12])
    ax.set_ylim([-5.12, 5.12])
    ax.set_zlim([0, 90])  # Set a reasonable upper limit
    
    # Improve grid appearance
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Set background color to white
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Make pane edges more visible
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    
    # Adjust tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Add legend
    ax.legend(fontsize=11, loc='upper left')
    
    # Save figure to results/ directory
    results_output_dir = os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'results')
    output_file = os.path.join(results_output_dir, 'rastrigin_3d_surface_plot.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_rastrigin_cross_sections():
    """
    Plot cross-sections of Rastrigin function to show its multimodal nature.
    """

    print("Generating cross-section plots for Rastrigin function...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.linspace(-5.12, 5.12, 500)

    # Cross-section along x1 (x2 = 0)
    z1 = np.array([rastrigin(np.array([xi, 0])) for xi in x])
    axes[0].plot(x, z1, 'b-', linewidth=2)
    axes[0].set_xlabel('X₁', fontsize=12)
    axes[0].set_ylabel('f(X₁, 0)', fontsize=12)
    axes[0].set_title('Cross-section along X₁ axis (X₂=0)',
                      fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=1,
                    alpha=0.7, label='Global Minimum')
    axes[0].legend(fontsize=10)

    # Cross-section along x2 (x1 = 0)
    z2 = np.array([rastrigin(np.array([0, xi])) for xi in x])
    axes[1].plot(x, z2, 'g-', linewidth=2)
    axes[1].set_xlabel('X₂', fontsize=12)
    axes[1].set_ylabel('f(0, X₂)', fontsize=12)
    axes[1].set_title('Cross-section along X₂ axis (X₁=0)',
                      fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=1,
                    alpha=0.7, label='Global Minimum')
    axes[1].legend(fontsize=10)

    # Save figure to results/ directory
    results_output_dir = os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'results')
    output_file = os.path.join(
        results_output_dir, 'rastrigin_cross_sections.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


if __name__ == '__main__':
    print("=" * 60)
    print("PLOTTING RASTRIGIN FUNCTION VISUALIZATION")
    print("=" * 60)

    print("\n[1] Creating heatmap and contour plots...")
    plot_rastrigin_3d_surface()

    print("\n[2] Creating 3D surface plot...")
    plot_rastrigin_3d_surface_plot()

    print("\n[3] Creating cross-section plots...")
    plot_rastrigin_cross_sections()

    print("\n" + "=" * 60)
    print("RASTRIGIN VISUALIZATION COMPLETED!")
    print("=" * 60)
    print("\nThese plots illustrate the multimodal and highly complex")
    print("nature of the Rastrigin function, with many local minima")
    print("surrounding the global minimum at (0, 0).")
