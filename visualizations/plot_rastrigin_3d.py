import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import sys
import os

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
    ax1.set_title('Rastrigin Function - Heatmap', fontsize=14, fontweight='bold')
    ax1.plot(0, 0, 'r*', markersize=15, label='Global Minimum (0, 0)')
    ax1.legend(fontsize=10, loc='upper right')
    fig.colorbar(im, ax=ax1, shrink=0.8)
    
    # Contour plot
    contour = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
    ax2.contour(X, Y, Z, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    ax2.set_xlabel('X₁', fontsize=12)
    ax2.set_ylabel('X₂', fontsize=12)
    ax2.set_title('Rastrigin Function - Contour Plot', fontsize=14, fontweight='bold')
    ax2.plot(0, 0, 'r*', markersize=15, label='Global Minimum (0, 0)')
    ax2.legend(fontsize=10)
    fig.colorbar(contour, ax=ax2, shrink=0.8)
    
    # Save figure
    vis_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(vis_dir, 'rastrigin_3d_surface.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
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
    axes[0].set_title('Cross-section along X₁ axis (X₂=0)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=1, alpha=0.7, label='Global Minimum')
    axes[0].legend(fontsize=10)
    
    # Cross-section along x2 (x1 = 0)
    z2 = np.array([rastrigin(np.array([0, xi])) for xi in x])
    axes[1].plot(x, z2, 'g-', linewidth=2)
    axes[1].set_xlabel('X₂', fontsize=12)
    axes[1].set_ylabel('f(0, X₂)', fontsize=12)
    axes[1].set_title('Cross-section along X₂ axis (X₁=0)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=1, alpha=0.7, label='Global Minimum')
    axes[1].legend(fontsize=10)
    
    # Save figure
    vis_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(vis_dir, 'rastrigin_cross_sections.png')
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
    
    print("\n[2] Creating cross-section plots...")
    plot_rastrigin_cross_sections()
    
    print("\n" + "=" * 60)
    print("RASTRIGIN VISUALIZATION COMPLETED!")
    print("=" * 60)
    print("\nThese plots illustrate the multimodal and highly complex")
    print("nature of the Rastrigin function, with many local minima")
    print("surrounding the global minimum at (0, 0).")

