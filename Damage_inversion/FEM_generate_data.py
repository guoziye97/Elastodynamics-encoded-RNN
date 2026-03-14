"""
Anisotropic Elastic Wave Propagation - FEM with Structured Triangular Mesh + Damage Regions

Fixed boundary version using checkerboard alternating diagonal pattern
Boundary condition: Fixed boundary (Dirichlet)

Key features:
1. Same node positions as FDM (structured mesh)
2. Triangular element finite element solution
3. Anisotropic material
4. Ricker wavelet (Mexican hat wavelet) force source
5. GPU acceleration (PyTorch)
6. Fixed boundary condition (u = 0 on boundary)
7. Support for circular/square damage regions (stiffness degradation)

"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon, Patch
from matplotlib.collections import PatchCollection
import time
import torch
import scipy.io as sio
import matplotlib
import h5py
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Matplotlib settings
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'stixsans'

# GPU settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================================
# Part 1: Utility Functions
# ============================================================================

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Truncate colormap"""
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def gaussian_point_load(x, y, x0, y0, a=150):
    """
    Gaussian function approximating Dirac delta function

    Parameters:
    -----------
    x, y : coordinates
    x0, y0 : load application point
    a : Gaussian parameter controlling load concentration

    Returns:
    --------
    Gaussian function value (approximating delta(x-x0, y-y0))
    """
    r_squared = (x - x0) ** 2 + (y - y0) ** 2
    return (a / np.pi) * np.exp(-a * r_squared)


def ricker_wavelet(t, f0, t0, amplitude=1.0):
    """
    Ricker wavelet (Mexican hat wavelet)

    Parameters:
    -----------
    t : time (s)
    f0 : dominant frequency (Hz)
    t0 : time delay (s)
    amplitude : amplitude (N)

    Returns:
    --------
    Ricker wavelet value

    Formula: R(t) = A * (1 - 2*pi^2*f0^2*(t-t0)^2) * exp(-pi^2*f0^2*(t-t0)^2)
    """
    tau = t - t0
    pi_f0_tau_sq = (np.pi * f0 * tau) ** 2
    return amplitude * (1 - 2 * pi_f0_tau_sq) * np.exp(-pi_f0_tau_sq)


def calculate_fwhm(a):
    """Calculate Full Width at Half Maximum (FWHM) of Gaussian function"""
    return 2 * np.sqrt(np.log(2) / a)


def calculate_wave_velocities_anisotropic(C11, C22, C12, C16, C26, C66, rho):
    """
    Calculate approximate wave velocities for anisotropic material

    Returns:
    --------
    vp_approx : Approximate P-wave velocity
    vs_approx : Approximate S-wave velocity
    anisotropy_factor : Measure of anisotropy
    """
    vp_approx = np.sqrt(max(C11, C22) / rho)
    vs_approx = np.sqrt(C66 / rho)
    anisotropy_factor = abs(C11 - C22) / (C11 + C22)

    return vp_approx, vs_approx, anisotropy_factor


# ============================================================================
# Part 2: Damage Region Functions
# ============================================================================

def create_damage_regions(X, Y, damage_configs):
    """
    Create damage regions in the computational domain

    Parameters:
    X, Y: 2D coordinate meshgrids
    damage_configs: list of damage configuration dictionaries
        Each dict should contain:
        - 'type': 'circle' or 'square'
        - 'center': (x0, y0) center coordinates
        - 'size': radius for circle, half-width for square
        - 'reduction_factor': factor to reduce stiffness (default 0.5 for 50% reduction)

    Returns:
    damage_mask: boolean array indicating damaged regions
    damage_info: dictionary with detailed damage information
    """
    damage_mask = np.zeros_like(X, dtype=bool)
    damage_regions = []

    for i, config in enumerate(damage_configs):
        damage_type = config['type']
        center = config['center']
        size = config['size']
        reduction_factor = config.get('reduction_factor', 0.5)

        if damage_type == 'circle':
            # Circular damage
            dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
            region_mask = dist <= size

        elif damage_type == 'square':
            # Square damage
            region_mask = (np.abs(X - center[0]) <= size) & (np.abs(Y - center[1]) <= size)

        else:
            raise ValueError(f"Unknown damage type: {damage_type}")

        damage_mask = damage_mask | region_mask

        damage_regions.append({
            'id': i,
            'type': damage_type,
            'center': center,
            'size': size,
            'reduction_factor': reduction_factor,
            'n_points': np.sum(region_mask)
        })

        print(f"Damage {i + 1} ({damage_type}): center=({center[0] * 1e2:.2f}, {center[1] * 1e2:.2f}) cm, "
              f"size={size * 1e2:.2f} cm, reduction={reduction_factor:.1%}, "
              f"points={np.sum(region_mask)}")

    damage_info = {
        'regions': damage_regions,
        'total_damaged_points': np.sum(damage_mask),
        'damage_percentage': 100 * np.sum(damage_mask) / damage_mask.size
    }

    return damage_mask, damage_info


def apply_damage_to_stiffness(C11_base, C22_base, C12_base, C16_base, C26_base, C66_base,
                              damage_mask, reduction_factor=0.5):
    """
    Apply damage to stiffness coefficients

    Parameters:
    C_base: base stiffness coefficients (scalars)
    damage_mask: boolean array indicating damaged regions
    reduction_factor: factor to multiply stiffness in damaged regions

    Returns:
    C11, C22, C12, C16, C26, C66: 2D arrays of stiffness coefficients
    """
    nx, ny = damage_mask.shape

    # Initialize with base values
    C11 = np.full((nx, ny), C11_base)
    C22 = np.full((nx, ny), C22_base)
    C12 = np.full((nx, ny), C12_base)
    C16 = np.full((nx, ny), C16_base)
    C26 = np.full((nx, ny), C26_base)
    C66 = np.full((nx, ny), C66_base)

    # Apply damage (reduce stiffness in damaged regions)
    C11[damage_mask] *= reduction_factor
    C22[damage_mask] *= reduction_factor
    C12[damage_mask] *= reduction_factor
    C16[damage_mask] *= reduction_factor
    C26[damage_mask] *= reduction_factor
    C66[damage_mask] *= reduction_factor

    return C11, C22, C12, C16, C26, C66


def visualize_damage_regions(X, Y, damage_mask, damage_info, output_dir):
    """
    Visualize damage regions in the computational domain
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot damage mask
    im = ax.imshow(damage_mask.T, origin='lower',
                   extent=[X.min() * 1e2, X.max() * 1e2, Y.min() * 1e2, Y.max() * 1e2],
                   cmap='Reds', alpha=0.6)

    # Mark damage centers
    for region in damage_info['regions']:
        center = region['center']
        ax.plot(center[0] * 1e2, center[1] * 1e2, 'b*', markersize=15)

        # Add labels
        if region['type'] == 'circle':
            label = f"Circle {region['id'] + 1}\nR={region['size'] * 1e2:.1f}cm"
        else:
            label = f"Square {region['id'] + 1}\nW={2 * region['size'] * 1e2:.1f}cm"

        ax.text(center[0] * 1e2, center[1] * 1e2 + region['size'] * 1e2 * 1.5,
                label, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('x (cm)', fontsize=12)
    ax.set_ylabel('y (cm)', fontsize=12)
    ax.set_title(f'Damage Regions (Total: {damage_info["damage_percentage"]:.2f}% of domain)',
                 fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.colorbar(im, ax=ax, label='Damaged (1) / Intact (0)')

    fig_path = os.path.join(output_dir, 'damage_regions_fixed_bc.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Damage visualization saved to: {fig_path}")
    return fig_path


# ============================================================================
# Part 3: Mesh Generation - Checkerboard Alternating Diagonal
# ============================================================================

def create_structured_triangular_mesh(nx, ny, Lx, Ly):
    """
    Create structured triangular mesh (node positions identical to FDM)

    Uses checkerboard alternating diagonal pattern to ensure left-right symmetry

    For cells where (i+j) is even: diagonal from bottom-left to top-right (node0 -> node2)
    For cells where (i+j) is odd: diagonal from bottom-right to top-left (node1 -> node3)

    Parameters:
    -----------
    nx, ny : number of grid nodes
    Lx, Ly : computational domain dimensions

    Returns:
    --------
    x, y : coordinate arrays
    elements : triangular element connectivity table
    boundary_nodes : boundary node indices
    dx, dy : grid spacing
    """
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    x = np.linspace(-Lx / 2, Lx / 2, nx)
    y = np.linspace(-Ly / 2, Ly / 2, ny)

    print(f"\nStructured triangular mesh generated (checkerboard alternating diagonal pattern):")
    print(f"  Nodes: {nx} x {ny} = {nx * ny}")
    print(f"  Grid spacing: dx={dx * 1e3:.4f} mm, dy={dy * 1e3:.4f} mm")
    print(f"  Domain: [{x[0] * 1e2:.2f}, {x[-1] * 1e2:.2f}] x [{y[0] * 1e2:.2f}, {y[-1] * 1e2:.2f}] cm")

    # Build triangular element connectivity table
    num_elements = 2 * (nx - 1) * (ny - 1)
    elements = np.zeros((num_elements, 3), dtype=np.int64)

    elem_idx = 0
    for j in range(ny - 1):
        for i in range(nx - 1):
            node0 = j * nx + i  # Bottom-left
            node1 = j * nx + (i + 1)  # Bottom-right
            node2 = (j + 1) * nx + (i + 1)  # Top-right
            node3 = (j + 1) * nx + i  # Top-left

            # Checkerboard alternating diagonal pattern
            if (i + j) % 2 == 0:
                # Even cell: diagonal from bottom-left to top-right (node0 -> node2)
                elements[elem_idx] = [node0, node1, node2]  # Lower triangle
                elements[elem_idx + 1] = [node0, node2, node3]  # Upper triangle
            else:
                # Odd cell: diagonal from bottom-right to top-left (node1 -> node3)
                elements[elem_idx] = [node0, node1, node3]  # Lower triangle
                elements[elem_idx + 1] = [node1, node2, node3]  # Upper triangle

            elem_idx += 2

    print(f"  Triangular elements: {num_elements}")
    print(f"  Diagonal pattern: checkerboard alternating (ensures left-right symmetry)")

    # Identify boundary nodes
    boundary_nodes = set()
    # Bottom edge
    boundary_nodes.update(range(nx))
    # Top edge
    boundary_nodes.update(range((ny - 1) * nx, ny * nx))
    # Left and right edges
    for j in range(ny):
        boundary_nodes.add(j * nx)
        boundary_nodes.add(j * nx + nx - 1)

    boundary_nodes = np.array(sorted(boundary_nodes), dtype=np.int64)
    print(f"  Boundary nodes: {len(boundary_nodes)} ({100 * len(boundary_nodes) / (nx * ny):.1f}%)")

    return x, y, elements, boundary_nodes, dx, dy


# ============================================================================
# Part 4: Damage Region Mesh Visualization Functions
# ============================================================================

def visualize_damage_mesh(x, y, elements, damage_mask, damage_info, output_dir,
                          zoom_region=None, show_nodes=False):
    """
    Visualize finite element mesh in damage regions

    Parameters:
    -----------
    x, y : coordinate arrays
    elements : triangular element connectivity table (num_elements, 3)
    damage_mask : damage mask (nx, ny)
    damage_info : damage information dictionary
    output_dir : output directory
    zoom_region : zoom region [x_min, x_max, y_min, y_max] (unit: m), None for full domain
    show_nodes : whether to display nodes

    Returns:
    --------
    element_in_damage : boolean array marking whether each element is in damage region
    element_centers : center point coordinates of each element
    """
    os.makedirs(output_dir, exist_ok=True)

    nx, ny = len(x), len(y)
    num_elements = len(elements)

    # Compute center point of each element and determine if in damage region
    element_in_damage = np.zeros(num_elements, dtype=bool)
    element_centers = np.zeros((num_elements, 2))

    for elem_idx in range(num_elements):
        node_indices = elements[elem_idx]
        i_coords = node_indices % nx
        j_coords = node_indices // nx

        x_coords = x[i_coords]
        y_coords = y[j_coords]

        # Element center
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        element_centers[elem_idx] = [center_x, center_y]

        # Find nearest grid point
        center_i = np.argmin(np.abs(x - center_x))
        center_j = np.argmin(np.abs(y - center_y))

        # Determine if in damage region
        element_in_damage[elem_idx] = damage_mask[center_i, center_j]

    print(f"\n{'=' * 60}")
    print(f"Damage Region Mesh Statistics")
    print(f"{'=' * 60}")
    print(f"  Total elements: {num_elements}")
    print(f"  Damaged elements: {np.sum(element_in_damage)}")
    print(f"  Damaged element ratio: {100 * np.sum(element_in_damage) / num_elements:.2f}%")

    # ==================== Figure 1: Full domain mesh ====================
    print(f"\nGenerating full domain mesh figure...")
    fig1, ax1 = plt.subplots(figsize=(12, 10))

    # Create polygon collections
    patches_intact = []
    patches_damage = []

    for elem_idx in range(num_elements):
        node_indices = elements[elem_idx]
        i_coords = node_indices % nx
        j_coords = node_indices // nx

        vertices = np.array([
            [x[i_coords[0]] * 1e2, y[j_coords[0]] * 1e2],
            [x[i_coords[1]] * 1e2, y[j_coords[1]] * 1e2],
            [x[i_coords[2]] * 1e2, y[j_coords[2]] * 1e2]
        ])

        if element_in_damage[elem_idx]:
            patches_damage.append(Polygon(vertices, closed=True))
        else:
            patches_intact.append(Polygon(vertices, closed=True))

    # Plot intact region elements (blue border, transparent fill)
    pc_intact = PatchCollection(patches_intact, facecolor='lightblue',
                                edgecolor='blue', linewidth=0.3, alpha=0.3)
    ax1.add_collection(pc_intact)

    # Plot damaged region elements (red fill, dark red border)
    pc_damage = PatchCollection(patches_damage, facecolor='salmon',
                                edgecolor='darkred', linewidth=0.5, alpha=0.7)
    ax1.add_collection(pc_damage)

    # Mark damage region centers and boundaries
    for region in damage_info['regions']:
        center = region['center']
        ax1.plot(center[0] * 1e2, center[1] * 1e2, 'k*', markersize=15,
                 label=f"Damage {region['id'] + 1} center")

        # Draw damage region boundary
        if region['type'] == 'circle':
            theta = np.linspace(0, 2 * np.pi, 100)
            circle_x = center[0] * 1e2 + region['size'] * 1e2 * np.cos(theta)
            circle_y = center[1] * 1e2 + region['size'] * 1e2 * np.sin(theta)
            ax1.plot(circle_x, circle_y, 'k--', linewidth=2, label='Damage boundary')
        elif region['type'] == 'square':
            s = region['size'] * 1e2
            rect_x = [center[0] * 1e2 - s, center[0] * 1e2 + s, center[0] * 1e2 + s,
                      center[0] * 1e2 - s, center[0] * 1e2 - s]
            rect_y = [center[1] * 1e2 - s, center[1] * 1e2 - s, center[1] * 1e2 + s,
                      center[1] * 1e2 + s, center[1] * 1e2 - s]
            ax1.plot(rect_x, rect_y, 'k--', linewidth=2, label='Damage boundary')

    ax1.set_xlim(x.min() * 1e2, x.max() * 1e2)
    ax1.set_ylim(y.min() * 1e2, y.max() * 1e2)
    ax1.set_xlabel('x (cm)', fontsize=12)
    ax1.set_ylabel('y (cm)', fontsize=12)
    ax1.set_title(f'FEM Mesh with Damage Regions\n'
                  f'(Total: {num_elements} elements, Damaged: {np.sum(element_in_damage)} elements)',
                  fontsize=14)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Add legend
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='blue', alpha=0.3, label='Intact elements'),
        Patch(facecolor='salmon', edgecolor='darkred', alpha=0.7, label='Damaged elements'),
        plt.Line2D([0], [0], color='k', linestyle='--', linewidth=2, label='Damage boundary')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)

    fig1_path = os.path.join(output_dir, 'fem_mesh_damage_overview.png')
    plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
    print(f"  Full domain mesh figure saved: {fig1_path}")
    plt.close()

    # ==================== Figure 2: Damage region zoomed view ====================
    print(f"Generating damage region zoomed view...")
    if zoom_region is None:
        # Automatically determine zoom region (centered on first damage region)
        if len(damage_info['regions']) > 0:
            region = damage_info['regions'][0]
            center = region['center']
            size = region['size']
            margin = 3 * size  # Zoom region margin
            zoom_region = [
                center[0] - margin,
                center[0] + margin,
                center[1] - margin,
                center[1] + margin
            ]
        else:
            zoom_region = [x.min(), x.max(), y.min(), y.max()]

    fig2, ax2 = plt.subplots(figsize=(12, 10))

    # Filter elements within zoom region
    for elem_idx in range(num_elements):
        node_indices = elements[elem_idx]
        i_coords = node_indices % nx
        j_coords = node_indices // nx

        x_coords = x[i_coords]
        y_coords = y[j_coords]

        # Check if element is within zoom region
        if (np.max(x_coords) < zoom_region[0] or np.min(x_coords) > zoom_region[1] or
                np.max(y_coords) < zoom_region[2] or np.min(y_coords) > zoom_region[3]):
            continue

        vertices = np.array([
            [x_coords[0] * 1e2, y[j_coords[0]] * 1e2],
            [x_coords[1] * 1e2, y[j_coords[1]] * 1e2],
            [x_coords[2] * 1e2, y[j_coords[2]] * 1e2]
        ])

        if element_in_damage[elem_idx]:
            color = 'salmon'
            edge_color = 'darkred'
            alpha = 0.7
        else:
            color = 'lightblue'
            edge_color = 'blue'
            alpha = 0.5

        triangle = Polygon(vertices, closed=True, facecolor=color,
                           edgecolor=edge_color, linewidth=1, alpha=alpha)
        ax2.add_patch(triangle)

    # Display nodes (optional)
    if show_nodes:
        X, Y = np.meshgrid(x, y, indexing='ij')
        in_zoom = ((X >= zoom_region[0]) & (X <= zoom_region[1]) &
                   (Y >= zoom_region[2]) & (Y <= zoom_region[3]))
        ax2.scatter(X[in_zoom] * 1e2, Y[in_zoom] * 1e2, s=5, c='black',
                    zorder=5, label='Nodes')

    # Mark damage region boundaries
    for region in damage_info['regions']:
        center = region['center']
        if region['type'] == 'circle':
            theta = np.linspace(0, 2 * np.pi, 100)
            circle_x = center[0] * 1e2 + region['size'] * 1e2 * np.cos(theta)
            circle_y = center[1] * 1e2 + region['size'] * 1e2 * np.sin(theta)
            ax2.plot(circle_x, circle_y, 'k-', linewidth=2.5)
        elif region['type'] == 'square':
            s = region['size'] * 1e2
            rect_x = [center[0] * 1e2 - s, center[0] * 1e2 + s, center[0] * 1e2 + s,
                      center[0] * 1e2 - s, center[0] * 1e2 - s]
            rect_y = [center[1] * 1e2 - s, center[1] * 1e2 - s, center[1] * 1e2 + s,
                      center[1] * 1e2 + s, center[1] * 1e2 - s]
            ax2.plot(rect_x, rect_y, 'k-', linewidth=2.5)

    ax2.set_xlim(zoom_region[0] * 1e2, zoom_region[1] * 1e2)
    ax2.set_ylim(zoom_region[2] * 1e2, zoom_region[3] * 1e2)
    ax2.set_xlabel('x (cm)', fontsize=12)
    ax2.set_ylabel('y (cm)', fontsize=12)
    ax2.set_title(f'FEM Mesh - Zoomed View of Damage Region\n'
                  f'(Checkerboard Alternating Diagonal Pattern)', fontsize=14)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Legend
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='blue', alpha=0.5, label='Intact elements'),
        Patch(facecolor='salmon', edgecolor='darkred', alpha=0.7, label='Damaged elements'),
        plt.Line2D([0], [0], color='k', linewidth=2.5, label='Damage boundary')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)

    fig2_path = os.path.join(output_dir, 'fem_mesh_damage_zoomed.png')
    plt.savefig(fig2_path, dpi=200, bbox_inches='tight')
    print(f"  Zoomed mesh figure saved: {fig2_path}")
    plt.close()

    # ==================== Figure 3: Mesh detail view (with element numbers) ====================
    print(f"Generating mesh detail view...")
    if len(damage_info['regions']) > 0:
        region = damage_info['regions'][0]
        center = region['center']
        detail_size = region['size'] * 0.5  # Smaller zoom region
        detail_region = [
            center[0] - detail_size,
            center[0] + detail_size,
            center[1] - detail_size,
            center[1] + detail_size
        ]

        fig3, ax3 = plt.subplots(figsize=(12, 10))

        # Collect elements and nodes within detail region
        nodes_in_region = set()
        elements_in_region = []

        for elem_idx in range(num_elements):
            node_indices = elements[elem_idx]
            i_coords = node_indices % nx
            j_coords = node_indices // nx

            x_coords = x[i_coords]
            y_coords = y[j_coords]

            # Check if element is within detail region
            if (np.max(x_coords) < detail_region[0] or np.min(x_coords) > detail_region[1] or
                    np.max(y_coords) < detail_region[2] or np.min(y_coords) > detail_region[3]):
                continue

            elements_in_region.append(elem_idx)
            for ni in node_indices:
                nodes_in_region.add(ni)

            vertices = np.array([
                [x_coords[0] * 1e2, y[j_coords[0]] * 1e2],
                [x_coords[1] * 1e2, y[j_coords[1]] * 1e2],
                [x_coords[2] * 1e2, y[j_coords[2]] * 1e2]
            ])

            if element_in_damage[elem_idx]:
                color = 'salmon'
                edge_color = 'darkred'
            else:
                color = 'lightblue'
                edge_color = 'blue'

            triangle = Polygon(vertices, closed=True, facecolor=color,
                               edgecolor=edge_color, linewidth=1.5, alpha=0.6)
            ax3.add_patch(triangle)

            # Label element number
            center_x = np.mean(x_coords) * 1e2
            center_y = np.mean(y_coords) * 1e2
            ax3.text(center_x, center_y, str(elem_idx), fontsize=6,
                     ha='center', va='center', color='black', fontweight='bold')

        # Display nodes
        for node_id in nodes_in_region:
            i = node_id % nx
            j = node_id // nx
            node_x = x[i] * 1e2
            node_y = y[j] * 1e2

            # Determine if node is in damage region
            if damage_mask[i, j]:
                ax3.scatter(node_x, node_y, s=30, c='red', zorder=5, marker='o')
            else:
                ax3.scatter(node_x, node_y, s=30, c='blue', zorder=5, marker='o')

        # Mark damage boundary
        if region['type'] == 'circle':
            theta = np.linspace(0, 2 * np.pi, 100)
            circle_x = center[0] * 1e2 + region['size'] * 1e2 * np.cos(theta)
            circle_y = center[1] * 1e2 + region['size'] * 1e2 * np.sin(theta)
            ax3.plot(circle_x, circle_y, 'k--', linewidth=2)
        elif region['type'] == 'square':
            s = region['size'] * 1e2
            rect_x = [center[0] * 1e2 - s, center[0] * 1e2 + s, center[0] * 1e2 + s,
                      center[0] * 1e2 - s, center[0] * 1e2 - s]
            rect_y = [center[1] * 1e2 - s, center[1] * 1e2 - s, center[1] * 1e2 + s,
                      center[1] * 1e2 + s, center[1] * 1e2 - s]
            ax3.plot(rect_x, rect_y, 'k--', linewidth=2)

        ax3.set_xlim(detail_region[0] * 1e2, detail_region[1] * 1e2)
        ax3.set_ylim(detail_region[2] * 1e2, detail_region[3] * 1e2)
        ax3.set_xlabel('x (cm)', fontsize=12)
        ax3.set_ylabel('y (cm)', fontsize=12)
        ax3.set_title(f'FEM Mesh Detail View\n'
                      f'({len(elements_in_region)} elements, {len(nodes_in_region)} nodes shown)',
                      fontsize=14)
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)

        # Legend
        legend_elements = [
            Patch(facecolor='lightblue', edgecolor='blue', alpha=0.6, label='Intact elements'),
            Patch(facecolor='salmon', edgecolor='darkred', alpha=0.6, label='Damaged elements'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                       markersize=8, label='Intact nodes'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                       markersize=8, label='Damaged nodes'),
        ]
        ax3.legend(handles=legend_elements, loc='upper right', fontsize=10)

        fig3_path = os.path.join(output_dir, 'fem_mesh_damage_detail.png')
        plt.savefig(fig3_path, dpi=200, bbox_inches='tight')
        print(f"  Detail mesh figure saved: {fig3_path}")
        plt.close()

    return element_in_damage, element_centers


def visualize_mesh_pattern(x, y, elements, output_dir):
    """
    Visualize checkerboard alternating diagonal mesh pattern
    """
    os.makedirs(output_dir, exist_ok=True)

    nx = len(x)

    # Only plot a small region near the center to show the pattern
    center_i = nx // 2
    center_j = nx // 2
    half_range = 3  # Show 6x6 cells

    fig, ax = plt.subplots(figsize=(10, 10))

    colors_even = ['#FFB3B3', '#B3FFB3']  # Even cell triangle colors
    colors_odd = ['#B3B3FF', '#FFFFB3']  # Odd cell triangle colors

    for j in range(center_j - half_range, center_j + half_range):
        for i in range(center_i - half_range, center_i + half_range):
            if i < 0 or i >= nx - 1 or j < 0 or j >= nx - 1:
                continue

            # Four corner node coordinates
            x0, x1 = x[i] * 1e2, x[i + 1] * 1e2
            y0, y1 = y[j] * 1e2, y[j + 1] * 1e2

            if (i + j) % 2 == 0:
                # Even: diagonal from bottom-left to top-right
                tri1 = Polygon([[x0, y0], [x1, y0], [x1, y1]], closed=True,
                               facecolor=colors_even[0], edgecolor='black', linewidth=1.5)
                tri2 = Polygon([[x0, y0], [x1, y1], [x0, y1]], closed=True,
                               facecolor=colors_even[1], edgecolor='black', linewidth=1.5)
            else:
                # Odd: diagonal from bottom-right to top-left
                tri1 = Polygon([[x0, y0], [x1, y0], [x0, y1]], closed=True,
                               facecolor=colors_odd[0], edgecolor='black', linewidth=1.5)
                tri2 = Polygon([[x1, y0], [x1, y1], [x0, y1]], closed=True,
                               facecolor=colors_odd[1], edgecolor='black', linewidth=1.5)

            ax.add_patch(tri1)
            ax.add_patch(tri2)

            # Label (i+j) parity
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            ax.text(cx, cy, f'({i},{j})\n{"Even" if (i + j) % 2 == 0 else "Odd"}',
                    ha='center', va='center', fontsize=8)

    # Plot nodes
    for j in range(center_j - half_range, center_j + half_range + 1):
        for i in range(center_i - half_range, center_i + half_range + 1):
            if 0 <= i < nx and 0 <= j < nx:
                ax.scatter(x[i] * 1e2, y[j] * 1e2, s=50, c='black', zorder=5)

    ax.set_xlabel('x (cm)', fontsize=12)
    ax.set_ylabel('y (cm)', fontsize=12)
    ax.set_title('Checkerboard Alternating Diagonal Pattern\n'
                 '(Even: diagonal, Odd: diagonal)', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Legend
    legend_elements = [
        Patch(facecolor=colors_even[0], edgecolor='black', label='Even cell - Lower triangle'),
        Patch(facecolor=colors_even[1], edgecolor='black', label='Even cell - Upper triangle'),
        Patch(facecolor=colors_odd[0], edgecolor='black', label='Odd cell - Lower triangle'),
        Patch(facecolor=colors_odd[1], edgecolor='black', label='Odd cell - Upper triangle'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    fig_path = os.path.join(output_dir, 'mesh_pattern_checkerboard.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nMesh pattern figure saved: {fig_path}")
    plt.close()


# ============================================================================
# Part 5: FEM Solver (Anisotropic Material - Ricker Wavelet - Fixed BC - With Damage)
# ============================================================================

def solve_2d_anisotropic_wave_fem_ricker_fixed(
        nx=301,
        ny=301,
        Lx=0.4,
        Ly=0.4,
        nt=1001,
        t_total=30e-6,
        source_amplitude=1e13,
        source_f0=170e3,
        source_t0=6e-6,
        gaussian_param=5e8,
        C11=16.5e10,
        C22=6.2e10,
        C12=5.0e10,
        C16=0.0,
        C26=0.0,
        C66=3.96e10,
        rho=7100.0,
        source_position=(0.0, 0.0),
        damage_configs=None,
        device=None
):
    """
    Solve 2D anisotropic elastic wave equation using FEM
    (Ricker wavelet + fixed boundary + damage regions)

    Parameters:
    -----------
    nx, ny : number of grid nodes
    Lx, Ly : computational domain dimensions (m)
    nt : number of time steps
    t_total : total time (s)
    source_amplitude : Ricker wavelet amplitude (N)
    source_f0 : Ricker wavelet dominant frequency (Hz)
    source_t0 : Ricker wavelet time delay (s)
    gaussian_param : Gaussian parameter (spatial distribution)
    C11, C22, C12, C16, C26, C66 : stiffness matrix coefficients (Pa)
    rho : density (kg/m^3)
    source_position : force source position
    damage_configs : damage configuration list
    device : computation device

    Returns:
    --------
    t : time array
    u_x_cpu, u_y_cpu : displacement fields
    other info...
    """
    start_time = time.time()

    print("\n" + "=" * 70)
    print("2D Anisotropic Elastic Wave Equation - FEM Solver (GPU Accelerated) + Damage Regions")
    print("Source: Ricker wavelet | BC: Fixed boundary (Dirichlet: u=0)")
    print("Checkerboard alternating diagonal mesh")
    print("=" * 70)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nComputation device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(device.index if device.index else 0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")

    # Compute wave velocities (using base stiffness)
    vp_approx, vs_approx, anisotropy_factor = calculate_wave_velocities_anisotropic(
        C11, C22, C12, C16, C26, C66, rho)

    print(f"\nMaterial parameters (anisotropic - base values):")
    print(f"  Stiffness coefficients:")
    print(f"    C11 = {C11 / 1e10:.2f} x 10^10 Pa, C22 = {C22 / 1e10:.2f} x 10^10 Pa")
    print(f"    C12 = {C12 / 1e10:.2f} x 10^10 Pa, C66 = {C66 / 1e10:.2f} x 10^10 Pa")
    print(f"    C16 = {C16 / 1e10:.2f} x 10^10 Pa, C26 = {C26 / 1e10:.2f} x 10^10 Pa")
    print(f"  Density: rho = {rho:.1f} kg/m^3")
    print(f"  Wave velocities:")
    print(f"    P-wave vp ~ {vp_approx:.1f} m/s")
    print(f"    S-wave vs ~ {vs_approx:.1f} m/s")
    print(f"  Anisotropy factor = {anisotropy_factor:.3f}")

    # Generate mesh
    x, y, elements, boundary_nodes, dx, dy = create_structured_triangular_mesh(nx, ny, Lx, Ly)

    # Create 2D coordinate meshgrid
    X, Y = np.meshgrid(x, y, indexing='ij')

    # ============= Create damage regions =============
    if damage_configs is None:
        # Default: one circular damage
        damage_configs = [
            {
                'type': 'circle',
                'center': (0.1, 0.1),  # 10cm, 10cm from center
                'size': 0.02,  # 2cm radius
                'reduction_factor': 0.5  # 50% stiffness reduction
            }
        ]

    print(f"\n{'=' * 60}")
    print(f"Creating Damage Regions")
    print(f"{'=' * 60}")

    damage_mask, damage_info = create_damage_regions(X, Y, damage_configs)

    print(f"Total damaged points: {damage_info['total_damaged_points']}")
    print(f"Damage ratio: {damage_info['damage_percentage']:.2f}%")

    # Apply damage to stiffness coefficients
    C11_field, C22_field, C12_field, C16_field, C26_field, C66_field = apply_damage_to_stiffness(
        C11, C22, C12, C16, C26, C66,
        damage_mask, reduction_factor=0.5
    )

    print(f"\nStiffness fields created (spatially varying)")
    print(f"  Intact region: 100% stiffness")
    print(f"  Damaged region: 50% stiffness")

    # Time discretization
    dt = t_total / (nt - 1)
    t = np.linspace(0, t_total, nt)

    # CFL condition check
    max_velocity = max(vp_approx, vs_approx)
    cfl_number = max_velocity * dt / min(dx, dy)

    if cfl_number > 0.5:
        dt_stable = 0.5 * min(dx, dy) / max_velocity
        print(f"\nWarning: CFL number = {cfl_number:.4f} > 0.5")
        print(f"  Recommended time step: dt <= {dt_stable:.6e}")

        nt_new = int(t_total / dt_stable) + 1
        dt = t_total / (nt_new - 1)
        t = np.linspace(0, t_total, nt_new)
        nt = nt_new
        print(f"  Adjusted to: dt = {dt:.6e} s, nt = {nt}")

    print(f"\nComputation parameters:")
    print(f"  dx = {dx * 1e3:.4f} mm, dy = {dy * 1e3:.4f} mm")
    print(f"  dt = {dt * 1e9:.3f} ns")
    print(f"  CFL = {max_velocity * dt / min(dx, dy):.4f}")

    # Force source setup
    x0_src, y0_src = source_position
    gaussian_distribution = gaussian_point_load(X, Y, x0_src, y0_src, gaussian_param)

    fwhm = calculate_fwhm(gaussian_param)
    print(f"\nForce source parameters (Ricker wavelet):")
    print(f"  Position: ({x0_src * 1e2:.2f}, {y0_src * 1e2:.2f}) cm")
    print(f"  Dominant frequency: f0 = {source_f0 / 1e3:.1f} kHz")
    print(f"  Time delay: t0 = {source_t0 * 1e6:.1f} us")
    print(f"  Amplitude: {source_amplitude:.2e} N")
    print(f"  Spatial distribution FWHM: {fwhm * 1e3:.2f} mm")

    # Visualize Gaussian distribution and Ricker wavelet
    output_dir = "./data_anisotropic_fem_ricker_fixed_bc_damage"
    os.makedirs(output_dir, exist_ok=True)
    plot_gaussian_distribution_func(x, y, gaussian_distribution, x0_src, y0_src, fwhm,
                                    damage_mask, output_dir)
    plot_ricker_wavelet(t, source_f0, source_t0, source_amplitude, output_dir)

    # Visualize damage regions
    visualize_damage_regions(X, Y, damage_mask, damage_info, output_dir)

    # Transfer to GPU
    print(f"\nStep 1: Transferring data to {device}...")

    # Gaussian distribution
    gaussian_nodal_flat = torch.from_numpy(gaussian_distribution.T.flatten()).float().to(device)

    elements_gpu = torch.from_numpy(elements).long().to(device)
    boundary_nodes_gpu = torch.from_numpy(boundary_nodes).long().to(device)

    # Build lumped mass matrix
    print("\nStep 2: Building lumped mass matrix...")
    num_nodes = nx * ny
    num_elements = len(elements)
    element_area = dx * dy / 2.0

    M_lumped = torch.zeros(num_nodes, dtype=torch.float32, device=device)

    for i in range(3):
        nodes = elements_gpu[:, i]
        node_masses = torch.full((num_elements,), rho * element_area / 3.0,
                                 dtype=torch.float32, device=device)
        M_lumped.index_add_(0, nodes, node_masses)

    print(f"  Mass range: [{M_lumped.min():.6e}, {M_lumped.max():.6e}]")
    print(f"  Average mass: {M_lumped.mean():.6e}")

    # Pre-compute triangle element Jacobians and B matrices
    print("\nStep 3: Pre-computing element information...")

    # Gauss integration points (3-point)
    gauss_points = np.array([
        [1.0 / 6.0, 1.0 / 6.0],
        [2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0]
    ])
    gauss_weights = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])

    def triangle_shape_functions(xi, eta):
        """Triangle element shape function derivatives"""
        dN_dxi = np.array([-1.0, 1.0, 0.0])
        dN_deta = np.array([-1.0, 0.0, 1.0])
        return dN_dxi, dN_deta

    # Pre-compute Jacobian for each element
    detJ_all = np.zeros(num_elements)
    J_inv_all = np.zeros((num_elements, 2, 2))

    # Compute center point coordinates of each element (for obtaining stiffness values)
    element_centers_i = np.zeros(num_elements, dtype=np.int64)
    element_centers_j = np.zeros(num_elements, dtype=np.int64)

    for elem_idx in range(num_elements):
        node_indices = elements[elem_idx]
        i_coords = node_indices % nx
        j_coords = node_indices // nx

        x_coords = x[i_coords]
        y_coords = y[j_coords]

        J = np.array([
            [x_coords[1] - x_coords[0], x_coords[2] - x_coords[0]],
            [y_coords[1] - y_coords[0], y_coords[2] - y_coords[0]]
        ])

        detJ = np.linalg.det(J)
        detJ_all[elem_idx] = detJ

        if detJ < 0:
            print(f"  Warning: element {elem_idx} has detJ = {detJ:.6e} < 0")

        J_inv_all[elem_idx] = np.linalg.inv(J)

        # Compute grid index of element center point
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        center_i = np.argmin(np.abs(x - center_x))
        center_j = np.argmin(np.abs(y - center_y))
        element_centers_i[elem_idx] = center_i
        element_centers_j[elem_idx] = center_j

    detJ_gpu = torch.from_numpy(np.abs(detJ_all)).float().to(device)
    J_inv_gpu = torch.from_numpy(J_inv_all).float().to(device)

    print(f"  detJ range: [{np.abs(detJ_all).min():.6e}, {np.abs(detJ_all).max():.6e}]")

    # Get stiffness coefficients for each element (based on element center point)
    C11_elem = torch.from_numpy(C11_field[element_centers_i, element_centers_j]).float().to(device)
    C22_elem = torch.from_numpy(C22_field[element_centers_i, element_centers_j]).float().to(device)
    C12_elem = torch.from_numpy(C12_field[element_centers_i, element_centers_j]).float().to(device)
    C16_elem = torch.from_numpy(C16_field[element_centers_i, element_centers_j]).float().to(device)
    C26_elem = torch.from_numpy(C26_field[element_centers_i, element_centers_j]).float().to(device)
    C66_elem = torch.from_numpy(C66_field[element_centers_i, element_centers_j]).float().to(device)

    print(f"\nElement stiffness coefficients computed (spatially varying)")
    print(f"  C11: [{C11_elem.min() / 1e10:.2f}, {C11_elem.max() / 1e10:.2f}] x 10^10 Pa")
    print(f"  C22: [{C22_elem.min() / 1e10:.2f}, {C22_elem.max() / 1e10:.2f}] x 10^10 Pa")

    # Pre-compute B matrices
    B_matrices_gp = []
    for gp_idx in range(3):
        xi, eta = gauss_points[gp_idx]
        dN_dxi, dN_deta = triangle_shape_functions(xi, eta)

        dN_dxi_gpu = torch.from_numpy(dN_dxi).float().to(device)
        dN_deta_gpu = torch.from_numpy(dN_deta).float().to(device)

        B_matrices_gp.append((dN_dxi_gpu, dN_deta_gpu))

    gauss_weights_gpu = torch.from_numpy(gauss_weights).float().to(device)

    print(f"\nMaterial matrix C (x 10^10 Pa) - base values:")
    print(f"  [{C11 / 1e10:.2f}  {C12 / 1e10:.2f}  {C16 / 1e10:.2f}]")
    print(f"  [{C12 / 1e10:.2f}  {C22 / 1e10:.2f}  {C26 / 1e10:.2f}]")
    print(f"  [{C16 / 1e10:.2f}  {C26 / 1e10:.2f}  {C66 / 1e10:.2f}]")

    # Internal force operator (anisotropic - spatially varying stiffness)
    def apply_stiffness_operator(u_x_flat, u_y_flat):
        """Compute internal forces (anisotropic material - spatially varying stiffness)"""
        f_x = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        f_y = torch.zeros(num_nodes, dtype=torch.float32, device=device)

        for gp_idx in range(3):
            dN_dxi, dN_deta = B_matrices_gp[gp_idx]
            weight = gauss_weights_gpu[gp_idx]

            u_elem_x = u_x_flat[elements_gpu]
            u_elem_y = u_y_flat[elements_gpu]

            # Compute shape function derivatives in physical space
            dN_dx = J_inv_gpu[:, 0, 0].unsqueeze(1) * dN_dxi.unsqueeze(0) + \
                    J_inv_gpu[:, 1, 0].unsqueeze(1) * dN_deta.unsqueeze(0)
            dN_dy = J_inv_gpu[:, 0, 1].unsqueeze(1) * dN_dxi.unsqueeze(0) + \
                    J_inv_gpu[:, 1, 1].unsqueeze(1) * dN_deta.unsqueeze(0)

            # Strain components
            eps_xx = torch.sum(dN_dx * u_elem_x, dim=1)
            eps_yy = torch.sum(dN_dy * u_elem_y, dim=1)
            gamma_xy = torch.sum(dN_dy * u_elem_x + dN_dx * u_elem_y, dim=1)

            # Anisotropic stress-strain relation (using spatially varying stiffness)
            sigma_xx = C11_elem * eps_xx + C12_elem * eps_yy + C16_elem * gamma_xy
            sigma_yy = C12_elem * eps_xx + C22_elem * eps_yy + C26_elem * gamma_xy
            tau_xy = C16_elem * eps_xx + C26_elem * eps_yy + C66_elem * gamma_xy

            factor = detJ_gpu * weight

            f_elem_x = factor.unsqueeze(1) * (dN_dx * sigma_xx.unsqueeze(1) +
                                              dN_dy * tau_xy.unsqueeze(1))
            f_elem_y = factor.unsqueeze(1) * (dN_dy * sigma_yy.unsqueeze(1) +
                                              dN_dx * tau_xy.unsqueeze(1))

            for i in range(3):
                nodes = elements_gpu[:, i]
                f_x.index_add_(0, nodes, f_elem_x[:, i])
                f_y.index_add_(0, nodes, f_elem_y[:, i])

        return f_x, f_y

    # Time factors (pre-compute Ricker wavelet)
    print("\nStep 4: Pre-computing Ricker wavelet time factors...")
    time_factors = torch.zeros(nt, dtype=torch.float32, device=device)
    for n in range(nt):
        time_factors[n] = ricker_wavelet(t[n], source_f0, source_t0, source_amplitude)

    # Force source normalization
    print("\nStep 5: Force source normalization...")
    rho_inv = 1.0 / rho
    print(f"  Inverse density: 1/rho = {rho_inv:.6e}")
    print(f"  Normalization strategy: F = ricker(t) x gaussian x M_lumped / rho")

    # Time integration (fixed boundary condition)
    print(f"\nStep 6: Time integration (central difference - fixed BC - with damage)...")
    print(f"  Fixed BC: boundary node displacements forced to 0")
    print(f"  Boundary nodes: {len(boundary_nodes)}")

    u_x_flat = torch.zeros((num_nodes, nt), dtype=torch.float32, device=device)
    u_y_flat = torch.zeros((num_nodes, nt), dtype=torch.float32, device=device)

    time_integration_start = time.time()

    for n in range(1, nt - 1):
        # External force computation (Ricker wavelet)
        acc_ext_y = time_factors[n] * gaussian_nodal_flat * rho_inv
        f_ext_y = acc_ext_y * M_lumped

        # Internal forces
        f_int_x, f_int_y = apply_stiffness_operator(u_x_flat[:, n], u_y_flat[:, n])

        # Update displacement (central difference)
        u_x_flat[:, n + 1] = 2 * u_x_flat[:, n] - u_x_flat[:, n - 1] - \
                             (dt * dt) * f_int_x / M_lumped
        u_y_flat[:, n + 1] = 2 * u_y_flat[:, n] - u_y_flat[:, n - 1] + \
                             (dt * dt) * (f_ext_y - f_int_y) / M_lumped

        # ========== Fixed BC: force boundary node displacements to 0 ==========
        u_x_flat[boundary_nodes_gpu, n + 1] = 0
        u_y_flat[boundary_nodes_gpu, n + 1] = 0
        # ======================================================================

        # Progress display
        if n % max(1, nt // 10) == 0:
            elapsed = time.time() - time_integration_start
            eta = elapsed / n * (nt - n)
            max_ux = torch.abs(u_x_flat[:, n + 1]).max().item()
            max_uy = torch.abs(u_y_flat[:, n + 1]).max().item()

            if torch.isnan(u_x_flat[:, n + 1]).any():
                print(f"\nError: NaN encountered at time step {n}!")
                break

            print(f"  Progress: {n}/{nt} ({100 * n / nt:.1f}%) | "
                  f"Elapsed: {elapsed:.1f}s | Remaining: {eta:.1f}s | "
                  f"max|u_x|: {max_ux:.2e} | max|u_y|: {max_uy:.2e}")

    time_integration_time = time.time() - time_integration_start

    # Reshape to (nx, ny, nt) format
    print("\nStep 7: Reshaping data format...")
    u_x_temp = u_x_flat.reshape(ny, nx, nt)
    u_y_temp = u_y_flat.reshape(ny, nx, nt)

    u_x = torch.transpose(u_x_temp, 0, 1)
    u_y = torch.transpose(u_y_temp, 0, 1)

    u_x_cpu = 0.7 * u_x.cpu().numpy()
    u_y_cpu = 0.7 * u_y.cpu().numpy()

    # Compute total displacement
    total_displacement = np.sqrt(u_x_cpu ** 2 + u_y_cpu ** 2)

    # ========== Verify boundary condition ==========
    print("\nStep 8: Verifying fixed boundary condition...")
    t_check = -1
    u_x_boundary = u_x_cpu[boundary_nodes % nx, boundary_nodes // nx, t_check]
    u_y_boundary = u_y_cpu[boundary_nodes % nx, boundary_nodes // nx, t_check]

    max_u_boundary = max(np.abs(u_x_boundary).max(), np.abs(u_y_boundary).max())
    print(f"  Max displacement at boundary nodes: {max_u_boundary:.2e} m")

    if max_u_boundary < 1e-15:
        print("  Fixed boundary condition strictly satisfied (u = 0 on boundary)")
    else:
        print(f"  Boundary condition has small error: {max_u_boundary:.2e}")

    total_time = time.time() - start_time

    print(f"\n" + "=" * 70)
    print(f"FEM computation completed (fixed BC + damage regions)!")
    print(f"=" * 70)
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Time integration: {time_integration_time:.2f} seconds")
    print(f"  Average per step: {time_integration_time / nt:.4f} seconds")
    print(f"  Output shape: u_x={u_x_cpu.shape}, u_y={u_y_cpu.shape}")
    print(f"\nDisplacement statistics:")
    print(f"  u_x: range [{u_x_cpu.min():.6e}, {u_x_cpu.max():.6e}]")
    print(f"  u_y: range [{u_y_cpu.min():.6e}, {u_y_cpu.max():.6e}]")
    print(f"  |u|: max {total_displacement.max():.6e}")

    # Build node information
    nodes = np.zeros((nx * ny, 2))
    for j in range(ny):
        for i in range(nx):
            node_id = j * nx + i
            nodes[node_id, 0] = x[i]
            nodes[node_id, 1] = y[j]

    # Source mask
    source_mask = gaussian_distribution > 0.01 * np.max(gaussian_distribution)
    i_center = np.argmin(np.abs(x - x0_src))
    j_center = np.argmin(np.abs(y - y0_src))
    source_indices = [j_center * nx + i_center]

    source_info = {
        'source_amplitude': source_amplitude,
        'source_f0': source_f0,
        'source_t0': source_t0,
        'gaussian_param': gaussian_param,
        'fwhm': fwhm,
        'C11': C11, 'C22': C22, 'C12': C12, 'C16': C16, 'C26': C26, 'C66': C66,
        'rho': rho, 'vp_approx': vp_approx, 'vs_approx': vs_approx,
        'anisotropy_factor': anisotropy_factor,
        'boundary_condition': 'fixed',
        'max_u_boundary': max_u_boundary,
        'damage_mask': damage_mask,
        'damage_info': damage_info,
        'damage_configs': damage_configs
    }

    return x, y, t, u_x_cpu, u_y_cpu, total_displacement, source_mask, source_indices, nodes, elements, source_info


# ============================================================================
# Part 6: Visualization Functions
# ============================================================================

def plot_gaussian_distribution_func(x, y, gaussian_distribution, x0_src, y0_src, fwhm,
                                    damage_mask, output_dir):
    """Plot Gaussian distribution (with damage region overlay)"""
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Create coordinate meshgrid for overlay
    X, Y = np.meshgrid(x, y, indexing='ij')

    # 2D distribution
    im1 = ax1.imshow(gaussian_distribution.T, origin='lower',
                     extent=[x.min() * 1e2, x.max() * 1e2, y.min() * 1e2, y.max() * 1e2],
                     cmap='hot')
    ax1.set_title('Gaussian Function Spatial Distribution\n(Anisotropic Material with Damage - FEM Fixed BC)',
                  fontsize=14)
    ax1.set_xlabel('x (cm)', fontsize=12)
    ax1.set_ylabel('y (cm)', fontsize=12)
    ax1.plot(x0_src * 1e2, y0_src * 1e2, 'b*', markersize=10, label='Load Center')

    # Overlay damage region contour
    ax1.contour(X.T * 1e2, Y.T * 1e2, damage_mask.T, levels=[0.5], colors='cyan', linewidths=2)

    plt.colorbar(im1, ax=ax1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # x cross-section
    i_center = np.argmin(np.abs(x - x0_src))
    gaussian_1d = gaussian_distribution[i_center, :]
    max_gaussian = np.max(gaussian_distribution)

    ax2.plot(y * 1e2, gaussian_1d, 'r-', linewidth=2, label='Gaussian Function')
    ax2.axhline(y=max_gaussian / 2, color='g', linestyle='--', alpha=0.7, label='Half Maximum')
    ax2.axvline(x=(y0_src - fwhm / 2) * 1e2, color='g', linestyle='--', alpha=0.7)
    ax2.axvline(x=(y0_src + fwhm / 2) * 1e2, color='g', linestyle='--', alpha=0.7)
    ax2.set_title(f'Gaussian Function Cross-section at x={x0_src * 1e2:.1f} cm', fontsize=14)
    ax2.set_xlabel('y (cm)', fontsize=12)
    ax2.set_ylabel('Intensity', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'gaussian_distribution_anisotropic_fem_fixed_bc_damage.png')
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Gaussian distribution figure saved: {output_path}")


def plot_ricker_wavelet(t, f0, t0, amplitude, output_dir):
    """Plot Ricker wavelet"""
    os.makedirs(output_dir, exist_ok=True)

    ricker = np.array([ricker_wavelet(ti, f0, t0, amplitude) for ti in t])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(t * 1e6, ricker, 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=t0 * 1e6, color='r', linestyle='--', alpha=0.7, label=f't0 = {t0 * 1e6:.1f} us')
    ax.set_title(f'Ricker Wavelet (f0 = {f0 / 1e3:.0f} kHz, Amplitude = {amplitude:.1e} N)', fontsize=14)
    ax.set_xlabel('t (us)', fontsize=12)
    ax.set_ylabel('G_time(t) (N)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    max_idx = np.argmax(ricker)
    ax.plot(t[max_idx] * 1e6, ricker[max_idx], 'ro', markersize=8)
    ax.annotate(f'Peak: {ricker[max_idx]:.2e} N',
                xy=(t[max_idx] * 1e6, ricker[max_idx]),
                xytext=(t[max_idx] * 1e6 + 2, ricker[max_idx] * 0.9),
                fontsize=10)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'ricker_wavelet_fixed_bc_damage.png')
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Ricker wavelet figure saved: {output_path}")


def visualize_wave_propagation(x, y, t, u_x, u_y, total_displacement, source_mask, source_indices,
                               nodes, elements, source_info=None,
                               output_dir="./data_anisotropic_fem_ricker_fixed_bc_damage"):
    """
    Visualize wave field propagation (FEM version - fixed BC + damage regions)
    Displays displacement magnitude

    Uses contour(damage_mask) to draw the outer contour of damage regions,
    avoiding internal boundary lines when drawing L-shaped or composite regions.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    C11 = source_info.get('C11', 16.5e10)
    C22 = source_info.get('C22', 6.2e10)
    vp_approx = source_info.get('vp_approx', 5000)
    vs_approx = source_info.get('vs_approx', 2400)
    source_f0 = source_info.get('source_f0', 170e3)
    damage_mask = source_info.get('damage_mask', None)
    damage_info = source_info.get('damage_info', None)

    # Create coordinate meshgrid for contour overlay
    X, Y = np.meshgrid(x, y, indexing='ij')

    t_max = t[-1]
    target_times = [t_max * 0.25, t_max * 0.5, t_max * 0.75, t_max * 1.0]

    time_indices = []
    for target_t in target_times:
        idx = np.argmin(np.abs(t - target_t))
        time_indices.append(idx)
        print(f"Target time {target_t * 1e6:.2f} us -> Actual time {t[idx] * 1e6:.2f} us (index {idx})")

    # Convert damage_mask to float with slight Gaussian smoothing
    # to ensure contour draws connected region outer contour without internal dividing lines
    damage_mask_float = None
    if damage_mask is not None:
        from scipy.ndimage import binary_fill_holes, gaussian_filter
        # Convert to float with slight smoothing to remove discrete grid jaggedness at junctions
        damage_mask_float = gaussian_filter(damage_mask.astype(np.float64), sigma=0.9)

    # ==================== Figure 1: Displacement magnitude ====================
    fig, axes = plt.subplots(1, 4, figsize=(16, 10))
    axes = axes.flatten()

    def get_order_of_magnitude(val):
        if val == 0:
            return 0
        return int(np.floor(np.log10(abs(val))))

    for idx, (t_idx, target_t) in enumerate(zip(time_indices, target_times)):
        field = np.sqrt(u_x[:, :, t_idx] ** 2 + u_y[:, :, t_idx] ** 2).T

        vmax = field.max()
        vmin = 0

        if vmax < 1e-15:
            vmax = 1e-15

        im = axes[idx].imshow(field, origin='lower',
                              extent=[x.min() * 1e2, x.max() * 1e2,
                                      y.min() * 1e2, y.max() * 1e2],
                              cmap='gray_r', vmin=vmin, vmax=vmax,
                              aspect='equal')

        # Draw damage outer contour using smoothed damage_mask
        if damage_mask_float is not None:
            axes[idx].contour(X.T * 1e2, Y.T * 1e2, damage_mask_float.T,
                              levels=[0.5], colors='red', linewidths=1.5,
                              linestyles='--', alpha=0.8)

        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('')

    n_damages = len(damage_info['regions']) if damage_info else 0
    fig.suptitle(f'Displacement Magnitude $|\\mathbf{{u}}| = \\sqrt{{u_x^2 + u_y^2}}$\n'
                 f'(Anisotropic FEM Fixed BC with {n_damages} Damage Region(s), Ricker f0={source_f0 / 1e3:.0f} kHz)',
                 fontsize=14, fontweight='bold')

    damage_text = f", Damage: {damage_info['damage_percentage']:.2f}% area" if damage_info else ""
    material_text = (f"C11={C11 / 1e10:.1f}, C22={C22 / 1e10:.1f}, C12={source_info.get('C12', 5e10) / 1e10:.1f}, "
                     f"C66={source_info.get('C66', 3.96e10) / 1e10:.2f} (x 10^10 Pa), "
                     f"rho={source_info.get('rho', 7100):.0f} kg/m^3 | Fixed BC{damage_text}")
    fig.text(0.5, 0.01, material_text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fig_path = os.path.join(output_dir, 'wave_propagation_magnitude_fixed_bc_damage.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nDisplacement magnitude figure saved: {fig_path}")
    plt.close()

    # ==================== Figure 2: Displacement components ====================
    fig2, axes2 = plt.subplots(2, 4, figsize=(16, 7))

    for col_idx, (t_idx, target_t) in enumerate(zip(time_indices, target_times)):
        field_ux = u_x[:, :, t_idx].T
        vmax_ux = max(abs(field_ux.min()), abs(field_ux.max()))
        if vmax_ux < 1e-15:
            vmax_ux = 1e-15

        order_ux = get_order_of_magnitude(vmax_ux)

        im_ux = axes2[0, col_idx].imshow(field_ux, origin='lower',
                                         extent=[x.min() * 1e2, x.max() * 1e2,
                                                 y.min() * 1e2, y.max() * 1e2],
                                         cmap='RdBu_r', vmin=-vmax_ux, vmax=vmax_ux,
                                         aspect='equal')

        # Draw damage outer contour using contour
        if damage_mask_float is not None:
            axes2[0, col_idx].contour(X.T * 1e2, Y.T * 1e2, damage_mask_float.T,
                                      levels=[0.5], colors='black', linewidths=1.5,
                                      linestyles='--', alpha=0.7)

        field_uy = u_y[:, :, t_idx].T
        vmax_uy = max(abs(field_uy.min()), abs(field_uy.max()))
        if vmax_uy < 1e-15:
            vmax_uy = 1e-15

        order_uy = get_order_of_magnitude(vmax_uy)

        im_uy = axes2[1, col_idx].imshow(field_uy, origin='lower',
                                         extent=[x.min() * 1e2, x.max() * 1e2,
                                                 y.min() * 1e2, y.max() * 1e2],
                                         cmap='RdBu_r', vmin=-vmax_uy, vmax=vmax_uy,
                                         aspect='equal')

        # Draw damage outer contour using contour
        if damage_mask_float is not None:
            axes2[1, col_idx].contour(X.T * 1e2, Y.T * 1e2, damage_mask_float.T,
                                      levels=[0.5], colors='black', linewidths=1.5,
                                      linestyles='--', alpha=0.7)

        for row, (im, vmax, order) in enumerate([(im_ux, vmax_ux, order_ux),
                                                 (im_uy, vmax_uy, order_uy)]):
            divider = make_axes_locatable(axes2[row, col_idx])
            cax = divider.append_axes("right", size="4%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            scale_factor = 10 ** (-order)
            cbar.set_ticks([-vmax, 0, vmax])
            cbar.ax.yaxis.set_major_formatter(
                FuncFormatter(lambda val, pos, sf=scale_factor: f'{val * sf:.1f}'.replace('-', '\u2212'))
            )
            cbar.ax.tick_params(labelsize=10)
            order_text = r'$\times 10^{' + str(order) + '}$'
            cax.text(0.5, 1.02, order_text, transform=cax.transAxes,
                     ha='center', va='bottom', fontsize=11)

        axes2[0, col_idx].set_title(f't = {t[t_idx] * 1e6:.1f} us', fontsize=12, fontweight='bold')

        for row in [0, 1]:
            axes2[row, col_idx].set_xticks([])
            axes2[row, col_idx].set_yticks([])

    axes2[0, 0].set_ylabel('$u_x$ (m)', fontsize=13, fontweight='bold')
    axes2[1, 0].set_ylabel('$u_y$ (m)', fontsize=13, fontweight='bold')

    fig2.suptitle(f'Displacement Components (FEM Fixed BC with {n_damages} Damage Region(s))',
                  fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fig2_path = os.path.join(output_dir, 'wave_propagation_components_fixed_bc_damage.png')
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"Displacement components figure saved: {fig2_path}")
    plt.close()


# ============================================================================
# Part 7: Main Program
# ============================================================================

def main():
    """
    Main function: set parameters and run FEM solver
    (Ricker wavelet + fixed boundary + damage regions)
    Checkerboard alternating diagonal mesh

    Includes damage region mesh visualization
    """
    # ============= Parameter settings =============
    nx = ny = 301
    Lx = Ly = 0.4  # 40 cm
    t_total = 120e-6  # 120 us
    nt = 4001

    # Ricker wavelet parameters
    source_amplitude = 1e10  # 10^10 N
    source_f0 = 100e3  # 100 kHz
    source_t0 = 10e-6  # 10 us

    gaussian_param = 5e6

    # Material parameters
    C11 = 130.74e9
    C22 = 11.50e9
    C12 = 10.49e9
    C16 = 30.67e9
    C26 = 3.75e9
    C66 = 14.77e9
    rho = 1610.0

    source_position = (0.0, 0.0)

    # ============= Damage configuration =============
    damage_configs = [
        {
            'type': 'circle',
            'center': (0.1, 0.1),  # 10cm, 10cm from center
            'size': 0.02,  # 2cm radius
            'reduction_factor': 0.5  # 50% stiffness reduction
        },
        {
            'type': 'square',
            'center': (-0.1, -0.1),  # -10cm, -10cm from center
            'size': 0.02,  # 4cm half-width (4cm x 4cm square)
            'reduction_factor': 0.5  # 50% stiffness reduction
        },
        {
            'type': 'square',
            'center': (-0.1, -0.06),  # -10cm, -6cm from center
            'size': 0.02,  # 4cm half-width (4cm x 4cm square)
            'reduction_factor': 0.5  # 50% stiffness reduction
        },
        {
            'type': 'square',
            'center': (-0.06, -0.1),  # -6cm, -10cm from center
            'size': 0.02,  # 4cm half-width (4cm x 4cm square)
            'reduction_factor': 0.5  # 50% stiffness reduction
        }
    ]

    output_dir = "./data_anisotropic_fem_ricker_fixed_bc_damage"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("2D Anisotropic Elastic Wave Equation - FEM + Damage Regions")
    print("Source: Ricker wavelet | BC: Fixed boundary (u=0)")
    print("Checkerboard alternating diagonal mesh")
    print("=" * 70)
    print("\nBoundary condition:")
    print("  Type: Fixed boundary (displacement = 0)")
    print("  Implementation: force boundary node u_x = u_y = 0 after each time step")
    print("=" * 70)
    print(f"\nMaterial parameters:")
    print(f"  rho = {rho} kg/m^3")
    print(f"  C11 = {C11 / 1e10:.2f} x 10^10 Pa")
    print(f"  C22 = {C22 / 1e10:.2f} x 10^10 Pa")
    print(f"  C12 = {C12 / 1e10:.2f} x 10^10 Pa")
    print(f"  C16 = {C16 / 1e10:.2f} x 10^10 Pa")
    print(f"  C26 = {C26 / 1e10:.2f} x 10^10 Pa")
    print(f"  C66 = {C66 / 1e10:.2f} x 10^10 Pa")
    print(f"\nRicker wavelet parameters:")
    print(f"  Dominant frequency f0 = {source_f0 / 1e3:.0f} kHz")
    print(f"  Time delay t0 = {source_t0 * 1e6:.0f} us")
    print(f"  Amplitude = {source_amplitude:.1e} N")
    print(f"\nDamage regions:")
    for i, dmg in enumerate(damage_configs):
        print(f"  Damage {i + 1}: {dmg['type']}, center=({dmg['center'][0] * 1e2:.1f}, {dmg['center'][1] * 1e2:.1f}) cm, "
              f"size={dmg['size'] * 1e2:.1f} cm, stiffness reduction={dmg['reduction_factor']:.0%}")
    print("=" * 70)

    # Solve
    x, y, t, u_x, u_y, total_displacement, source_mask, source_indices, nodes, elements, source_info = \
        solve_2d_anisotropic_wave_fem_ricker_fixed(
            nx=nx, ny=ny, Lx=Lx, Ly=Ly, nt=nt, t_total=t_total,
            source_amplitude=source_amplitude,
            source_f0=source_f0,
            source_t0=source_t0,
            gaussian_param=gaussian_param,
            C11=C11, C22=C22, C12=C12, C16=C16, C26=C26, C66=C66,
            rho=rho,
            source_position=source_position,
            damage_configs=damage_configs,
            device=device
        )

    # ==================== Damage region mesh visualization ====================
    print("\n" + "=" * 70)
    print("Visualizing damage region FEM mesh")
    print("=" * 70)

    # Visualize damage region mesh
    element_in_damage, element_centers = visualize_damage_mesh(
        x, y, elements,
        source_info['damage_mask'],
        source_info['damage_info'],
        output_dir,
        zoom_region=None,  # Automatically determine zoom region
        show_nodes=True  # Show nodes
    )

    # Visualize checkerboard alternating diagonal pattern
    visualize_mesh_pattern(x, y, elements, output_dir)
    # ==================================================================

    # Visualize wave field propagation
    print("\nStarting wave field visualization...")
    visualize_wave_propagation(
        x, y, t, u_x, u_y, total_displacement, source_mask, source_indices, nodes, elements,
        source_info=source_info,
        output_dir=output_dir
    )

    # Save data (time downsampling)
    stride = 5
    nt_ds = (len(t) + stride - 1) // stride

    mat_file_path = os.path.join(output_dir, f"anisotropic_ricker_damage_data_ds{stride}_FEM_fixed_bc.mat")

    print(f"\nSaving data to: {mat_file_path}")
    with h5py.File(mat_file_path, "w", libver='earliest') as f:
        dset_ux = f.create_dataset(
            "ux",
            shape=(nt_ds, nx, ny),
            dtype="float32",
            chunks=(1, nx, ny),
            compression="gzip",
            compression_opts=4
        )
        dset_uy = f.create_dataset(
            "uy",
            shape=(nt_ds, nx, ny),
            dtype="float32",
            chunks=(1, nx, ny),
            compression="gzip",
            compression_opts=4
        )

        out_k = 0
        for k in range(0, len(t), stride):
            dset_ux[out_k, :, :] = u_x[:, :, k]
            dset_uy[out_k, :, :] = u_y[:, :, k]
            out_k += 1
            if out_k % 10 == 0:
                print(f"  Save progress: {out_k}/{nt_ds} ({100 * out_k / nt_ds:.1f}%)")

        t_ds = t[::stride]

        f.create_dataset("x", data=x.astype("float32"))
        f.create_dataset("y", data=y.astype("float32"))
        f.create_dataset("t", data=t_ds.astype("float32"))

        # Save damage mask
        f.create_dataset("damage_mask", data=source_info['damage_mask'].astype("uint8"))

        # Save damage information
        g_damage = f.create_group("damage_info")
        g_damage.create_dataset("total_damaged_points",
                                data=source_info['damage_info']['total_damaged_points'])
        g_damage.create_dataset("damage_percentage",
                                data=source_info['damage_info']['damage_percentage'])

        # Save information for each damage region
        for i, region in enumerate(source_info['damage_info']['regions']):
            g_region = g_damage.create_group(f"region_{i}")
            g_region.create_dataset("type", data=region['type'])
            g_region.create_dataset("center", data=np.array(region['center']))
            g_region.create_dataset("size", data=region['size'])
            g_region.create_dataset("reduction_factor", data=region['reduction_factor'])
            g_region.create_dataset("n_points", data=region['n_points'])

        # Save source_info
        g = f.create_group("source_info")
        for key, val in source_info.items():
            if key in ['damage_mask', 'damage_info', 'damage_configs']:
                continue  # Already saved separately
            if np.isscalar(val):
                g.create_dataset(key, data=val)
            else:
                g.create_dataset(key, data=np.array(val))

    print(f"\nData saved: {mat_file_path}")
    print(f"  Downsampling stride = {stride}")
    print(f"  Saved shape: ux, uy = {(nt_ds, nx, ny)}")
    print(f"  Time length: {len(t_ds)}")
    print(f"  Damage mask shape: {source_info['damage_mask'].shape}")
    print(f"  Number of damage regions: {len(source_info['damage_info']['regions'])}")

    print(f"\nBoundary condition verification:")
    print(f"  - Max boundary displacement: {source_info['max_u_boundary']:.2e} m")
    print(f"  - Boundary condition: {source_info['boundary_condition']}")

    print(f"\nMaterial properties summary:")
    print(f"  - Anisotropic material")
    print(f"  - C11 = {source_info['C11'] / 1e10:.2f} x 10^10 Pa")
    print(f"  - C22 = {source_info['C22'] / 1e10:.2f} x 10^10 Pa")
    print(f"  - C12 = {source_info['C12'] / 1e10:.2f} x 10^10 Pa")
    print(f"  - C66 = {source_info['C66'] / 1e10:.2f} x 10^10 Pa")
    print(f"  - C16 = {source_info['C16'] / 1e10:.2f} x 10^10 Pa")
    print(f"  - C26 = {source_info['C26'] / 1e10:.2f} x 10^10 Pa")
    print(f"  - Density rho = {source_info['rho']:.0f} kg/m^3")
    print(f"  - Anisotropy factor ~ {source_info['anisotropy_factor']:.3f}")
    print(f"  - P-wave velocity ~ {source_info['vp_approx']:.0f} m/s")
    print(f"  - S-wave velocity ~ {source_info['vs_approx']:.0f} m/s")

    print(f"\nDamage region summary:")
    print(f"  - Number of damage regions: {len(source_info['damage_info']['regions'])}")
    print(f"  - Total damaged points: {source_info['damage_info']['total_damaged_points']}")
    print(f"  - Damage ratio: {source_info['damage_info']['damage_percentage']:.2f}%")
    for i, region in enumerate(source_info['damage_info']['regions']):
        print(f"  - Damage {i + 1}: {region['type']}, "
              f"center=({region['center'][0] * 1e2:.1f}, {region['center'][1] * 1e2:.1f}) cm, "
              f"size={region['size'] * 1e2:.1f} cm, stiffness reduction={region['reduction_factor']:.0%}")

    # Mesh visualization summary
    print(f"\nDamage region mesh statistics:")
    print(f"  - Total elements: {len(elements)}")
    print(f"  - Damaged elements: {np.sum(element_in_damage)}")
    print(f"  - Damaged element ratio: {100 * np.sum(element_in_damage) / len(elements):.2f}%")

    print(f"\nComputation completed! All results saved to: {output_dir}")
    print(f"\nGenerated mesh visualization images:")
    print(f"  - fem_mesh_damage_overview.png  : Full domain mesh")
    print(f"  - fem_mesh_damage_zoomed.png    : Damage region zoomed view")
    print(f"  - fem_mesh_damage_detail.png    : Damage boundary detail (with element numbers)")
    print(f"  - mesh_pattern_checkerboard.png : Checkerboard alternating diagonal pattern diagram")


if __name__ == "__main__":
    main()