"""
Anisotropic Elastic Wave Propagation - FEM with Structured Triangular Mesh

[Fixed Boundary Version] Uses checkerboard alternating diagonal pattern.
Boundary condition: Fixed (Dirichlet, u = 0 on boundary).

Key features:
1. Same node positions as FDM (structured grid)
2. Triangular element finite element solver
3. Anisotropic material
4. Ricker wavelet (Mexican hat wavelet) force source
5. GPU acceleration (PyTorch)
6. Fixed boundary conditions (u = 0 on boundary)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap, Normalize
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

# GPU setup
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================================
# Part 1: Utility Functions
# ============================================================================

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Truncate a colormap to a specified range."""
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def gaussian_point_load(x, y, x0, y0, a=150):
    """
    Gaussian function approximating a Dirac delta function.

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
    Ricker wavelet (Mexican hat wavelet).

    Parameters:
    -----------
    t : time (s)
    f0 : center frequency (Hz)
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
    """Calculate the full width at half maximum (FWHM) of the Gaussian function."""
    return 2 * np.sqrt(np.log(2) / a)


def calculate_wave_velocities_anisotropic(C11, C22, C12, C16, C26, C66, rho):
    """
    Calculate approximate wave velocities for anisotropic material.

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
# Part 2: Mesh Generation - Checkerboard Alternating Diagonal
# ============================================================================

def create_structured_triangular_mesh(nx, ny, Lx, Ly):
    """
    Create a structured triangular mesh (node positions identical to FDM).

    Uses a checkerboard alternating diagonal pattern to ensure left-right symmetry.

    For cells where (i+j) is even: diagonal from bottom-left to top-right (node0 -> node2)
    For cells where (i+j) is odd:  diagonal from bottom-right to top-left (node1 -> node3)

    Parameters:
    -----------
    nx, ny : number of grid nodes
    Lx, Ly : computational domain dimensions

    Returns:
    --------
    x, y : coordinate arrays
    elements : triangle element connectivity table
    boundary_nodes : boundary node indices
    dx, dy : grid spacing
    """
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    x = np.linspace(-Lx / 2, Lx / 2, nx)
    y = np.linspace(-Ly / 2, Ly / 2, ny)

    print(f"\n* Structured triangular mesh generation (checkerboard alternating diagonal pattern):")
    print(f"  Nodes: {nx} x {ny} = {nx * ny}")
    print(f"  Grid spacing: dx={dx * 1e3:.4f} mm, dy={dy * 1e3:.4f} mm")
    print(f"  Domain: [{x[0] * 1e2:.2f}, {x[-1] * 1e2:.2f}] x [{y[0] * 1e2:.2f}, {y[-1] * 1e2:.2f}] cm")

    # Build triangle element connectivity table
    num_elements = 2 * (nx - 1) * (ny - 1)
    elements = np.zeros((num_elements, 3), dtype=np.int64)

    elem_idx = 0
    for j in range(ny - 1):
        for i in range(nx - 1):
            node0 = j * nx + i          # bottom-left
            node1 = j * nx + (i + 1)    # bottom-right
            node2 = (j + 1) * nx + (i + 1)  # top-right
            node3 = (j + 1) * nx + i    # top-left

            # Checkerboard alternating diagonal pattern
            if (i + j) % 2 == 0:
                # Even cell: diagonal from bottom-left to top-right (node0 -> node2)
                elements[elem_idx] = [node0, node1, node2]      # lower triangle
                elements[elem_idx + 1] = [node0, node2, node3]  # upper triangle
            else:
                # Odd cell: diagonal from bottom-right to top-left (node1 -> node3)
                elements[elem_idx] = [node0, node1, node3]      # lower triangle
                elements[elem_idx + 1] = [node1, node2, node3]  # upper triangle

            elem_idx += 2

    print(f"  Triangle elements: {num_elements}")
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
# Part 3: FEM Solver (Anisotropic Material - Ricker Wavelet - Fixed Boundary)
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
        device=None
):
    """
    Solve 2D anisotropic elastic wave equation using FEM (Ricker wavelet + fixed boundary).

    Parameters:
    -----------
    nx, ny : number of grid nodes
    Lx, Ly : computational domain dimensions (m)
    nt : number of time steps
    t_total : total simulation time (s)
    source_amplitude : Ricker wavelet amplitude (N)
    source_f0 : Ricker wavelet center frequency (Hz)
    source_t0 : Ricker wavelet time delay (s)
    gaussian_param : Gaussian parameter (spatial distribution)
    C11, C22, C12, C16, C26, C66 : stiffness matrix coefficients (Pa)
    rho : density (kg/m^3)
    source_position : force source position
    device : computation device

    Returns:
    --------
    t : time array
    u_x_cpu, u_y_cpu : displacement fields
    additional info...
    """
    start_time = time.time()

    print("\n" + "=" * 70)
    print("2D Anisotropic Elastic Wave Equation - FEM Solver (GPU Accelerated)")
    print("Source: Ricker Wavelet | Boundary: Fixed (Dirichlet: u=0)")
    print("Checkerboard Alternating Diagonal Mesh")
    print("=" * 70)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nComputation device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(device.index if device.index else 0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")

    # Calculate wave velocities
    vp_approx, vs_approx, anisotropy_factor = calculate_wave_velocities_anisotropic(
        C11, C22, C12, C16, C26, C66, rho)

    print(f"\nMaterial parameters (anisotropic):")
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

    # Time discretization
    dt = t_total / (nt - 1)
    t = np.linspace(0, t_total, nt)

    # CFL condition check
    max_velocity = max(vp_approx, vs_approx)
    cfl_number = max_velocity * dt / min(dx, dy)

    if cfl_number > 0.5:
        dt_stable = 0.5 * min(dx, dy) / max_velocity
        print(f"\n  Warning: CFL number = {cfl_number:.4f} > 0.5")
        print(f"  Suggested time step: dt <= {dt_stable:.6e}")

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
    X, Y = np.meshgrid(x, y, indexing='ij')
    gaussian_distribution = gaussian_point_load(X, Y, x0_src, y0_src, gaussian_param)

    fwhm = calculate_fwhm(gaussian_param)
    print(f"\nForce source parameters (Ricker wavelet):")
    print(f"  Position: ({x0_src * 1e2:.2f}, {y0_src * 1e2:.2f}) cm")
    print(f"  Center frequency: f0 = {source_f0 / 1e3:.1f} kHz")
    print(f"  Time delay: t0 = {source_t0 * 1e6:.1f} us")
    print(f"  Amplitude: {source_amplitude:.2e} N")
    print(f"  Spatial distribution FWHM: {fwhm * 1e3:.2f} mm")

    # Visualize Gaussian distribution and Ricker wavelet
    output_dir = "./data_anisotropic_fem_ricker_fixed_bc"
    os.makedirs(output_dir, exist_ok=True)
    plot_gaussian_distribution_func(x, y, gaussian_distribution, x0_src, y0_src, fwhm, output_dir)
    plot_ricker_wavelet(t, source_f0, source_t0, source_amplitude, output_dir)

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

    # Precompute Jacobian and B matrices for triangle elements
    print("\nStep 3: Precomputing element information...")

    # Gauss integration points (3-point)
    gauss_points = np.array([
        [1.0 / 6.0, 1.0 / 6.0],
        [2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0]
    ])
    gauss_weights = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])

    def triangle_shape_functions(xi, eta):
        """Shape function derivatives for triangle elements."""
        dN_dxi = np.array([-1.0, 1.0, 0.0])
        dN_deta = np.array([-1.0, 0.0, 1.0])
        return dN_dxi, dN_deta

    # Precompute Jacobian for each element
    detJ_all = np.zeros(num_elements)
    J_inv_all = np.zeros((num_elements, 2, 2))

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
            print(f"  Warning: Element {elem_idx} has detJ = {detJ:.6e} < 0")

        J_inv_all[elem_idx] = np.linalg.inv(J)

    detJ_gpu = torch.from_numpy(np.abs(detJ_all)).float().to(device)
    J_inv_gpu = torch.from_numpy(J_inv_all).float().to(device)

    print(f"  detJ range: [{np.abs(detJ_all).min():.6e}, {np.abs(detJ_all).max():.6e}]")

    # Precompute B matrices
    B_matrices_gp = []
    for gp_idx in range(3):
        xi, eta = gauss_points[gp_idx]
        dN_dxi, dN_deta = triangle_shape_functions(xi, eta)

        dN_dxi_gpu = torch.from_numpy(dN_dxi).float().to(device)
        dN_deta_gpu = torch.from_numpy(dN_deta).float().to(device)

        B_matrices_gp.append((dN_dxi_gpu, dN_deta_gpu))

    gauss_weights_gpu = torch.from_numpy(gauss_weights).float().to(device)

    # Material matrix (anisotropic)
    C_gpu = torch.tensor([
        [C11, C12, C16],
        [C12, C22, C26],
        [C16, C26, C66]
    ], dtype=torch.float32, device=device)

    print(f"\nMaterial matrix C (x10^10 Pa):")
    print(f"  [{C11 / 1e10:.2f}  {C12 / 1e10:.2f}  {C16 / 1e10:.2f}]")
    print(f"  [{C12 / 1e10:.2f}  {C22 / 1e10:.2f}  {C26 / 1e10:.2f}]")
    print(f"  [{C16 / 1e10:.2f}  {C26 / 1e10:.2f}  {C66 / 1e10:.2f}]")

    # Internal force operator (anisotropic)
    def apply_stiffness_operator(u_x_flat, u_y_flat):
        """Compute internal forces (anisotropic material)."""
        f_x = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        f_y = torch.zeros(num_nodes, dtype=torch.float32, device=device)

        for gp_idx in range(3):
            dN_dxi, dN_deta = B_matrices_gp[gp_idx]
            weight = gauss_weights_gpu[gp_idx]

            u_elem_x = u_x_flat[elements_gpu]
            u_elem_y = u_y_flat[elements_gpu]

            # Shape function derivatives in physical space
            dN_dx = J_inv_gpu[:, 0, 0].unsqueeze(1) * dN_dxi.unsqueeze(0) + \
                    J_inv_gpu[:, 1, 0].unsqueeze(1) * dN_deta.unsqueeze(0)
            dN_dy = J_inv_gpu[:, 0, 1].unsqueeze(1) * dN_dxi.unsqueeze(0) + \
                    J_inv_gpu[:, 1, 1].unsqueeze(1) * dN_deta.unsqueeze(0)

            # Strain components
            eps_xx = torch.sum(dN_dx * u_elem_x, dim=1)
            eps_yy = torch.sum(dN_dy * u_elem_y, dim=1)
            gamma_xy = torch.sum(dN_dy * u_elem_x + dN_dx * u_elem_y, dim=1)

            # Anisotropic stress-strain relationship
            sigma_xx = C_gpu[0, 0] * eps_xx + C_gpu[0, 1] * eps_yy + C_gpu[0, 2] * gamma_xy
            sigma_yy = C_gpu[1, 0] * eps_xx + C_gpu[1, 1] * eps_yy + C_gpu[1, 2] * gamma_xy
            tau_xy = C_gpu[2, 0] * eps_xx + C_gpu[2, 1] * eps_yy + C_gpu[2, 2] * gamma_xy

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

    # Time factors (precompute Ricker wavelet)
    print("\nStep 4: Precomputing Ricker wavelet time factors...")
    time_factors = torch.zeros(nt, dtype=torch.float32, device=device)
    for n in range(nt):
        time_factors[n] = ricker_wavelet(t[n], source_f0, source_t0, source_amplitude)

    # Force source normalization
    print("\nStep 5: Force source normalization...")
    rho_inv = 1.0 / rho
    print(f"  Inverse density: 1/rho = {rho_inv:.6e}")
    print(f"  Force normalization strategy: F = ricker(t) x gaussian x M_lumped / rho")

    # Time integration (fixed boundary conditions)
    print(f"\nStep 6: Time integration (central difference - fixed boundary)...")
    print(f"  Fixed boundary condition: boundary node displacements forced to 0")
    print(f"  Number of boundary nodes: {len(boundary_nodes)}")

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

        # ========== Fixed boundary condition: force boundary node displacements to 0 ==========
        u_x_flat[boundary_nodes_gpu, n + 1] = 0
        u_y_flat[boundary_nodes_gpu, n + 1] = 0
        # ======================================================================================

        # Progress display
        if n % max(1, nt // 10) == 0:
            elapsed = time.time() - time_integration_start
            eta = elapsed / n * (nt - n)
            max_ux = torch.abs(u_x_flat[:, n + 1]).max().item()
            max_uy = torch.abs(u_y_flat[:, n + 1]).max().item()

            if torch.isnan(u_x_flat[:, n + 1]).any():
                print(f"\n  Error: NaN detected at time step {n}!")
                break

            print(f"  Progress: {n}/{nt} ({100 * n / nt:.1f}%) | "
                  f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s | "
                  f"max|u_x|: {max_ux:.2e} | max|u_y|: {max_uy:.2e}")

    time_integration_time = time.time() - time_integration_start

    # Reshape to (nx, ny, nt) format
    print("\nStep 7: Reshaping data format...")
    u_x_temp = u_x_flat.reshape(ny, nx, nt)
    u_y_temp = u_y_flat.reshape(ny, nx, nt)

    u_x = torch.transpose(u_x_temp, 0, 1)
    u_y = torch.transpose(u_y_temp, 0, 1)

    u_x_cpu = 0.7*u_x.cpu().numpy()
    u_y_cpu = 0.7*u_y.cpu().numpy()

    # Compute total displacement
    total_displacement = np.sqrt(u_x_cpu ** 2 + u_y_cpu ** 2)

    # ========== Verify boundary conditions ==========
    print("\nStep 8: Verifying fixed boundary conditions...")
    t_check = -1
    u_x_boundary = u_x_cpu[boundary_nodes % nx, boundary_nodes // nx, t_check]
    u_y_boundary = u_y_cpu[boundary_nodes % nx, boundary_nodes // nx, t_check]

    max_u_boundary = max(np.abs(u_x_boundary).max(), np.abs(u_y_boundary).max())
    print(f"  Maximum boundary node displacement: {max_u_boundary:.2e} m")

    if max_u_boundary < 1e-15:
        print("  * Fixed boundary condition strictly satisfied (u = 0 on boundary)")
    else:
        print(f"  Warning: Small boundary condition error: {max_u_boundary:.2e}")

    total_time = time.time() - start_time

    print(f"\n" + "=" * 70)
    print(f"* FEM computation completed!")
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
        'max_u_boundary': max_u_boundary
    }

    return x, y, t, u_x_cpu, u_y_cpu, total_displacement, source_mask, source_indices, nodes, elements, source_info


# ============================================================================
# Part 4: Visualization Functions
# ============================================================================

def plot_gaussian_distribution_func(x, y, gaussian_distribution, x0_src, y0_src, fwhm, output_dir):
    """Plot the Gaussian spatial distribution."""
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 2D distribution
    im1 = ax1.imshow(gaussian_distribution.T, origin='lower',
                     extent=[x.min() * 1e2, x.max() * 1e2, y.min() * 1e2, y.max() * 1e2],
                     cmap='hot')
    ax1.set_title('Gaussian Function Spatial Distribution\n(Anisotropic Material - FEM Fixed BC)', fontsize=14)
    ax1.set_xlabel('x (cm)', fontsize=12)
    ax1.set_ylabel('y (cm)', fontsize=12)
    ax1.plot(x0_src * 1e2, y0_src * 1e2, 'b*', markersize=10, label='Load Center')
    plt.colorbar(im1, ax=ax1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Cross-section along x
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

    output_path = os.path.join(output_dir, 'gaussian_distribution_anisotropic_fem_fixed_bc.png')
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"* Gaussian distribution plot saved: {output_path}")


def plot_ricker_wavelet(t, f0, t0, amplitude, output_dir):
    """Plot the Ricker wavelet."""
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

    output_path = os.path.join(output_dir, 'ricker_wavelet_fixed_bc.png')
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"* Ricker wavelet plot saved: {output_path}")


def visualize_wave_propagation(x, y, t, u_x, u_y, total_displacement, source_mask, source_indices,
                               nodes, elements, source_info=None, output_dir="./data_anisotropic_fem_ricker_fixed_bc"):
    """
    Visualize wave field propagation (FEM version - fixed boundary) - displays displacement magnitude.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    C11 = source_info.get('C11', 16.5e10)
    C22 = source_info.get('C22', 6.2e10)
    vp_approx = source_info.get('vp_approx', 5000)
    vs_approx = source_info.get('vs_approx', 2400)
    source_f0 = source_info.get('source_f0', 170e3)

    t_max = t[-1]
    target_times = [t_max * 0.25, t_max * 0.5, t_max * 0.75, t_max * 1.0]

    time_indices = []
    for target_t in target_times:
        idx = np.argmin(np.abs(t - target_t))
        time_indices.append(idx)
        print(f"Target time {target_t * 1e6:.2f} us -> Actual time {t[idx] * 1e6:.2f} us (index {idx})")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
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

        order = get_order_of_magnitude(vmax)
        scale_factor = 10 ** (-order)

        im = axes[idx].imshow(field, origin='lower',
                              extent=[x.min() * 1e2, x.max() * 1e2,
                                      y.min() * 1e2, y.max() * 1e2],
                              cmap='gray_r', vmin=vmin, vmax=vmax,
                              aspect='equal')

        divider = make_axes_locatable(axes[idx])
        cax = divider.append_axes("right", size="4%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)

        cbar.set_ticks([vmin, vmax])
        cbar.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda val, pos, sf=scale_factor: f'{val * sf:.1f}')
        )
        cbar.ax.tick_params(labelsize=11, pad=1)

        order_text = r'$\times 10^{' + str(order) + '}$ m'
        cax.text(0.5, 1.02, order_text, transform=cax.transAxes,
                 ha='center', va='bottom', fontsize=12)

        axes[idx].set_title(f't = {t[t_idx] * 1e6:.1f} us', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('x (cm)', fontsize=12)
        axes[idx].set_ylabel('y (cm)', fontsize=12)

    fig.suptitle(f'Displacement Magnitude $|\\mathbf{{u}}| = \\sqrt{{u_x^2 + u_y^2}}$\n'
                 f'(Anisotropic FEM Fixed BC, Ricker f0={source_f0 / 1e3:.0f} kHz)',
                 fontsize=14, fontweight='bold')

    material_text = (f"C11={C11 / 1e10:.1f}, C22={C22 / 1e10:.1f}, C12={source_info.get('C12', 5e10) / 1e10:.1f}, "
                     f"C66={source_info.get('C66', 3.96e10) / 1e10:.2f} (x10^10 Pa), "
                     f"rho={source_info.get('rho', 7100):.0f} kg/m^3 | Fixed BC")
    fig.text(0.5, 0.01, material_text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.8))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig_path = os.path.join(output_dir, 'wave_propagation_magnitude_fixed_bc.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n* Displacement magnitude plot saved: {fig_path}")

    plt.show()
    plt.close()

    # Component plots
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

    fig2.suptitle('Displacement Components (FEM Fixed BC)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fig2_path = os.path.join(output_dir, 'wave_propagation_components_fixed_bc.png')
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"* Displacement component plot saved: {fig2_path}")

    plt.show()
    plt.close()


# ============================================================================
# Part 5: Main Program
# ============================================================================

def main():
    """
    Main function: Set parameters and run FEM solver (Ricker wavelet + fixed boundary).
    Checkerboard alternating diagonal mesh.
    """
    # ============= Parameter Setup =============
    nx = ny = 301
    Lx = Ly = 0.4  # 40 cm
    t_total = 60e-6  # 60 us
    nt = 2001

    # Ricker wavelet parameters
    source_amplitude = 1e10  # 10^10 N
    source_f0 = 100e3        # 100 kHz
    source_t0 = 10e-6        # 10 us

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

    output_dir = "./data_anisotropic_fem_ricker_fixed_bc"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("2D Anisotropic Elastic Wave Equation - FEM")
    print("Source: Ricker Wavelet | Boundary: Fixed (u=0)")
    print("Checkerboard Alternating Diagonal Mesh")
    print("=" * 70)
    print("\nBoundary condition change:")
    print("  Previous version: Free boundary (zero stress)")
    print("  Current version:  Fixed boundary (zero displacement)")
    print("  Implementation: Force boundary node u_x = u_y = 0 after each time step")
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
    print(f"  Center frequency f0 = {source_f0 / 1e3:.0f} kHz")
    print(f"  Time delay t0 = {source_t0 * 1e6:.0f} us")
    print(f"  Amplitude = {source_amplitude:.1e} N")
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
            device=device
        )

    # Visualization
    print("\nStarting visualization...")
    visualize_wave_propagation(
        x, y, t, u_x, u_y, total_displacement, source_mask, source_indices, nodes, elements,
        source_info=source_info,
        output_dir=output_dir
    )

    # Save data (time downsampled)
    stride = 5
    nt_ds = (len(t) + stride - 1) // stride

    mat_file_path = os.path.join(output_dir, f"anisotropic_ricker_data_ds{stride}_FEM_fixed_bc.mat")

    print(f"\nSaving data to: {mat_file_path}")
    with h5py.File(mat_file_path, "w") as f:
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

        g = f.create_group("source_info")
        for key, val in source_info.items():
            if np.isscalar(val):
                g.create_dataset(key, data=val)
            else:
                g.create_dataset(key, data=np.array(val))

    print(f"\n* Data saved: {mat_file_path}")
    print(f"  Downsampling stride = {stride}")
    print(f"  Saved shape: ux, uy = {(nt_ds, nx, ny)}")

    print(f"\nBoundary condition verification:")
    print(f"  - Maximum boundary displacement: {source_info['max_u_boundary']:.2e} m")
    print(f"  - Boundary condition: {source_info['boundary_condition']}")

    print(f"\n* Computation completed! All results saved to: {output_dir}")


if __name__ == "__main__":
    main()