"""
Anisotropic Elastic Parameter Inversion - Spatially Varying Parameters (FEM Version)
Spatially varying elastic parameter inversion for damage identification
Matching FEM data generation code (fixed boundary version), using Adam optimizer

Key features:
1. Independent 6 elastic parameters (C11, C22, C12, C16, C26, C66) at each grid point
2. Total parameter count: 6 x nx x ny (e.g. 6 x 301 x 301 = 543,606 parameters)
3. No spatial downsampling, full observation data retained
4. Regularization to avoid overfitting
5. Ricker wavelet as source signal (matching FEM data generation)
6. Fixed boundary condition (Dirichlet: u=0 on boundary)
7. Adam optimizer
"""

import torch

# torch.cuda.empty_cache()
# torch.cuda.ipc_collect()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import h5py
import os
import re
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Only expose physical GPU 0
import torch

print("is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("name0:", torch.cuda.get_device_name(0))
# Specify GPU to use
device = torch.device("cuda:0")
print(f"Using device: {device}")

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


def gaussian_point_load(X, Y, x0, y0, a=150):
    """
    Gaussian function approximating Dirac delta function
    """
    r_squared = (X - x0) ** 2 + (Y - y0) ** 2
    return (a / np.pi) * torch.exp(-a * r_squared)


def ricker_wavelet(t, f0, t0, amplitude=1.0):
    """
    Ricker wavelet (Mexican hat wavelet)

    Parameters:
    -----------
    t : time (s) - can be numpy array or torch tensor
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
    if isinstance(t, torch.Tensor):
        return amplitude * (1 - 2 * pi_f0_tau_sq) * torch.exp(-pi_f0_tau_sq)
    else:
        return amplitude * (1 - 2 * pi_f0_tau_sq) * np.exp(-pi_f0_tau_sq)


def conv2d_fixed(field, kernel):
    """
    Convolution with zero padding (fixed boundary condition)

    Args:
        field: input field (batch, 1, ny, nx)
        kernel: convolution kernel (1, 1, 3, 3)

    Returns:
        Convolution result (batch, 1, ny, nx)
    """
    # Zero padding instead of reflection padding (fixed BC: u=0)
    field_pad = F.pad(field, (1, 1, 1, 1), mode='constant', value=0.0)
    return F.conv2d(field_pad, kernel, padding=0)


def _general_anisotropic_elastic_operator_with_force_spatially_varying(
        ux, uy, C11, C22, C12, C16, C26, C66, h, force_x, force_y, rho):
    """
    General anisotropic elastic operator with spatially varying stiffness coefficients
    Uses zero padding for fixed boundary condition (u=0 on boundary)

    Args:
        ux, uy: displacement fields (batch, nx, ny)
        C11, C22, C12, C16, C26, C66: stiffness fields (batch, nx, ny)
        h: spatial step
        force_x, force_y: force fields (batch, nx, ny)
        rho: density (scalar or field)
    """
    h_inv_sq = h ** (-2)

    # Define differential operators
    d2dx2 = h_inv_sq * torch.tensor([[[[0.0, 0.0, 0.0],
                                       [1.0, -2.0, 1.0],
                                       [0.0, 0.0, 0.0]]]], device=ux.device, dtype=ux.dtype)

    d2dy2 = h_inv_sq * torch.tensor([[[[0.0, 1.0, 0.0],
                                       [0.0, -2.0, 0.0],
                                       [0.0, 1.0, 0.0]]]], device=ux.device, dtype=ux.dtype)

    d2dxdy = h_inv_sq * torch.tensor([[[[0.25, 0.0, -0.25],
                                        [0.0, 0.0, 0.0],
                                        [-0.25, 0.0, 0.25]]]], device=ux.device, dtype=ux.dtype)

    ux = ux.transpose(1, 2)
    uy = uy.transpose(1, 2)

    # Compute derivatives using zero padding (fixed boundary condition: u=0)
    ux_unsqueezed = ux.unsqueeze(1)
    uy_unsqueezed = uy.unsqueeze(1)

    d2ux_dx2 = conv2d_fixed(ux_unsqueezed, d2dx2).squeeze(1)
    d2ux_dy2 = conv2d_fixed(ux_unsqueezed, d2dy2).squeeze(1)
    d2uy_dx2 = conv2d_fixed(uy_unsqueezed, d2dx2).squeeze(1)
    d2uy_dy2 = conv2d_fixed(uy_unsqueezed, d2dy2).squeeze(1)
    d2ux_dxdy = conv2d_fixed(ux_unsqueezed, d2dxdy).squeeze(1)
    d2uy_dxdy = conv2d_fixed(uy_unsqueezed, d2dxdy).squeeze(1)

    # Transpose stiffness fields to match derivative shape
    C11_t = C11.transpose(1, 2)
    C22_t = C22.transpose(1, 2)
    C12_t = C12.transpose(1, 2)
    C16_t = C16.transpose(1, 2)
    C26_t = C26.transpose(1, 2)
    C66_t = C66.transpose(1, 2)

    # General anisotropic elastic operator with spatially varying coefficients
    Lux = (C11_t * d2ux_dx2 +
           C66_t * d2ux_dy2 +
           (C12_t + C66_t) * d2uy_dxdy +
           2 * C16_t * d2ux_dxdy +
           C16_t * d2uy_dx2 +
           C26_t * d2uy_dy2)

    Luy = (C66_t * d2uy_dx2 +
           C22_t * d2uy_dy2 +
           (C12_t + C66_t) * d2ux_dxdy +
           C16_t * d2ux_dx2 +
           C26_t * d2ux_dy2 +
           2 * C26_t * d2uy_dxdy)

    Lux = Lux.transpose(1, 2)
    Luy = Luy.transpose(1, 2)

    # Transpose force fields
    force_x_t = force_x.transpose(1, 2)
    force_y_t = force_y.transpose(1, 2)

    # Add force terms divided by density
    Lux = (Lux + force_x_t) / rho
    Luy = (Luy + force_y_t) / rho

    return Lux, Luy


def total_variation_2d(field):
    """
    Compute 2D Total Variation for spatial regularization

    Args:
        field: (batch, nx, ny) tensor

    Returns:
        TV value (scalar)
    """
    # Compute spatial gradients
    dx = field[:, 1:, :] - field[:, :-1, :]
    dy = field[:, :, 1:] - field[:, :, :-1]

    # TV = sum of absolute gradients
    tv = torch.sum(torch.abs(dx)) + torch.sum(torch.abs(dy))
    return tv


class SpatiallyVaryingAnisotropicSimulator(nn.Module):
    """
    Anisotropic elastic wave simulator with spatially varying stiffness parameters
    Using Ricker wavelet source (matching FEM data generation)
    Fixed boundary condition (u=0 on boundary)

    Parameters to invert: C11, C22, C12, C16, C26, C66 at each grid point
    Total parameters: 6 × nx × ny
    """

    def __init__(self, domain_shape, h, dt, nt, primary_time_stride,
                 source_component='y',
                 source_f0=100e3,  # Ricker dominant frequency (Hz)
                 source_t0=10e-6,  # Ricker time delay (s)
                 source_amplitude=1e10,
                 total_time=60e-6,
                 # Initial material parameters (matching fixed BC FEM code)
                 C11_init=130.74e9,
                 C22_init=11.50e9,
                 C12_init=10.49e9,
                 C16_init=30.67e9,
                 C26_init=3.75e9,
                 C66_init=14.77e9,
                 rho_known=1610.0,
                 gaussian_param=5e6,
                 use_log_params=True):
        super().__init__()

        self.domain_shape = domain_shape
        self.h = h
        self.dt = dt
        self.nt = nt
        self.total_time = total_time
        self.primary_time_stride = primary_time_stride
        self.use_log_params = use_log_params

        # Source parameters (Ricker wavelet)
        self.source_component = source_component
        self.source_f0 = source_f0  # Dominant frequency
        self.source_t0 = source_t0  # Time delay
        self.source_amplitude = source_amplitude
        self.gaussian_param = gaussian_param

        nx, ny = domain_shape

        # Spatially varying stiffness parameters (nx, ny)
        # Using log parameterization for better optimization
        if use_log_params:
            self.log_C11 = nn.Parameter(torch.full((1, nx, ny), np.log(C11_init), dtype=torch.float32))
            self.log_C22 = nn.Parameter(torch.full((1, nx, ny), np.log(C22_init), dtype=torch.float32))
            self.log_C12 = nn.Parameter(torch.full((1, nx, ny), np.log(C12_init), dtype=torch.float32))
            self.log_C16 = nn.Parameter(torch.full((1, nx, ny), np.log(abs(C16_init) + 1e6), dtype=torch.float32))
            self.log_C26 = nn.Parameter(torch.full((1, nx, ny), np.log(abs(C26_init) + 1e6), dtype=torch.float32))
            self.log_C66 = nn.Parameter(torch.full((1, nx, ny), np.log(C66_init), dtype=torch.float32))
        else:
            self.C11 = nn.Parameter(torch.full((1, nx, ny), C11_init, dtype=torch.float32))
            self.C22 = nn.Parameter(torch.full((1, nx, ny), C22_init, dtype=torch.float32))
            self.C12 = nn.Parameter(torch.full((1, nx, ny), C12_init, dtype=torch.float32))
            self.C16 = nn.Parameter(torch.full((1, nx, ny), C16_init, dtype=torch.float32))
            self.C26 = nn.Parameter(torch.full((1, nx, ny), C26_init, dtype=torch.float32))
            self.C66 = nn.Parameter(torch.full((1, nx, ny), C66_init, dtype=torch.float32))

        # Known density (fixed)
        self.register_buffer('rho', torch.tensor(rho_known, dtype=torch.float32))
        self.rho_known = rho_known

        # Parameter bounds (in Pa) - adjusted for material
        self.C11_bounds = (50e9, 250e9)
        self.C22_bounds = (5e9, 50e9)
        self.C12_bounds = (5e9, 50e9)
        self.C16_bounds = (0, 60e9)
        self.C26_bounds = (0, 20e9)
        self.C66_bounds = (5e9, 30e9)

        # Create coordinate grids for Gaussian calculation
        Lx = h * (nx - 1)
        Ly = h * (ny - 1)
        x = torch.linspace(-Lx / 2, Lx / 2, nx, device=device)
        y = torch.linspace(-Ly / 2, Ly / 2, ny, device=device)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        # Pre-compute Gaussian distribution
        x0_src = 0.0
        y0_src = 0.0
        self.gaussian_distribution = gaussian_point_load(X, Y, x0_src, y0_src, gaussian_param)
        self.register_buffer('gaussian_dist', self.gaussian_distribution)

        # Generate source signal (Ricker wavelet)
        self._generate_ricker_source_signal()

        total_params = 6 * nx * ny
        print(f"\nSpatially Varying Anisotropic Simulator initialized (Fixed BC):")
        print(f"  Domain: {nx}×{ny} = {nx * ny:,} points")
        print(f"  Total parameters to invert: 6 × {nx} × {ny} = {total_params:,}")
        print(f"  Time steps (FULL): {nt}")
        print(f"  PRIMARY time stride: {primary_time_stride}")
        print(f"  Known density: ρ = {rho_known} kg/m³")
        print(f"  Source: Ricker wavelet, f0={source_f0 / 1e3:.1f} kHz, t0={source_t0 * 1e6:.1f} μs")
        print(f"  Boundary condition: Fixed (u=0 on boundary)")

    def _generate_ricker_source_signal(self):
        """Generate Ricker wavelet source signal"""
        t = np.arange(self.nt) * self.dt
        signal = ricker_wavelet(t, self.source_f0, self.source_t0, self.source_amplitude)
        self.register_buffer('source_signal', torch.tensor(signal, dtype=torch.float32).unsqueeze(0))

    def get_material_params(self):
        """Get current stiffness parameters with constraints"""
        if self.use_log_params:
            C11 = torch.clamp(torch.exp(self.log_C11), *self.C11_bounds)
            C22 = torch.clamp(torch.exp(self.log_C22), *self.C22_bounds)
            C12 = torch.clamp(torch.exp(self.log_C12), *self.C12_bounds)
            # For C16 and C26, allow positive values (based on material)
            C16 = torch.clamp(torch.exp(self.log_C16), *self.C16_bounds)
            C26 = torch.clamp(torch.exp(self.log_C26), *self.C26_bounds)
            C66 = torch.clamp(torch.exp(self.log_C66), *self.C66_bounds)
        else:
            C11 = torch.clamp(self.C11, *self.C11_bounds)
            C22 = torch.clamp(self.C22, *self.C22_bounds)
            C12 = torch.clamp(self.C12, *self.C12_bounds)
            C16 = torch.clamp(self.C16, *self.C16_bounds)
            C26 = torch.clamp(self.C26, *self.C26_bounds)
            C66 = torch.clamp(self.C66, *self.C66_bounds)

        return C11, C22, C12, C16, C26, C66, self.rho

    def get_mean_params(self):
        """Get mean parameter values over the domain (in GPa for display)"""
        C11, C22, C12, C16, C26, C66, rho = self.get_material_params()
        return (C11.mean().item() / 1e9, C22.mean().item() / 1e9, C12.mean().item() / 1e9,
                C16.mean().item() / 1e9, C26.mean().item() / 1e9, C66.mean().item() / 1e9,
                rho.item())

    def forward(self, save_all_steps=False):
        """
        Execute forward simulation with fixed boundary condition

        Returns:
            ux_fields, uy_fields: (1, nt_output, nx, ny)
        """
        device = self.source_signal.device
        batch_size = 1

        # Initialize fields
        hidden_shape = (batch_size,) + self.domain_shape
        ux1 = torch.zeros(hidden_shape, device=device)
        uy1 = torch.zeros(hidden_shape, device=device)
        ux2 = torch.zeros(hidden_shape, device=device)
        uy2 = torch.zeros(hidden_shape, device=device)

        # Get spatially varying material parameters
        C11, C22, C12, C16, C26, C66, rho = self.get_material_params()

        ux_history = []
        uy_history = []

        # Time stepping loop
        for i in range(self.nt):
            # Create force source terms (Ricker wavelet)
            force_amplitude = self.source_signal[0, i]

            if self.source_component == 'x':
                force_x = force_amplitude * self.gaussian_dist.unsqueeze(0)
                force_y = torch.zeros_like(ux1)
            else:
                force_x = torch.zeros_like(ux1)
                force_y = force_amplitude * self.gaussian_dist.unsqueeze(0)

            # Compute elastic operator with spatially varying stiffness
            Lux1, Luy1 = _general_anisotropic_elastic_operator_with_force_spatially_varying(
                ux1, uy1, C11, C22, C12, C16, C26, C66, self.h,
                force_x, force_y, rho
            )

            # Time stepping
            ux_new = 2 * ux1 - ux2 + self.dt ** 2 * Lux1
            uy_new = 2 * uy1 - uy2 + self.dt ** 2 * Luy1

            # Update states
            ux2, uy2 = ux1, uy1
            ux1, uy1 = ux_new, uy_new

            # Save history
            if save_all_steps:
                ux_history.append(ux1)
                uy_history.append(uy1)
            else:
                if i % self.primary_time_stride == 0:
                    ux_history.append(ux1)
                    uy_history.append(uy1)

        # Stack time history
        ux_fields = torch.stack(ux_history, dim=1)
        uy_fields = torch.stack(uy_history, dim=1)

        return ux_fields, uy_fields


class SpatiallyVaryingParameterInverter:
    """
    Spatially varying parameter inverter for damage identification
    Using Adam optimizer with fixed boundary condition
    """

    def __init__(self, simulator, target_ux_primary, target_uy_primary,
                 secondary_time_stride=1,
                 lambda_tv=0.0):
        """
        Initialize inverter

        Args:
            simulator: SpatiallyVaryingAnisotropicSimulator instance
            target_ux_primary: Target displacement data (1, nt_primary, nx, ny)
            target_uy_primary: Target displacement data
            secondary_time_stride: Time downsampling for loss computation
            lambda_tv: Total Variation regularization weight
        """
        self.simulator = simulator
        self.target_ux_primary = target_ux_primary
        self.target_uy_primary = target_uy_primary
        self.secondary_time_stride = secondary_time_stride
        self.lambda_tv = lambda_tv

        # Apply secondary downsampling in time only (NO spatial downsampling)
        self.target_ux_secondary = target_ux_primary[:, ::secondary_time_stride, :, :]
        self.target_uy_secondary = target_uy_primary[:, ::secondary_time_stride, :, :]

        # History tracking
        self.history = {
            'loss': [],
            'loss_data': [],
            'loss_tv': [],
            'C11_mean': [], 'C22_mean': [], 'C12_mean': [],
            'C16_mean': [], 'C26_mean': [], 'C66_mean': []
        }

        print(f"\nSpatially Varying Inverter initialized (Adam, Fixed BC):")
        print(f"  Target data shape: {target_ux_primary.shape}")
        print(f"  Loss computation shape: {self.target_ux_secondary.shape}")
        print(f"  NO spatial downsampling - using full grid")
        print(f"  Regularization: TV={lambda_tv}")
        print(f"  Boundary condition: Fixed (u=0 on boundary)")

    def compute_loss(self, pred_ux_primary, pred_uy_primary):
        """Compute loss with regularization"""
        # Apply secondary time downsampling
        pred_ux_secondary = pred_ux_primary[:, ::self.secondary_time_stride, :, :]
        pred_uy_secondary = pred_uy_primary[:, ::self.secondary_time_stride, :, :]

        # Data fidelity loss
        loss_data_ux = torch.mean((pred_ux_secondary - self.target_ux_secondary) ** 2)
        loss_data_uy = torch.mean((pred_uy_secondary - self.target_uy_secondary) ** 2)
        loss_data = 5*loss_data_ux + loss_data_uy

        # Total Variation regularization (spatial smoothness)
        loss_tv = 0.0
        if self.lambda_tv > 0:
            C11, C22, C12, C16, C26, C66, rho = self.simulator.get_material_params()
            loss_tv = (total_variation_2d(C11) + total_variation_2d(C22) +
                       total_variation_2d(C12) + total_variation_2d(C16) +
                       total_variation_2d(C26) + total_variation_2d(C66))
            loss_tv = self.lambda_tv * loss_tv

        total_loss = loss_data + loss_tv

        return total_loss, loss_data, loss_tv

    def invert(self, max_epochs=1000, lr=1e-3, betas=(0.9, 0.999),
               weight_decay=0.0, checkpoint_interval=50):
        """
        Execute spatially varying parameter inversion using Adam optimizer

        Args:
            max_epochs: Maximum number of optimization iterations
            lr: Learning rate for Adam optimizer
            betas: Coefficients for computing running averages of gradient and its square
            weight_decay: Weight decay (L2 penalty)
            checkpoint_interval: Interval for saving checkpoints
        """
        # Adam optimizer
        optimizer = optim.Adam(
            self.simulator.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )

        print("\n" + "=" * 70)
        print("Starting Spatially Varying Parameter Inversion (Adam, Fixed BC)")
        print("=" * 70)
        print(f"Total parameters: {6 * self.simulator.domain_shape[0] * self.simulator.domain_shape[1]:,}")
        print(f"Known: ρ = {self.simulator.rho_known} kg/m³")
        print(f"Regularization: TV={self.lambda_tv}")
        print(f"Adam settings: lr={lr}, betas={betas}, weight_decay={weight_decay}")
        print(f"Boundary condition: Fixed (u=0 on boundary)")
        print("=" * 70)

        best_loss = float('inf')
        patience = 100
        patience_counter = 0

        for epoch in range(max_epochs):
            # Zero gradients
            optimizer.zero_grad()

            # Forward simulation
            pred_ux_primary, pred_uy_primary = self.simulator(save_all_steps=False)

            # Compute loss
            total_loss, loss_data, loss_tv = self.compute_loss(
                pred_ux_primary, pred_uy_primary)

            # Backpropagation
            total_loss.backward()

            # Optimization step
            optimizer.step()

            # Get current loss values
            current_loss = total_loss.item()
            current_loss_data = loss_data.item()
            current_loss_tv = loss_tv if isinstance(loss_tv, float) else loss_tv.item()

            # Record history
            C11_m, C22_m, C12_m, C16_m, C26_m, C66_m, rho = self.simulator.get_mean_params()
            self.history['loss'].append(current_loss)
            self.history['loss_data'].append(current_loss_data)
            self.history['loss_tv'].append(current_loss_tv)
            self.history['C11_mean'].append(C11_m)
            self.history['C22_mean'].append(C22_m)
            self.history['C12_mean'].append(C12_m)
            self.history['C16_mean'].append(C16_m)
            self.history['C26_mean'].append(C26_m)
            self.history['C66_mean'].append(C66_m)

            # Check for improvement
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Print progress
            if (epoch + 1) % 1 == 0 or epoch == max_epochs - 1:
                print(f"Epoch {epoch + 1:4d}: Loss={current_loss:.6e} "
                      f"(Data={current_loss_data:.6e}, TV={current_loss_tv:.6e}) | "
                      f"Mean: C11={C11_m:.2f}, C22={C22_m:.2f}, C66={C66_m:.2f} GPa")

            # Save checkpoint
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch + 1)

            # Early stopping
            if patience_counter >= patience and epoch > 50:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            # Check convergence
            if epoch > 0 and abs(self.history['loss'][-1] - self.history['loss'][-2]) < 1e-40:
                print(f"Converged at epoch {epoch + 1} (loss change < 1e-40)")
                break

        return self.get_results()

    def save_checkpoint(self, epoch, output_dir="./checkpoints_fem_damage_adam_fixed_bc"):
        """Save parameter fields as checkpoint"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        C11, C22, C12, C16, C26, C66, rho = self.simulator.get_material_params()

        checkpoint = {
            'epoch': epoch,
            'C11': C11.detach().cpu().numpy(),
            'C22': C22.detach().cpu().numpy(),
            'C12': C12.detach().cpu().numpy(),
            'C16': C16.detach().cpu().numpy(),
            'C26': C26.detach().cpu().numpy(),
            'C66': C66.detach().cpu().numpy(),
        }

        filename = os.path.join(output_dir, f'checkpoint_epoch_{epoch:04d}.npz')
        np.savez_compressed(filename, **checkpoint)
        print(f"  Checkpoint saved: {filename}")

    def get_results(self):
        """Get final inversion results"""
        C11, C22, C12, C16, C26, C66, rho = self.simulator.get_material_params()

        return {
            'inverted_fields': {
                'C11': C11.detach().cpu().numpy(),
                'C22': C22.detach().cpu().numpy(),
                'C12': C12.detach().cpu().numpy(),
                'C16': C16.detach().cpu().numpy(),
                'C26': C26.detach().cpu().numpy(),
                'C66': C66.detach().cpu().numpy(),
            },
            'mean_values': {
                'C11': C11.mean().item() / 1e9,
                'C22': C22.mean().item() / 1e9,
                'C12': C12.mean().item() / 1e9,
                'C16': C16.mean().item() / 1e9,
                'C26': C26.mean().item() / 1e9,
                'C66': C66.mean().item() / 1e9,
            },
            'history': self.history
        }


def load_fem_damage_data(h5_file_path):
    """
    Load FEM reference data with damage information (Fixed BC version)
    """
    print(f"\nLoading FEM damage data (Fixed BC) from: {h5_file_path}")

    if not os.path.exists(h5_file_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_file_path}")

    with h5py.File(h5_file_path, 'r') as f:
        # Load displacement fields
        ux_data = f['ux'][:]
        uy_data = f['uy'][:]

        # Load coordinate arrays
        x_original = f['x'][:]
        y_original = f['y'][:]
        t_primary = f['t'][:]

        # Load damage mask
        damage_mask = None
        if 'damage_mask' in f:
            damage_mask = f['damage_mask'][:]

        # Load source information
        source_info = {}
        if 'source_info' in f:
            for key in f['source_info'].keys():
                source_info[key] = f['source_info'][key][()]

        # Load damage information
        damage_info = {}
        if 'damage_info' in f:
            if 'total_damaged_points' in f['damage_info']:
                damage_info['total_damaged_points'] = f['damage_info']['total_damaged_points'][()]
            if 'damage_percentage' in f['damage_info']:
                damage_info['damage_percentage'] = f['damage_info']['damage_percentage'][()]

        print(f"Loaded data shapes:")
        print(f"  ux, uy: {ux_data.shape}")
        print(f"  Spatial grid: {len(x_original)}×{len(y_original)}")
        print(f"  Time steps: {len(t_primary)}")
        if damage_mask is not None:
            print(f"  Damage mask: {damage_mask.shape}")
            if 'damage_percentage' in damage_info:
                print(f"  Damage percentage: {damage_info['damage_percentage']:.2f}%")

    # Convert to PyTorch tensors
    # Data shape from FEM: (nt, nx, ny) -> (1, nt, nx, ny)
    ux = torch.tensor(ux_data, dtype=torch.float32).unsqueeze(0).to(device)
    uy = torch.tensor(uy_data, dtype=torch.float32).unsqueeze(0).to(device)

    # Extract primary stride from filename
    match = re.search(r'_ds(\d+)', h5_file_path)
    primary_stride = int(match.group(1)) if match else 5

    metadata = {
        'x_original': x_original,
        'y_original': y_original,
        't_primary': t_primary,
        'source_info': source_info,
        'primary_stride': primary_stride,
        'damage_mask': damage_mask,
        'damage_info': damage_info
    }

    return ux, uy, metadata


def visualize_parameter_fields(results, metadata, output_dir="./fem_damage_inversion_results_adam_fixed_bc"):
    """Visualize inverted parameter fields (Fixed BC version)"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fields = results['inverted_fields']
    x = metadata['x_original']
    y = metadata['y_original']
    damage_mask = metadata.get('damage_mask')

    # Create meshgrid
    X, Y = np.meshgrid(x, y, indexing='ij')

    params = ['C11', 'C22', 'C12', 'C16', 'C26', 'C66']

    # True values for reference (matching fixed BC FEM code)
    true_values = {
        'C11': 130.74,  # GPa
        'C22': 11.50,
        'C12': 10.49,
        'C16': 30.67,
        'C26': 3.75,
        'C66': 14.77
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Inverted Spatially Varying Elastic Parameters (FEM Fixed BC, Adam)',
                 fontsize=16, fontweight='bold')

    for idx, param in enumerate(params):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        field = fields[param][0, :, :] / 1e9  # Convert to GPa

        im = ax.imshow(field.T, origin='lower',
                       extent=[x.min() * 1e2, x.max() * 1e2, y.min() * 1e2, y.max() * 1e2],
                       cmap='jet', aspect='equal')

        # Overlay damage mask if available
        if damage_mask is not None:
            ax.contour(X.T * 1e2, Y.T * 1e2, damage_mask.T,
                       levels=[0.5], colors='black', linewidths=2, linestyles='--')

        ax.set_xlabel('x (cm)', fontsize=11)
        ax.set_ylabel('y (cm)', fontsize=11)
        ax.set_title(f'{param} (GPa)\nMean: {field.mean():.2f}, True: {true_values[param]:.2f} GPa', fontsize=12)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'inverted_parameter_fields_fixed_bc.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Parameter fields saved to: {save_path}")
    plt.show()


def visualize_convergence(results, output_dir="./fem_damage_inversion_results_adam_fixed_bc"):
    """Visualize convergence history (Fixed BC version)"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    history = results['history']

    # True values (matching fixed BC FEM code, in GPa)
    true_values = {
        'C11': 130.74,
        'C22': 11.50,
        'C12': 10.49,
        'C16': 30.67,
        'C26': 3.75,
        'C66': 14.77
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Inversion Convergence (FEM Fixed BC Data, Adam)', fontsize=14, fontweight='bold')

    # Loss components
    ax = axes[0, 0]
    ax.semilogy(history['loss'], 'b-', linewidth=2, label='Total Loss')
    ax.semilogy(history['loss_data'], 'r--', linewidth=1.5, label='Data Loss')
    if max(history['loss_tv']) > 0:
        ax.semilogy(history['loss_tv'], 'g--', linewidth=1.5, label='TV Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mean C11, C22, C66
    ax = axes[0, 1]
    ax.plot(history['C11_mean'], 'b-', linewidth=2, label='C11')
    ax.plot(history['C22_mean'], 'r-', linewidth=2, label='C22')
    ax.plot(history['C66_mean'], 'g-', linewidth=2, label='C66')
    ax.axhline(y=true_values['C11'], color='b', linestyle='--', alpha=0.5, label=f'C11 true={true_values["C11"]:.1f}')
    ax.axhline(y=true_values['C22'], color='r', linestyle='--', alpha=0.5, label=f'C22 true={true_values["C22"]:.1f}')
    ax.axhline(y=true_values['C66'], color='g', linestyle='--', alpha=0.5, label=f'C66 true={true_values["C66"]:.1f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Value (GPa)')
    ax.set_title('Mean C11, C22, C66 Evolution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Mean C12, C16, C26
    ax = axes[1, 0]
    ax.plot(history['C12_mean'], 'b-', linewidth=2, label='C12')
    ax.plot(history['C16_mean'], 'r-', linewidth=2, label='C16')
    ax.plot(history['C26_mean'], 'g-', linewidth=2, label='C26')
    ax.axhline(y=true_values['C12'], color='b', linestyle='--', alpha=0.5, label=f'C12 true={true_values["C12"]:.1f}')
    ax.axhline(y=true_values['C16'], color='r', linestyle='--', alpha=0.5, label=f'C16 true={true_values["C16"]:.1f}')
    ax.axhline(y=true_values['C26'], color='g', linestyle='--', alpha=0.5, label=f'C26 true={true_values["C26"]:.1f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Value (GPa)')
    ax.set_title('Mean C12, C16, C26 Evolution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Data loss only (zoomed)
    ax = axes[1, 1]
    ax.semilogy(history['loss_data'], 'r-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Data Loss')
    ax.set_title('Data Fidelity Loss (Detail)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'convergence_history_fixed_bc.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Convergence plot saved to: {save_path}")
    plt.show()


def main():
    """
    Main function for spatially varying parameter inversion (FEM Fixed BC version)
    Using Adam optimizer
    """
    print("=" * 70)
    print("ANISOTROPIC SPATIALLY VARYING PARAMETER INVERSION - FEM FIXED BC DATA")
    print("Optimizer: Adam")
    print("Boundary Condition: Fixed (u=0 on boundary)")
    print("Material: Composite material, Source: Ricker wavelet")
    print("=" * 70)
    print(f"Device: {device}")

    # ============= CONFIGURATION =============
    SECONDARY_TIME_STRIDE = 2  # Time downsampling (1=no downsampling)
    LAMBDA_TV = 1e-21  # Total Variation regularization

    # Adam configuration
    ADAM_LR = 0.02  # Adam learning rate
    ADAM_BETAS = (0.9, 0.999)  # Adam beta parameters
    ADAM_WEIGHT_DECAY = 0.0  # Weight decay

    # Material parameters (matching fixed BC FEM code)
    RHO_KNOWN = 1610.0  # kg/m³

    # True material values (for reference)
    C11_TRUE = 130.74e9  # Pa
    C22_TRUE = 11.50e9
    C12_TRUE = 10.49e9
    C16_TRUE = 30.67e9
    C26_TRUE = 3.75e9
    C66_TRUE = 14.77e9

    print(f"\nConfiguration:")
    print(f"  SECONDARY time stride: {SECONDARY_TIME_STRIDE}")
    print(f"  NO spatial downsampling (full grid)")
    print(f"  TV regularization: {LAMBDA_TV}")
    print(f"  Known density: {RHO_KNOWN} kg/m³")
    print(f"  Boundary condition: Fixed (u=0 on boundary)")
    print(f"\nAdam Configuration:")
    print(f"  Learning rate: {ADAM_LR}")
    print(f"  Betas: {ADAM_BETAS}")
    print(f"  Weight decay: {ADAM_WEIGHT_DECAY}")
    print(f"\nTrue material parameters:")
    print(f"  C11 = {C11_TRUE / 1e9:.2f} GPa")
    print(f"  C22 = {C22_TRUE / 1e9:.2f} GPa")
    print(f"  C12 = {C12_TRUE / 1e9:.2f} GPa")
    print(f"  C16 = {C16_TRUE / 1e9:.2f} GPa")
    print(f"  C26 = {C26_TRUE / 1e9:.2f} GPa")
    print(f"  C66 = {C66_TRUE / 1e9:.2f} GPa")

    # ============= LOAD DATA =============
    # FEM Fixed BC data file path
    h5_file_path = "./data_anisotropic_fem_ricker_fixed_bc_damage/anisotropic_ricker_damage_data_ds5_FEM_fixed_bc.mat"

    try:
        ux_primary, uy_primary, metadata = load_fem_damage_data(h5_file_path)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run the FEM forward simulation with damage and fixed BC first.")
        return

    # ============= EXTRACT PARAMETERS =============
    x_original = metadata['x_original']
    y_original = metadata['y_original']
    t_primary = metadata['t_primary']
    source_info = metadata['source_info']
    primary_stride = metadata['primary_stride']

    nx_original = len(x_original)
    ny_original = len(y_original)
    nt_primary = len(t_primary)

    Lx = x_original[-1] - x_original[0]
    h_original = Lx / (nx_original - 1)

    dt_primary = t_primary[1] - t_primary[0]
    total_time = t_primary[-1]

    nt_original = (nt_primary - 1) * primary_stride + 1
    dt_original = total_time / (nt_original - 1)

    domain_shape_original = (nx_original, ny_original)

    # Source parameters from data
    source_f0 = source_info.get('source_f0', 100e3)  # 100 kHz
    source_t0 = source_info.get('source_t0', 10e-6)  # 10 μs (matching fixed BC FEM code)
    source_amplitude = source_info.get('source_amplitude', 1e10)
    gaussian_param = source_info.get('gaussian_param', 5e6)

    print(f"\n{'=' * 70}")
    print(f"GRID CONFIGURATION:")
    print(f"  Spatial: {nx_original}×{ny_original} = {nx_original * ny_original:,} points")
    print(f"  Time steps (original): {nt_original}")
    print(f"  Time steps (PRIMARY sampled): {nt_primary}")
    print(f"  Total parameters to invert: 6 × {nx_original * ny_original:,} = {6 * nx_original * ny_original:,}")
    print(f"\nSource parameters (Ricker wavelet):")
    print(f"  f0 = {source_f0 / 1e3:.1f} kHz")
    print(f"  t0 = {source_t0 * 1e6:.1f} μs")
    print(f"  Amplitude = {source_amplitude:.2e}")
    print(f"{'=' * 70}")

    # ============= CREATE SIMULATOR =============
    simulator = SpatiallyVaryingAnisotropicSimulator(
        domain_shape=domain_shape_original,
        h=h_original,
        dt=dt_original,
        nt=nt_original,
        primary_time_stride=primary_stride,
        source_component='y',
        source_f0=source_f0,
        source_t0=source_t0,
        source_amplitude=source_amplitude,
        total_time=total_time,
        # Initial guess: True material values (healthy)
        C11_init=C11_TRUE,
        C22_init=C22_TRUE,
        C12_init=C12_TRUE,
        C16_init=C16_TRUE,
        C26_init=C26_TRUE,
        C66_init=C66_TRUE,
        rho_known=RHO_KNOWN,
        gaussian_param=gaussian_param,
        use_log_params=True
    ).to(device)

    # ============= RUN INVERSION =============
    inverter = SpatiallyVaryingParameterInverter(
        simulator,
        ux_primary,
        uy_primary,
        secondary_time_stride=SECONDARY_TIME_STRIDE,
        lambda_tv=LAMBDA_TV
    )

    # Run inversion with Adam optimizer
    results = inverter.invert(
        max_epochs=800,
        lr=ADAM_LR,
        betas=ADAM_BETAS,
        weight_decay=ADAM_WEIGHT_DECAY,
        checkpoint_interval=50
    )

    # ============= VISUALIZE RESULTS =============
    output_dir = "./fem_damage_inversion_results_adam_fixed_bc"
    visualize_parameter_fields(results, metadata, output_dir)
    visualize_convergence(results, output_dir)

    # ============= SAVE RESULTS =============
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save final parameter fields
    np.savez_compressed(
        os.path.join(output_dir, 'inverted_parameter_fields_fixed_bc.npz'),
        **results['inverted_fields'],
        mean_values=results['mean_values'],
        history=results['history'],
        damage_mask=metadata.get('damage_mask'),
        x=x_original,
        y=y_original,
        true_values={
            'C11': C11_TRUE, 'C22': C22_TRUE, 'C12': C12_TRUE,
            'C16': C16_TRUE, 'C26': C26_TRUE, 'C66': C66_TRUE,
            'rho': RHO_KNOWN
        }
    )

    print(f"\nResults saved to {output_dir}/")
    print("\nInversion completed successfully!")

    # Print final comparison
    print("\n" + "=" * 70)
    print("FINAL RESULTS COMPARISON:")
    print("=" * 70)
    mean_vals = results['mean_values']
    print(f"{'Parameter':<10} {'True (GPa)':<15} {'Inverted Mean (GPa)':<20} {'Error (%)':<10}")
    print("-" * 55)
    print(
        f"{'C11':<10} {C11_TRUE / 1e9:<15.2f} {mean_vals['C11']:<20.2f} {100 * abs(mean_vals['C11'] - C11_TRUE / 1e9) / (C11_TRUE / 1e9):<10.2f}")
    print(
        f"{'C22':<10} {C22_TRUE / 1e9:<15.2f} {mean_vals['C22']:<20.2f} {100 * abs(mean_vals['C22'] - C22_TRUE / 1e9) / (C22_TRUE / 1e9):<10.2f}")
    print(
        f"{'C12':<10} {C12_TRUE / 1e9:<15.2f} {mean_vals['C12']:<20.2f} {100 * abs(mean_vals['C12'] - C12_TRUE / 1e9) / (C12_TRUE / 1e9):<10.2f}")
    print(
        f"{'C16':<10} {C16_TRUE / 1e9:<15.2f} {mean_vals['C16']:<20.2f} {100 * abs(mean_vals['C16'] - C16_TRUE / 1e9) / (C16_TRUE / 1e9):<10.2f}")
    print(
        f"{'C26':<10} {C26_TRUE / 1e9:<15.2f} {mean_vals['C26']:<20.2f} {100 * abs(mean_vals['C26'] - C26_TRUE / 1e9) / (C26_TRUE / 1e9):<10.2f}")
    print(
        f"{'C66':<10} {C66_TRUE / 1e9:<15.2f} {mean_vals['C66']:<20.2f} {100 * abs(mean_vals['C66'] - C66_TRUE / 1e9) / (C66_TRUE / 1e9):<10.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()