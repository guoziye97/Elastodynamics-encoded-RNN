"""
CFRP Anisotropic Elastic Parameter Inversion - Ricker Wavelet Source

"""

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False



def ricker_wavelet(t, f0, t0, amplitude=1.0):
    """
    Mexican hat wavelet
    R(t) = A * (1 - 2*pi^2*f0^2*(t-t0)^2) * exp(-pi^2*f0^2*(t-t0)^2)
    """
    tau = t - t0
    pi_f0_tau_sq = (np.pi * f0 * tau) ** 2
    return amplitude * (1 - 2 * pi_f0_tau_sq) * np.exp(-pi_f0_tau_sq)


def gaussian_point_load(X, Y, x0, y0, a=150):

    r_squared = (X - x0) ** 2 + (Y - y0) ** 2
    return (a / np.pi) * torch.exp(-a * r_squared)


def conv2d_fixed(field, kernel):
    """
    Convolution with zero-padding (fixed boundary conditions)

    """

    field_pad = F.pad(field, (1, 1, 1, 1), mode='constant', value=0.0)
    return F.conv2d(field_pad, kernel, padding=0)


def _general_anisotropic_elastic_operator_with_force(ux, uy, C11, C22, C12, C16, C26, C66, h, force_x, force_y, rho):

    h_inv_sq = h ** (-2)

    # Define finite difference operators
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


    ux_unsqueezed = ux.unsqueeze(1)
    uy_unsqueezed = uy.unsqueeze(1)

    d2ux_dx2 = conv2d_fixed(ux_unsqueezed, d2dx2).squeeze(1)
    d2ux_dy2 = conv2d_fixed(ux_unsqueezed, d2dy2).squeeze(1)
    d2uy_dx2 = conv2d_fixed(uy_unsqueezed, d2dx2).squeeze(1)
    d2uy_dy2 = conv2d_fixed(uy_unsqueezed, d2dy2).squeeze(1)
    d2ux_dxdy = conv2d_fixed(ux_unsqueezed, d2dxdy).squeeze(1)
    d2uy_dxdy = conv2d_fixed(uy_unsqueezed, d2dxdy).squeeze(1)

    # General anisotropic elasticity operator
    Lux = (C11 * d2ux_dx2 +
           C66 * d2ux_dy2 +
           (C12 + C66) * d2uy_dxdy +
           2 * C16 * d2ux_dxdy +
           C16 * d2uy_dx2 +
           C26 * d2uy_dy2)

    Luy = (C66 * d2uy_dx2 +
           C22 * d2uy_dy2 +
           (C12 + C66) * d2ux_dxdy +
           C16 * d2ux_dx2 +
           C26 * d2ux_dy2 +
           2 * C26 * d2uy_dxdy)

    Lux = Lux.transpose(1, 2)
    Luy = Luy.transpose(1, 2)

    Lux = (Lux + force_x) / rho
    Luy = (Luy + force_y) / rho

    return Lux, Luy


def add_noise_snr(signal, snr_db):

    signal_power = torch.mean(signal ** 2)

    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = torch.randn_like(signal) * torch.sqrt(noise_power)

    noisy_signal = signal + noise

    actual_signal_power = torch.mean(signal ** 2)
    actual_noise_power = torch.mean(noise ** 2)
    actual_snr_db = 10 * torch.log10(actual_signal_power / actual_noise_power)

    print(f"  Target SNR: {snr_db:.2f} dB")
    print(f"  Actual SNR: {actual_snr_db:.2f} dB")
    print(f"  Signal RMS: {torch.sqrt(signal_power):.6e}")
    print(f"  Noise RMS: {torch.sqrt(noise_power):.6e}")

    return noisy_signal, noise


class CFRPWaveSimulatorRicker(nn.Module):
    """
    CFRP Anisotropic Elastic Wave Simulator - Ricker Wavelet Source

    The simulator always runs on the original full-resolution grid.
    """

    def __init__(self, domain_shape, h, dt, nt, primary_time_stride,
                 source_component='y',
                 source_f0=100e3,
                 source_t0=6e-6,
                 source_amplitude=1.0,
                 C11_init=100e9, C22_init=10e9, C12_init=8e9,
                 C16_init=25e9, C26_init=3e9, C66_init=12e9,
                 rho_known=1600.0, gaussian_param=5e6,
                 source_position=(0.0, 0.0)):
        super().__init__()

        # Spatial and temporal parameters (original full resolution)
        self.domain_shape = domain_shape
        self.h = h
        self.dt = dt
        self.nt = nt  # Original time steps
        self.primary_time_stride = primary_time_stride  # PRIMARY subsampling stride

        # Ricker wavelet source parameters
        self.source_component = source_component
        self.source_f0 = source_f0
        self.source_t0 = source_t0
        self.source_amplitude = source_amplitude
        self.gaussian_param = gaussian_param
        self.source_position = source_position

        # Material parameters - direct stiffness matrix components
        self.log_C11 = nn.Parameter(torch.log(torch.tensor(C11_init, dtype=torch.float32)))
        self.log_C22 = nn.Parameter(torch.log(torch.tensor(C22_init, dtype=torch.float32)))
        self.log_C12 = nn.Parameter(torch.log(torch.tensor(C12_init, dtype=torch.float32)))
        self.log_C16 = nn.Parameter(torch.log(torch.tensor(C16_init, dtype=torch.float32)))
        self.log_C26 = nn.Parameter(torch.log(torch.tensor(C26_init, dtype=torch.float32)))
        self.log_C66 = nn.Parameter(torch.log(torch.tensor(C66_init, dtype=torch.float32)))

        # Known density (fixed, not optimized)
        self.register_buffer('rho', torch.tensor(rho_known, dtype=torch.float32))
        self.rho_known = rho_known

        # Parameter bounds (Pa)
        self.C11_bounds = (1e9, 10000e9)
        self.C22_bounds = (1e9, 10000e9)
        self.C12_bounds = (1e9, 10000e9)
        self.C16_bounds = (1e9, 10000e9)
        self.C26_bounds = (1e9, 10000e9)
        self.C66_bounds = (1e9, 10000e9)

        # Create coordinate grid for Gaussian calculation
        nx, ny = domain_shape
        Lx = h * (nx - 1)
        Ly = h * (ny - 1)
        x = torch.linspace(-Lx / 2, Lx / 2, nx, device=device)
        y = torch.linspace(-Ly / 2, Ly / 2, ny, device=device)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        # Precompute Gaussian spatial distribution
        x0_src, y0_src = source_position
        self.gaussian_distribution = gaussian_point_load(X, Y, x0_src, y0_src, gaussian_param)
        self.register_buffer('gaussian_dist', self.gaussian_distribution)

        # Generate Ricker wavelet source signal (full time resolution)
        self._generate_ricker_source_signal()

        print(f"\nCFRP Simulator (Ricker Wavelet) initialized:")
        print(f"  Domain (FULL): {domain_shape[0]}x{domain_shape[1]} = {domain_shape[0] * domain_shape[1]:,} points")
        print(f"  Time steps (FULL): {nt}")
        print(f"  dt: {dt * 1e9:.4f} ns")
        print(f"  PRIMARY time stride for output: {primary_time_stride}")
        print(f"  Output time steps: {(nt + primary_time_stride - 1) // primary_time_stride}")
        print(f"  Known density: rho = {rho_known} kg/m^3")
        print(f"  Ricker wavelet: f0={source_f0 / 1e3:.1f} kHz, t0={source_t0 * 1e6:.1f} us")

    def _generate_ricker_source_signal(self):
        """Generate Ricker wavelet source signal (full time resolution)."""
        t = np.arange(self.nt) * self.dt

        # Compute Ricker wavelet
        signal = ricker_wavelet(t, self.source_f0, self.source_t0, self.source_amplitude)

        self.register_buffer('source_signal', torch.tensor(signal, dtype=torch.float32).unsqueeze(0))

        # Print Ricker wavelet info
        max_idx = np.argmax(np.abs(signal))
        print(f"  Ricker signal: max at t={t[max_idx] * 1e6:.2f} us, peak={signal[max_idx]:.4e}")

    def get_material_params(self):
        """Get current stiffness matrix parameters (with clamping)."""
        C11 = torch.clamp(torch.exp(self.log_C11), *self.C11_bounds)
        C22 = torch.clamp(torch.exp(self.log_C22), *self.C22_bounds)
        C12 = torch.clamp(torch.exp(self.log_C12), *self.C12_bounds)
        C16 = torch.clamp(torch.exp(self.log_C16), *self.C16_bounds)
        C26 = torch.clamp(torch.exp(self.log_C26), *self.C26_bounds)
        C66 = torch.clamp(torch.exp(self.log_C66), *self.C66_bounds)

        return C11, C22, C12, C16, C26, C66, self.rho

    def get_current_params_values(self):
        """Get current parameter values (displayed in GPa)."""
        C11, C22, C12, C16, C26, C66, rho = self.get_material_params()
        return (C11.item() / 1e9, C22.item() / 1e9, C12.item() / 1e9,
                C16.item() / 1e9, C26.item() / 1e9, C66.item() / 1e9, rho.item())

    def forward(self, save_all_steps=False):
        """
        Run forward simulation (Ricker wavelet source).

        Args:
            save_all_steps: If True, save all time steps; if False, only save
                            PRIMARY subsampled time steps.

        Returns:
            ux_fields, uy_fields: shape (1, nt_output, nx, ny)
        """
        device = self.source_signal.device
        batch_size = 1

        # Initialize fields
        hidden_shape = (batch_size,) + self.domain_shape
        ux1 = torch.zeros(hidden_shape, device=device)
        uy1 = torch.zeros(hidden_shape, device=device)
        ux2 = torch.zeros(hidden_shape, device=device)
        uy2 = torch.zeros(hidden_shape, device=device)

        # Get material parameters
        C11, C22, C12, C16, C26, C66, rho = self.get_material_params()

        ux_history = []
        uy_history = []

        # Time-stepping loop (runs at full resolution)
        for i in range(self.nt):
            # Create force source term (Ricker wavelet x Gaussian spatial distribution)
            force_amplitude = self.source_signal[0, i]

            if self.source_component == 'x':
                force_x = force_amplitude * self.gaussian_dist.unsqueeze(0)
                force_y = torch.zeros_like(ux1)
            else:  # 'y'
                force_x = torch.zeros_like(ux1)
                force_y = force_amplitude * self.gaussian_dist.unsqueeze(0)

            # Compute elastic operator (including force source)
            Lux1, Luy1 = _general_anisotropic_elastic_operator_with_force(
                ux1, uy1, C11, C22, C12, C16, C26, C66, self.h,
                force_x, force_y, rho
            )

            # Time stepping (central difference)
            ux_new = 2 * ux1 - ux2 + self.dt ** 2 * Lux1
            uy_new = 2 * uy1 - uy2 + self.dt ** 2 * Luy1

            # Update state
            ux2, uy2 = ux1, uy1
            ux1, uy1 = ux_new, uy_new

            # Save history based on sampling strategy
            if save_all_steps:
                ux_history.append(ux1)
                uy_history.append(uy1)
            else:
                # Only save PRIMARY subsampled time steps
                if i % self.primary_time_stride == 0:
                    ux_history.append(ux1)
                    uy_history.append(uy1)

        # Stack time history
        ux_fields = torch.stack(ux_history, dim=1)
        uy_fields = torch.stack(uy_history, dim=1)

        return ux_fields, uy_fields


# ============================================================================
# Parameter Inverter Class (Adam Optimizer)
# ============================================================================

class CFRPParameterInverterRicker:
    """
    CFRP Elastic Parameter Inverter - Ricker Wavelet Version
    Uses Adam optimizer.
    """

    def __init__(self, simulator, target_ux_primary, target_uy_primary,
                 secondary_time_stride=1, secondary_space_stride=1):
        """
        Initialize the inverter.

        Args:
            simulator: CFRPWaveSimulatorRicker instance
            target_ux_primary: Target x-displacement data (after PRIMARY subsampling)
            target_uy_primary: Target y-displacement data (after PRIMARY subsampling)
            secondary_time_stride: SECONDARY temporal subsampling stride
            secondary_space_stride: SECONDARY spatial subsampling stride
        """
        self.simulator = simulator
        self.target_ux_primary = target_ux_primary
        self.target_uy_primary = target_uy_primary
        self.secondary_time_stride = secondary_time_stride
        self.secondary_space_stride = secondary_space_stride

        # Apply SECONDARY subsampling to target data
        self.target_ux_secondary = target_ux_primary[:, ::secondary_time_stride,
                                   ::secondary_space_stride,
                                   ::secondary_space_stride]
        self.target_uy_secondary = target_uy_primary[:, ::secondary_time_stride,
                                   ::secondary_space_stride,
                                   ::secondary_space_stride]

        # History records
        self.history = {
            'loss': [],
            'C11': [], 'C22': [], 'C12': [],
            'C16': [], 'C26': [], 'C66': []
        }

        print(f"\nInverter (Ricker) initialized:")
        print(f"  Target output (PRIMARY sampled): {target_ux_primary.shape}")
        print(f"  Loss computation (SECONDARY sampled): {self.target_ux_secondary.shape}")
        print(f"  SECONDARY sampling - Time: {secondary_time_stride}, Space: {secondary_space_stride}")

    def compute_loss(self, pred_ux_primary, pred_uy_primary):
        """
        Compute the loss function.
        """
        # Apply SECONDARY subsampling to predictions
        pred_ux_secondary = pred_ux_primary[:, ::self.secondary_time_stride,
                            ::self.secondary_space_stride,
                            ::self.secondary_space_stride]
        pred_uy_secondary = pred_uy_primary[:, ::self.secondary_time_stride,
                            ::self.secondary_space_stride,
                            ::self.secondary_space_stride]

        # Ensure shape matching
        if pred_ux_secondary.shape != self.target_ux_secondary.shape:
            print(f"Warning: Shape mismatch!")
            print(f"  Predicted: {pred_ux_secondary.shape}")
            print(f"  Target: {self.target_ux_secondary.shape}")

            # Trim to minimum shape
            min_t = min(pred_ux_secondary.shape[1], self.target_ux_secondary.shape[1])
            min_x = min(pred_ux_secondary.shape[2], self.target_ux_secondary.shape[2])
            min_y = min(pred_ux_secondary.shape[3], self.target_uy_secondary.shape[3])

            pred_ux_secondary = pred_ux_secondary[:, :min_t, :min_x, :min_y]
            pred_uy_secondary = pred_uy_secondary[:, :min_t, :min_x, :min_y]
            target_ux = self.target_ux_secondary[:, :min_t, :min_x, :min_y]
            target_uy = self.target_uy_secondary[:, :min_t, :min_x, :min_y]
        else:
            target_ux = self.target_ux_secondary
            target_uy = self.target_uy_secondary

        loss_ux = torch.mean((pred_ux_secondary - target_ux) ** 2)
        loss_uy = torch.mean((pred_uy_secondary - target_uy) ** 2)
        return loss_ux + loss_uy

    def invert(self, max_epochs=500, lr=1e-3, true_params=None):
        """
        Execute parameter inversion (Adam optimizer).

        Args:
            max_epochs: Maximum number of iterations
            lr: Learning rate (typical Adam range: 1e-3 to 1e-4)
            true_params: Dictionary of true parameters (for display)
        """
        # Adam optimizer configuration
        optimizer = optim.Adam(
            self.simulator.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        print("\n" + "=" * 70)
        print("Starting CFRP Direct Stiffness Matrix Inversion (Ricker Source)")
        print("Optimizer: Adam")
        print("=" * 70)
        print(f"Known parameter: rho = {self.simulator.rho_known} kg/m^3")
        print(f"Learning rate: {lr}")

        if true_params:
            print("\nTrue parameters:")
            print(f"  C11={true_params['C11']:.2f} GPa, C22={true_params['C22']:.2f} GPa, "
                  f"C12={true_params['C12']:.2f} GPa")
            print(f"  C16={true_params['C16']:.2f} GPa, C26={true_params['C26']:.2f} GPa, "
                  f"C66={true_params['C66']:.2f} GPa")

        C11_init, C22_init, C12_init, C16_init, C26_init, C66_init, _ = \
            self.simulator.get_current_params_values()
        print(f"\nInitial guess:")
        print(f"  C11={C11_init:.2f} GPa, C22={C22_init:.2f} GPa, C12={C12_init:.2f} GPa")
        print(f"  C16={C16_init:.2f} GPa, C26={C26_init:.2f} GPa, C66={C66_init:.2f} GPa")
        print("=" * 70)

        best_loss = float('inf')
        best_params = None
        patience = 50
        patience_counter = 0

        for epoch in range(max_epochs):
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            pred_ux_primary, pred_uy_primary = self.simulator(save_all_steps=False)

            # Compute loss
            loss = self.compute_loss(pred_ux_primary, pred_uy_primary)

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            loss_val = loss.item()

            # Record history
            C11, C22, C12, C16, C26, C66, rho = self.simulator.get_current_params_values()
            self.history['loss'].append(loss_val)
            self.history['C11'].append(C11)
            self.history['C22'].append(C22)
            self.history['C12'].append(C12)
            self.history['C16'].append(C16)
            self.history['C26'].append(C26)
            self.history['C66'].append(C66)

            # Check for best loss
            if loss_val < best_loss:
                best_loss = loss_val
                best_params = (C11, C22, C12, C16, C26, C66)
                patience_counter = 0
            else:
                patience_counter += 1

            # Print progress
            if (epoch + 1) % 1 == 0 or epoch == 0 or epoch == max_epochs - 1:
                print(f"Epoch {epoch + 1:4d}: Loss={loss_val:.6e} | "
                      f"C11={C11:.3f}, C22={C22:.3f}, C12={C12:.3f} GPa | "
                      f"C16={C16:.3f}, C26={C26:.3f}, C66={C66:.3f} GPa")

            # Early stopping
            if patience_counter >= patience and epoch > 100:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        return self.get_results(true_params)

    def get_results(self, true_params=None):
        """Get inversion results."""
        C11, C22, C12, C16, C26, C66, rho = self.simulator.get_current_params_values()

        results = {
            'inverted': {
                'C11': C11, 'C22': C22, 'C12': C12,
                'C16': C16, 'C26': C26, 'C66': C66
            },
            'known': {'rho': self.simulator.rho_known},
            'history': self.history
        }

        if true_params:
            results['true'] = true_params
            results['error'] = {
                'C11': abs(C11 - true_params['C11']) / true_params['C11'] if true_params['C11'] != 0 else 0,
                'C22': abs(C22 - true_params['C22']) / true_params['C22'] if true_params['C22'] != 0 else 0,
                'C12': abs(C12 - true_params['C12']) / true_params['C12'] if true_params['C12'] != 0 else 0,
                'C16': abs(C16 - true_params['C16']) / true_params['C16'] if true_params['C16'] != 0 else 0,
                'C26': abs(C26 - true_params['C26']) / true_params['C26'] if true_params['C26'] != 0 else 0,
                'C66': abs(C66 - true_params['C66']) / true_params['C66'] if true_params['C66'] != 0 else 0
            }

        return results


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_reference_data(h5_file_path, add_noise=False, snr_db=20.0):
    """
    Load reference data that has already undergone PRIMARY subsampling,
    with optional noise addition.

    Args:
        h5_file_path: Path to HDF5 file
        add_noise: Whether to add noise
        snr_db: Signal-to-noise ratio (dB), default 20 dB

    Returns:
        ux, uy: Displacement field tensors (possibly with noise)
        metadata: Metadata dictionary
    """
    print(f"\nLoading reference data from: {h5_file_path}")

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

        # Load source information
        source_info = {}
        if 'source_info' in f:
            for key in f['source_info'].keys():
                source_info[key] = f['source_info'][key][()]

        print(f"Loaded data shapes:")
        print(f"  ux, uy: {ux_data.shape}")
        print(f"  Original spatial grid: {len(x_original)}x{len(y_original)}")
        print(f"  PRIMARY sampled time steps: {len(t_primary)}")

    # Convert to PyTorch tensors
    ux = torch.tensor(ux_data, dtype=torch.float32).unsqueeze(0).to(device)
    uy = torch.tensor(uy_data, dtype=torch.float32).unsqueeze(0).to(device)

    # Add noise (if requested)
    if add_noise:
        print(f"\nAdding noise to data (SNR = {snr_db} dB):")
        print("ux component:")
        ux, noise_ux = add_noise_snr(ux, snr_db)
        print("uy component:")
        uy, noise_uy = add_noise_snr(uy, snr_db)

        # Store noise information in metadata
        metadata_noise = {
            'noise_added': True,
            'snr_db': snr_db,
            'noise_ux': noise_ux.cpu().numpy(),
            'noise_uy': noise_uy.cpu().numpy()
        }
    else:
        metadata_noise = {'noise_added': False}

    # Extract PRIMARY subsampling stride from filename
    match = re.search(r'_ds(\d+)', h5_file_path)
    primary_stride = int(match.group(1)) if match else 5

    metadata = {
        'x_original': x_original,
        'y_original': y_original,
        't_primary': t_primary,
        'source_info': source_info,
        'primary_stride': primary_stride,
        **metadata_noise
    }

    return ux, uy, metadata


# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_inversion_results(results, output_dir="./inversion_results_ricker_adam"):
    """Visualize inversion results."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('CFRP Stiffness Matrix Inversion Results (Ricker Source, Adam)',
                 fontsize=14, fontweight='bold')

    params = ['C11', 'C22', 'C12', 'C16', 'C26', 'C66']

    for idx, param in enumerate(params):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        history = results['history'][param]
        final_val = results['inverted'][param]

        ax.plot(history, 'b-', linewidth=2, label='Inverted')

        if 'true' in results:
            true_val = results['true'][param]
            error_pct = results['error'][param] * 100
            ax.axhline(y=true_val, color='r', linestyle='--', linewidth=2, label='True')
            ax.set_title(f'{param}: True={true_val:.2f}, Final={final_val:.2f} GPa\nError: {error_pct:.2f}%',
                         fontsize=10)
        else:
            ax.set_title(f'{param}: Final={final_val:.2f} GPa', fontsize=10)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(f'{param} (GPa)', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(output_dir, 'inversion_convergence_ricker_adam.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Convergence plot saved to: {save_path}")
    plt.show()


def plot_ricker_comparison(t, f0, t0, amplitude, output_dir="./inversion_results_ricker_adam"):
    """Plot Ricker wavelet."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ricker = ricker_wavelet(t, f0, t0, amplitude)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t * 1e6, ricker, 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=t0 * 1e6, color='r', linestyle='--', alpha=0.7, label=f't0 = {t0 * 1e6:.1f} us')
    ax.set_title(f'Ricker Wavelet (f0 = {f0 / 1e3:.0f} kHz)', fontsize=14)
    ax.set_xlabel('t (us)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'ricker_wavelet_inversion.png')
    plt.savefig(save_path, dpi=150)
    print(f"Ricker wavelet plot saved to: {save_path}")
    plt.close()


# ============================================================================
# Main Function
# ============================================================================

def main():
    """
    Main function: CFRP parameter inversion (Ricker wavelet, Adam optimizer)
    """
    print("=" * 70)
    print("CFRP ANISOTROPIC ELASTIC PARAMETER INVERSION")
    print("Ricker Wavelet Source - Adam Optimizer")
    print("=" * 70)
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        torch.cuda.empty_cache()

    # ============= Configuration =============
    # Subsampling configuration
    SECONDARY_TIME_STRIDE = 2
    SECONDARY_SPACE_STRIDE = 8

    # Known density (set according to your forward simulation data)
    RHO_KNOWN = 1610.0  # kg/m^3

    # Ricker wavelet parameters (must match forward simulation)
    SOURCE_F0 = 100e3    # Center frequency 100 kHz
    SOURCE_T0 = 10e-6    # Time delay 10 us
    SOURCE_AMPLITUDE = 1e10
    GAUSSIAN_PARAM = 5e6  # Spatial Gaussian parameter
    SOURCE_POSITION = (0.0, 0.0)  # Source position

    # True parameters (extracted from forward simulation, theta=15 degrees)
    TRUE_PARAMS = {
        'C11': 160.06,  # GPa (1.6006e11 Pa)
        'C22': 70.86,   # GPa (7.086e10 Pa)
        'C12': 48.04,   # GPa (4.804e10 Pa)
        'C16': 9.48,    # GPa (9.48e9 Pa)
        'C26': 16.27,   # GPa (1.627e10 Pa)
        'C66': 37.64    # GPa (3.764e10 Pa)
    }

    print(f"\nConfiguration:")
    print(f"  SECONDARY sampling - Time: {SECONDARY_TIME_STRIDE}, Space: {SECONDARY_SPACE_STRIDE}")
    print(f"  Known density: rho = {RHO_KNOWN} kg/m^3")
    print(f"  Ricker wavelet: f0={SOURCE_F0 / 1e3:.0f} kHz, t0={SOURCE_T0 * 1e6:.0f} us")

    # ============= Load Data =============
    # Modify to your data file path
    h5_file_path = "./data_anisotropic_fem_ricker_fixed_bc/anisotropic_ricker_data_ds5_FEM_fixed_bc.mat"

    try:
        # Noise settings
        ADD_NOISE = True   # Set to False to disable noise
        SNR_DB = 30.0      # Signal-to-noise ratio (dB)

        ux_primary, uy_primary, metadata = load_reference_data(
            h5_file_path,
            add_noise=ADD_NOISE,
            snr_db=SNR_DB
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run the forward simulation code first to generate data.")
        return

    # ============= Extract Grid Parameters =============
    x_original = metadata['x_original']
    y_original = metadata['y_original']
    t_primary = metadata['t_primary']
    source_info = metadata['source_info']
    primary_stride = metadata['primary_stride']

    nx_original = len(x_original)
    ny_original = len(y_original)
    nt_primary = len(t_primary)

    Lx = x_original[-1] - x_original[0]
    Ly = y_original[-1] - y_original[0]
    h_original = Lx / (nx_original - 1)

    dt_primary = t_primary[1] - t_primary[0] if len(t_primary) > 1 else 1e-8
    total_time = t_primary[-1]

    # Reconstruct original time parameters
    nt_original = (nt_primary - 1) * primary_stride + 1
    dt_original = total_time / (nt_original - 1)

    domain_shape_original = (nx_original, ny_original)

    print(f"\n{'=' * 70}")
    print("GRID CONFIGURATION:")
    print(f"{'=' * 70}")
    print(f"ORIGINAL FULL RESOLUTION:")
    print(f"  Spatial: {nx_original}x{ny_original}")
    print(f"  Temporal: {nt_original} steps")
    print(f"  dt: {dt_original * 1e9:.4f} ns")
    print(f"  h: {h_original * 1e3:.4f} mm")
    print(f"\nPRIMARY DOWNSAMPLED (from file):")
    print(f"  Time stride: {primary_stride}")
    print(f"  Loaded time steps: {nt_primary}")
    print(f"  Loaded data shape: {ux_primary.shape}")

    # Extract Ricker parameters from source_info (if available)
    if 'source_f0' in source_info:
        SOURCE_F0 = float(source_info['source_f0'])
    if 'source_t0' in source_info:
        SOURCE_T0 = float(source_info['source_t0'])
    if 'source_amplitude' in source_info:
        SOURCE_AMPLITUDE = float(source_info['source_amplitude'])
    if 'gaussian_param' in source_info:
        GAUSSIAN_PARAM = float(source_info['gaussian_param'])
    if 'rho' in source_info:
        RHO_KNOWN = float(source_info['rho'])

    # Update true parameters (if available in source_info)
    if 'C11' in source_info:
        TRUE_PARAMS['C11'] = float(source_info['C11']) / 1e9
    if 'C22' in source_info:
        TRUE_PARAMS['C22'] = float(source_info['C22']) / 1e9
    if 'C12' in source_info:
        TRUE_PARAMS['C12'] = float(source_info['C12']) / 1e9
    if 'C16' in source_info:
        TRUE_PARAMS['C16'] = float(source_info['C16']) / 1e9
    if 'C26' in source_info:
        TRUE_PARAMS['C26'] = float(source_info['C26']) / 1e9
    if 'C66' in source_info:
        TRUE_PARAMS['C66'] = float(source_info['C66']) / 1e9

    print(f"\nSource parameters from data file:")
    print(f"  f0: {SOURCE_F0 / 1e3:.1f} kHz")
    print(f"  t0: {SOURCE_T0 * 1e6:.1f} us")
    print(f"  Gaussian param: {GAUSSIAN_PARAM:.2e}")
    print(f"  Density: {RHO_KNOWN:.0f} kg/m^3")

    print(f"\nTrue parameters (GPa):")
    for k, v in TRUE_PARAMS.items():
        print(f"  {k}: {v:.2f}")

    # ============= Build Simulator =============
    print(f"\n{'=' * 70}")
    print("BUILDING SIMULATOR (Ricker Wavelet)")
    print(f"{'=' * 70}")

    simulator = CFRPWaveSimulatorRicker(
        domain_shape=domain_shape_original,
        h=h_original,
        dt=dt_original,
        nt=nt_original,
        primary_time_stride=primary_stride,
        source_component='y',
        source_f0=SOURCE_F0,
        source_t0=SOURCE_T0,
        source_amplitude=SOURCE_AMPLITUDE,
        # Initial guess (intentionally different from true values)
        C11_init=90e9,
        C22_init=20e9,
        C12_init=30e9,
        C16_init=20e9,
        C26_init=10e9,
        C66_init=10e9,
        rho_known=RHO_KNOWN,
        gaussian_param=GAUSSIAN_PARAM,
        source_position=SOURCE_POSITION
    ).to(device)

    # Plot Ricker wavelet
    output_dir = "./inversion_results_ricker_adam"
    t_full = np.arange(nt_original) * dt_original
    plot_ricker_comparison(t_full, SOURCE_F0, SOURCE_T0, SOURCE_AMPLITUDE, output_dir)

    # ============= Setup and Run Inversion =============
    print(f"\n{'=' * 70}")
    print("STARTING INVERSION WITH ADAM")
    print(f"{'=' * 70}")

    inverter = CFRPParameterInverterRicker(
        simulator,
        ux_primary,
        uy_primary,
        secondary_time_stride=SECONDARY_TIME_STRIDE,
        secondary_space_stride=SECONDARY_SPACE_STRIDE
    )

    # Adam learning rate is typically between 1e-3 and 1e-4
    results = inverter.invert(max_epochs=800, lr=0.04, true_params=TRUE_PARAMS)

    # ============= Display Results =============
    print("\n" + "=" * 70)
    print("INVERSION COMPLETED")
    print("=" * 70)
    print(f"\nKnown parameter: rho = {results['known']['rho']} kg/m^3")
    print("\nInverted stiffness matrix comparison (GPa):")
    print(f"{'Parameter':<10} {'True':>10} {'Inverted':>10} {'Error %':>10}")
    print("-" * 45)
    for param in ['C11', 'C22', 'C12', 'C16', 'C26', 'C66']:
        true_val = results['true'][param]
        inv_val = results['inverted'][param]
        error = results['error'][param] * 100
        print(f"{param:<10} {true_val:>10.2f} {inv_val:>10.2f} {error:>10.2f}")

    print(f"\nFinal loss: {results['history']['loss'][-1]:.6e}")
    print("=" * 70)

    # Visualize results
    visualize_inversion_results(results, output_dir)

    # Save results
    np.savez(os.path.join(output_dir, 'inverted_parameters_ricker_adam.npz'),
             inverted=results['inverted'],
             true=results['true'],
             errors=results['error'],
             history=results['history'],
             rho=results['known']['rho'],
             source_f0=SOURCE_F0,
             source_t0=SOURCE_T0,
             primary_stride=primary_stride,
             secondary_time_stride=SECONDARY_TIME_STRIDE,
             secondary_space_stride=SECONDARY_SPACE_STRIDE)

    print(f"\nResults saved to {output_dir}/")
    print("\nInversion completed successfully!")


if __name__ == "__main__":
    main()
