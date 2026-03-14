# Elastodynamics-Encoded Recurrent Neural Networks for Characterization of Anisotropic Elastic Constants and Stiffness Degradation

Official code for the paper *"Elastodynamics-Encoded Recurrent Neural Networks for Characterization of Anisotropic Elastic Constants and Stiffness Degradation"*.

## Overview

We propose a physics-encoded recurrent neural network (ERNN) that hard-codes the elastodynamic PDEs directly into the network architecture. The time-recurrent forward model is mathematically equivalent to an RNN whose physics-defined weights correspond to the six independent elastic constants. Training minimizes the waveform mismatch between predicted and observed ultrasonic wavefields, enabling:

- **Elastic constant inversion** — orthotropic & generally anisotropic materials, relative errors < 2%.
- **Stiffness degradation imaging** — spatially resolved damage localization and quantitative assessment.

## Project Structure

```
Elastodynamics-encoded-RNN/
├── README.md
├── requirements.txt
├── Parameter_inversion/
│   ├── FEM_generate_data.py      # FEM forward simulation
│   └── Inversion_adam.py         # ERNN elastic constant inversion
└── Damage_inversion/
    ├── FEM_generate_data.py      # FEM forward simulation with damage
    └── Inversion_adam.py         # ERNN stiffness degradation imaging
```

## Installation

**Requirements:** Python 3.8+, NVIDIA GPU (≥ 8 GB VRAM recommended)

```bash
conda create -n ernn python=3.10 -y
conda activate ernn

# PyTorch — choose your CUDA version at https://pytorch.org
pip install torch --index-url https://download.pytorch.org/whl/cu118

pip install numpy scipy matplotlib h5py
```

## Quick Start

### Parameter Inversion

```bash
cd Parameter_inversion
python FEM_generate_data.py    # Step 1: generate synthetic data
python Inversion_adam.py       # Step 2: ERNN inversion
```

### Damage Inversion

```bash
cd Damage_inversion
python FEM_generate_data.py    # Step 1: generate data with damage
python Inversion_adam.py       # Step 2: ERNN inversion
```

## License

For academic research only.
