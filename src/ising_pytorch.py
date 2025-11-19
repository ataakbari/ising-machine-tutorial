"""
PyTorch Implementation of Ising Machine with Simulated Annealing
Supports N-dimensional lattices with and without external field

Physics:
    Energy: E = -Σ J_ij s_i s_j - Σ h_i s_i
    where s_i ∈ {-1, +1} are spins
    J_ij: coupling between spins (interaction strength)
    h_i: external magnetic field (bias on each spin)
"""

import torch
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt


def initialize_spins(shape: Tuple[int, ...], device: str = 'cuda') -> torch.Tensor:
    """
    Initialize random spin configuration
    
    Args:
        shape: Lattice dimensions (e.g., (10, 10) for 2D, (5, 5, 5) for 3D)
        device: 'cuda' or 'cpu'
    
    Returns:
        Tensor of {-1, +1} spins
    """
    spins = torch.randint(0, 2, shape, device=device) * 2 - 1
    return spins.float()


def create_nearest_neighbor_couplings(shape: Tuple[int, ...], J: float = 1.0, 
                                      device: str = 'cuda') -> torch.Tensor:
    """
    Create coupling matrix for nearest-neighbor interactions on lattice
    
    For a lattice, only adjacent spins interact with strength J
    
    Args:
        shape: Lattice dimensions
        J: Coupling strength (J > 0 = ferromagnetic, J < 0 = antiferromagnetic)
        device: 'cuda' or 'cpu'
    
    Returns:
        Flattened coupling matrix (n_spins × n_spins)
    """
    n_spins = int(np.prod(shape))
    couplings = torch.zeros((n_spins, n_spins), device=device)
    
    # For N-D lattice, create neighbor connectivity
    indices = torch.arange(n_spins, device=device).reshape(shape)
    
    # Connect along each dimension
    for dim in range(len(shape)):
        # Shift by 1 in this dimension to get neighbors
        neighbor_indices = torch.roll(indices, shifts=-1, dims=dim)
        
        flat_indices = indices.flatten()
        flat_neighbors = neighbor_indices.flatten()
        
        # Set symmetric couplings
        couplings[flat_indices, flat_neighbors] = J
        couplings[flat_neighbors, flat_indices] = J
    
    return couplings


def compute_energy(spins: torch.Tensor, J: torch.Tensor, 
                   h: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute total Ising energy: E = -Σ J_ij s_i s_j - Σ h_i s_i
    
    Args:
        spins: Current spin configuration (any shape)
        J: Coupling matrix (n_spins × n_spins) where n_spins = prod(spins.shape)
        h: External field (same shape as spins), optional
    
    Returns:
        Scalar energy value
    """
    # Flatten spins to 1D for matrix operations
    flat_spins = spins.flatten()
    
    # Interaction energy: -Σ J_ij s_i s_j
    # Note: divide by 2 to avoid double-counting (J is symmetric)
    interaction_energy = -0.5 * torch.sum(J * torch.outer(flat_spins, flat_spins))
    
    # External field energy: -Σ h_i s_i
    if h is not None:
        field_energy = -torch.sum(h * spins)
    else:
        field_energy = 0.0
    
    return interaction_energy + field_energy


def compute_local_field(i: int, spins: torch.Tensor, J: torch.Tensor, 
                        h: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute local field acting on spin i: h_eff = Σ J_ij s_j + h_i
    
    This is used to efficiently compute energy change when flipping spin i
    
    Args:
        i: Index of spin in flattened configuration
        spins: Current spin configuration (any shape)
        J: Coupling matrix
        h: External field (same shape as spins), optional
    
    Returns:
        Effective field at spin i
    """
    flat_spins = spins.flatten()
    
    # Interaction contribution: Σ J_ij s_j
    interaction_field = torch.sum(J[i] * flat_spins)
    
    # External field contribution
    if h is not None:
        flat_h = h.flatten()
        field_contribution = flat_h[i]
    else:
        field_contribution = 0.0
    
    return interaction_field + field_contribution


def flip_energy_change(i: int, spins: torch.Tensor, J: torch.Tensor,
                       h: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute energy change if spin i is flipped: ΔE = 2 s_i h_eff
    
    This is much faster than recomputing the full energy
    
    Args:
        i: Index of spin to flip (in flattened configuration)
        spins: Current spin configuration
        J: Coupling matrix
        h: External field, optional
    
    Returns:
        Energy change ΔE
    """
    flat_spins = spins.flatten()
    h_eff = compute_local_field(i, spins, J, h)
    delta_E = 2.0 * flat_spins[i] * h_eff
    
    return delta_E


def metropolis_step(spins: torch.Tensor, J: torch.Tensor, 
                   h: Optional[torch.Tensor], temperature: float) -> torch.Tensor:
    """
    Perform one Metropolis Monte Carlo sweep (attempt to flip each spin once)
    
    Metropolis criterion:
        - Accept flip if ΔE < 0 (energy decreases)
        - Accept flip with probability exp(-ΔE/T) if ΔE > 0
    
    Args:
        spins: Current spin configuration
        J: Coupling matrix
        h: External field, optional
        temperature: Current temperature T
    
    Returns:
        Updated spin configuration
    """
    flat_spins = spins.flatten()
    n_spins = flat_spins.shape[0]
    
    # Shuffle order to avoid bias
    indices = torch.randperm(n_spins, device=spins.device)
    
    for idx in indices:
        # Compute energy change if we flip this spin
        delta_E = flip_energy_change(idx, spins, J, h)
        
        # Metropolis acceptance criterion
        if delta_E < 0:
            # Always accept energy-lowering moves
            flat_spins[idx] *= -1
        else:
            # Accept energy-raising moves probabilistically
            acceptance_prob = torch.exp(-delta_E / temperature)
            if torch.rand(1, device=spins.device) < acceptance_prob:
                flat_spins[idx] *= -1
    
    return flat_spins.reshape(spins.shape)


def simulated_annealing(J: torch.Tensor, h: Optional[torch.Tensor], 
                       shape: Tuple[int, ...],
                       T_initial: float = 10.0, T_final: float = 0.01,
                       n_steps: int = 10000, device: str = 'cuda') -> Tuple[torch.Tensor, list]:
    """
    Perform simulated annealing to find low-energy spin configuration
    
    Temperature is exponentially decreased: T(t) = T_initial * (T_final/T_initial)^(t/n_steps)
    
    Args:
        J: Coupling matrix (n_spins × n_spins)
        h: External field (shape of lattice), optional
        shape: Lattice dimensions
        T_initial: Starting temperature (high, allows exploration)
        T_final: Final temperature (low, refines solution)
        n_steps: Number of annealing steps
        device: 'cuda' or 'cpu'
    
    Returns:
        Final spin configuration and energy history
    """
    # Initialize spins randomly
    spins = initialize_spins(shape, device=device)
    
    # Track energy over time
    energy_history = []
    
    # Annealing schedule (exponential cooling)
    for step in range(n_steps):
        # Current temperature
        progress = step / n_steps
        T = T_initial * (T_final / T_initial) ** progress
        
        # Perform Metropolis sweep
        spins = metropolis_step(spins, J, h, T)
        
        # Record energy
        if step % 100 == 0:
            energy = compute_energy(spins, J, h)
            energy_history.append(energy.item())
    
    return spins, energy_history


def visualize_spins(spins: torch.Tensor, title: str = "Spin Configuration"):
    """
    Visualize 2D spin configuration
    
    Args:
        spins: 2D tensor of spins {-1, +1}
        title: Plot title
    """
    if len(spins.shape) != 2:
        print(f"Visualization only supports 2D lattices, got shape {spins.shape}")
        return
    
    spins_np = spins.cpu().numpy()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(spins_np, cmap='RdBu', vmin=-1, vmax=1, interpolation='nearest')
    plt.colorbar(label='Spin', ticks=[-1, 1])
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig('spin_configuration.png', dpi=150)
    plt.show()


def run_ferromagnetic_2d_example():
    """
    Example: 2D ferromagnetic Ising model without external field
    
    Spins want to align (J > 0), should form large domains
    """
    print("=" * 60)
    print("2D Ferromagnetic Ising Model (No External Field)")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 2D lattice
    shape = (32, 32)
    J_strength = 1.0  # Ferromagnetic coupling
    
    # Create nearest-neighbor couplings
    J = create_nearest_neighbor_couplings(shape, J=J_strength, device=device)
    h = None  # No external field
    
    print(f"Lattice shape: {shape}")
    print(f"Total spins: {np.prod(shape)}")
    print(f"Coupling strength J: {J_strength} (ferromagnetic)")
    
    # Run simulated annealing
    print("\nRunning simulated annealing...")
    spins, energy_history = simulated_annealing(
        J, h, shape, 
        T_initial=5.0, T_final=0.01, 
        n_steps=5000, device=device
    )
    
    final_energy = compute_energy(spins, J, h)
    print(f"Final energy: {final_energy.item():.2f}")
    
    # Visualize
    visualize_spins(spins, "Ferromagnetic Ising Model (J > 0)")
    
    return spins, energy_history


def run_antiferromagnetic_2d_example():
    """
    Example: 2D antiferromagnetic Ising model without external field
    
    Spins want to anti-align (J < 0), should form checkerboard pattern
    """
    print("=" * 60)
    print("2D Antiferromagnetic Ising Model (No External Field)")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 2D lattice
    shape = (32, 32)
    J_strength = -1.0  # Antiferromagnetic coupling
    
    # Create nearest-neighbor couplings
    J = create_nearest_neighbor_couplings(shape, J=J_strength, device=device)
    h = None  # No external field
    
    print(f"Lattice shape: {shape}")
    print(f"Total spins: {np.prod(shape)}")
    print(f"Coupling strength J: {J_strength} (antiferromagnetic)")
    
    # Run simulated annealing
    print("\nRunning simulated annealing...")
    spins, energy_history = simulated_annealing(
        J, h, shape,
        T_initial=5.0, T_final=0.01,
        n_steps=5000, device=device
    )
    
    final_energy = compute_energy(spins, J, h)
    print(f"Final energy: {final_energy.item():.2f}")
    
    # Visualize
    visualize_spins(spins, "Antiferromagnetic Ising Model (J < 0)")
    
    return spins, energy_history


def run_with_external_field_example():
    """
    Example: 2D ferromagnetic Ising model WITH external field
    
    Field biases spins up (+1), competing with thermal fluctuations
    """
    print("=" * 60)
    print("2D Ising Model WITH External Field")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 2D lattice
    shape = (32, 32)
    J_strength = 1.0  # Ferromagnetic
    h_strength = 0.5  # External field pointing up
    
    # Create couplings and external field
    J = create_nearest_neighbor_couplings(shape, J=J_strength, device=device)
    h = torch.full(shape, h_strength, device=device)  # Uniform field
    
    print(f"Lattice shape: {shape}")
    print(f"Coupling strength J: {J_strength}")
    print(f"External field h: {h_strength} (biases spins up)")
    
    # Run simulated annealing
    print("\nRunning simulated annealing...")
    spins, energy_history = simulated_annealing(
        J, h, shape,
        T_initial=5.0, T_final=0.01,
        n_steps=5000, device=device
    )
    
    final_energy = compute_energy(spins, J, h)
    magnetization = torch.mean(spins)
    print(f"Final energy: {final_energy.item():.2f}")
    print(f"Magnetization (avg spin): {magnetization.item():.3f}")
    print(f"  (ranges from -1 (all down) to +1 (all up))")
    
    # Visualize
    visualize_spins(spins, f"Ising Model with External Field h={h_strength}")
    
    return spins, energy_history


if __name__ == "__main__":
    # Run examples
    print("\n" + "=" * 60)
    print("PyTorch Ising Machine Examples")
    print("=" * 60 + "\n")
    
    # Example 1: Ferromagnetic (spins align)
    spins1, energy1 = run_ferromagnetic_2d_example()
    
    print("\n\n")
    
    # Example 2: Antiferromagnetic (spins anti-align)
    spins2, energy2 = run_antiferromagnetic_2d_example()
    
    print("\n\n")
    
    # Example 3: With external field
    spins3, energy3 = run_with_external_field_example()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)

