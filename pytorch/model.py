"""
PyTorch Ising Machine Model
===========================

A PyTorch implementation of the Ising model with simulated annealing.
Supports N-dimensional lattices with and without external fields.

Physics:
    Energy: E = -Σ J_ij s_i s_j - Σ h_i s_i
    where s_i ∈ {-1, +1} are spins
    J_ij: coupling between spins (interaction strength)
    h_i: external magnetic field (bias on each spin)
"""

import torch
import numpy as np
from typing import Tuple, Optional, List, Union
import matplotlib.pyplot as plt


class IsingMachine:
    """
    Ising Machine with Simulated Annealing
    
    Attributes:
        shape (Tuple[int, ...]): Lattice dimensions (e.g., (32, 32) for 2D)
        J (torch.Tensor): Coupling matrix (n_spins × n_spins)
        h (Optional[torch.Tensor]): External field (same shape as lattice)
        device (str): 'cuda' or 'cpu'
        spins (torch.Tensor): Current spin configuration
    """
    
    def __init__(self, shape: Tuple[int, ...], J_strength: float = 1.0, 
                 h_strength: Optional[float] = None,
                 h_field: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 device: str = 'cuda'):
        """
        Initialize Ising Machine
        
        Args:
            shape: Lattice dimensions (e.g., (32, 32) for 2D, (10, 10, 10) for 3D)
            J_strength: Coupling strength (J > 0 = ferromagnetic, J < 0 = antiferromagnetic)
            h_strength: Uniform external field strength (scalar, for convenience)
            h_field: Custom N-D external field (numpy array or torch tensor)
                     If both h_strength and h_field are provided, h_field takes precedence
            device: 'cuda', 'mps', or 'cpu'
        """
        self.shape = shape
        if torch.cuda.is_available():
            self.device = 'cuda'
        # elif torch.backends.mps.is_available():
        #     self.device = 'mps'
        else:
            self.device = 'cpu'
        self.n_spins = int(np.prod(shape))
        
        # Create nearest-neighbor coupling matrix
        self.J = self._create_nearest_neighbor_couplings(J_strength)
        
        # Create external field if specified
        if h_field is not None:
            # Custom field provided (takes precedence over h_strength)
            if isinstance(h_field, np.ndarray):
                if h_field.shape != shape:
                    raise ValueError(f"h_field shape {h_field.shape} must match lattice shape {shape}")
                self.h = torch.from_numpy(h_field).float().to(self.device)
            elif isinstance(h_field, torch.Tensor):
                if tuple(h_field.shape) != shape:
                    raise ValueError(f"h_field shape {tuple(h_field.shape)} must match lattice shape {shape}")
                self.h = h_field.float().to(self.device)
            else:
                raise TypeError(f"h_field must be numpy array or torch tensor, got {type(h_field)}")
        elif h_strength is not None:
            # Uniform field from scalar
            self.h = torch.full(shape, h_strength, device=self.device)
        else:
            # No field
            self.h = None
        
        # Initialize spins randomly
        self.spins = self._initialize_spins()
        
    def _initialize_spins(self) -> torch.Tensor:
        """Initialize random spin configuration with values {-1, +1}"""
        spins = torch.randint(0, 2, self.shape, device=self.device) * 2 - 1
        return spins.float()
    
    def _create_nearest_neighbor_couplings(self, J: float) -> torch.Tensor:
        """
        Create coupling matrix for nearest-neighbor interactions
        
        Args:
            J: Coupling strength
            
        Returns:
            Coupling matrix (n_spins × n_spins)
        """
        couplings = torch.zeros((self.n_spins, self.n_spins), device=self.device)
        
        # Create neighbor connectivity for N-D lattice
        indices = torch.arange(self.n_spins, device=self.device).reshape(self.shape)
        
        # Connect along each dimension (periodic boundary conditions)
        for dim in range(len(self.shape)):
            neighbor_indices = torch.roll(indices, shifts=-1, dims=dim)
            
            flat_indices = indices.flatten()
            flat_neighbors = neighbor_indices.flatten()
            
            # Set symmetric couplings
            couplings[flat_indices, flat_neighbors] = J
            couplings[flat_neighbors, flat_indices] = J
        
        return couplings
    
    def compute_energy(self) -> torch.Tensor:
        """
        Compute total Ising energy: E = -Σ J_ij s_i s_j - Σ h_i s_i
        
        Returns:
            Scalar energy value
        """
        flat_spins = self.spins.flatten()
        
        # Interaction energy: -Σ J_ij s_i s_j (divide by 2 to avoid double-counting)
        interaction_energy = -0.5 * torch.sum(self.J * torch.outer(flat_spins, flat_spins))
        
        # External field energy: -Σ h_i s_i
        field_energy = -torch.sum(self.h * self.spins) if self.h is not None else 0.0
        
        return interaction_energy + field_energy
    
    def compute_magnetization(self) -> float:
        """
        Compute average magnetization (mean spin value)
        
        Returns:
            Magnetization in range [-1, 1]
        """
        return torch.mean(self.spins).item()
    
    def _compute_local_field(self, i: int) -> torch.Tensor:
        """
        Compute local field acting on spin i: h_eff = Σ J_ij s_j + h_i
        
        Args:
            i: Index of spin in flattened configuration
            
        Returns:
            Effective field at spin i
        """
        flat_spins = self.spins.flatten()
        
        # Interaction contribution
        interaction_field = torch.sum(self.J[i] * flat_spins)
        
        # External field contribution
        field_contribution = self.h.flatten()[i] if self.h is not None else 0.0
        
        return interaction_field + field_contribution
    
    def _flip_energy_change(self, i: int) -> torch.Tensor:
        """
        Compute energy change if spin i is flipped: ΔE = 2 s_i h_eff
        
        Args:
            i: Index of spin to flip
            
        Returns:
            Energy change ΔE
        """
        flat_spins = self.spins.flatten()
        h_eff = self._compute_local_field(i)
        return 2.0 * flat_spins[i] * h_eff
    
    def metropolis_step(self, temperature: float) -> None:
        """
        Perform one Metropolis Monte Carlo sweep
        
        Attempts to flip each spin once using Metropolis criterion:
            - Accept flip if ΔE < 0 (energy decreases)
            - Accept flip with probability exp(-ΔE/T) if ΔE > 0
        
        Args:
            temperature: Current temperature T
        """
        flat_spins = self.spins.flatten()
        
        # Shuffle order to avoid bias
        indices = torch.randperm(self.n_spins, device=self.device)
        
        for idx in indices:
            # Compute energy change for flipping this spin
            delta_E = self._flip_energy_change(idx)
            
            # Metropolis acceptance criterion
            if delta_E < 0:
                # Always accept energy-lowering moves
                flat_spins[idx] *= -1
            else:
                # Accept energy-raising moves probabilistically
                acceptance_prob = torch.exp(-delta_E / temperature)
                if torch.rand(1, device=self.device) < acceptance_prob:
                    flat_spins[idx] *= -1
        
        # Update spins
        self.spins = flat_spins.reshape(self.shape)
    
    def anneal(self, T_initial: float = 10.0, T_final: float = 0.01,
               n_steps: int = 5000, record_interval: int = 100) -> Tuple[List[float], List[float]]:
        """
        Perform simulated annealing
        
        Temperature decreases exponentially: T(t) = T_initial * (T_final/T_initial)^(t/n_steps)
        
        Args:
            T_initial: Starting temperature (high for exploration)
            T_final: Final temperature (low for refinement)
            n_steps: Number of annealing steps
            record_interval: How often to record energy (for history)
            
        Returns:
            Tuple of (energy_history, temperature_history)
        """
        energy_history = []
        temperature_history = []
        
        for step in range(n_steps):
            # Exponential cooling schedule
            progress = step / n_steps
            T = T_initial * (T_final / T_initial) ** progress
            
            # Perform Metropolis sweep
            self.metropolis_step(T)
            
            # Record energy periodically
            if step % record_interval == 0:
                energy = self.compute_energy()
                energy_history.append(energy.item())
                temperature_history.append(T)
        
        return energy_history, temperature_history
    
    def visualize(self, title: str = "Spin Configuration", 
                  save_path: Optional[str] = None) -> None:
        """
        Visualize 2D spin configuration
        
        Args:
            title: Plot title
            save_path: Path to save figure (if None, just shows)
        """
        if len(self.shape) != 2:
            print(f"Visualization only supports 2D lattices, got shape {self.shape}")
            return
        
        spins_np = self.spins.cpu().numpy()
        
        plt.figure(figsize=(8, 8))
        plt.imshow(spins_np, cmap='RdBu', vmin=-1, vmax=1, interpolation='nearest')
        plt.colorbar(label='Spin', ticks=[-1, 1])
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.close()
    
    def get_spins(self) -> np.ndarray:
        """Get current spin configuration as numpy array"""
        return self.spins.cpu().numpy()
    
    def reset(self) -> None:
        """Reset spins to random configuration"""
        self.spins = self._initialize_spins()
    
    def __str__(self) -> str:
        """String representation of the Ising machine"""
        J_type = "ferromagnetic" if self.J[0, 1].item() > 0 else "antiferromagnetic"
        if self.h is not None:
            h_min = self.h.min().item()
            h_max = self.h.max().item()
            if h_min == h_max:
                h_info = f", h={h_min:.3f}"
            else:
                h_info = f", h=[{h_min:.3f}, {h_max:.3f}]"
        else:
            h_info = ", no field"
        return (f"IsingMachine(shape={self.shape}, "
                f"J={self.J[0, 1].item():.3f} ({J_type}){h_info}, "
                f"device={self.device})")

