"""
Antiferromagnetic Ising Model Experiment
=========================================

Demonstrates antiferromagnetic behavior (J < 0) where spins tend to anti-align,
forming a checkerboard pattern.
"""

import os
import torch
import numpy as np

from model import IsingMachine


def run_antiferromagnetic_experiment(
    lattice_size: int = 64,
    J_strength: float = -1.0,
    T_initial: float = 5.0,
    T_final: float = 0.01,
    n_steps: int = 2000,
    output_dir: str = "outputs/antiferromagnetic"
) -> None:
    """
    Run antiferromagnetic Ising model experiment
    
    Args:
        lattice_size: Size of the square lattice
        J_strength: Antiferromagnetic coupling strength (negative)
        T_initial: Initial temperature
        T_final: Final temperature
        n_steps: Number of annealing steps
        output_dir: Directory to save outputs
    """
    print("=" * 70)
    print("ANTIFERROMAGNETIC ISING MODEL EXPERIMENT")
    print("=" * 70)
    print(f"Lattice: {lattice_size} × {lattice_size}")
    print(f"Coupling J: {J_strength:.3f} (antiferromagnetic)")
    print(f"Temperature: {T_initial:.3f} → {T_final:.3f}")
    print(f"Steps: {n_steps}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Ising machine
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # cpu works as well as gpu
    print(f"Using device: {device}\n")
    
    model = IsingMachine(
        shape=(lattice_size, lattice_size),
        J_strength=J_strength,
        h_strength=None,
        device=device
    )
    
    print(f"Initial energy: {model.compute_energy().item():.2f}")
    print(f"Initial magnetization: {model.compute_magnetization():.3f}")
    
    # Run annealing
    print("\nRunning simulated annealing...")
    energy_history, temp_history = model.anneal(
        T_initial=T_initial,
        T_final=T_final,
        n_steps=n_steps,
        record_interval=100
    )
    
    final_energy = model.compute_energy().item()
    final_mag = model.compute_magnetization()
    
    print(f"\nFinal energy: {final_energy:.2f}")
    print(f"Final magnetization: {final_mag:.3f}")
    
    # Save final state image
    output_path = os.path.join(output_dir, "antiferromagnetic_final.png")
    model.visualize(title="Antiferromagnetic Final State (J < 0)", save_path=output_path)
    print(f"\nImage saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print("\nExpected behavior:")
    print("  - Spins form checkerboard pattern")
    print("  - Magnetization close to zero (M ≈ 0)")
    print("  - Energy minimum with perfect alternation")
    print()


if __name__ == "__main__":
    run_antiferromagnetic_experiment(
        lattice_size=64,
        J_strength=-1.0,
        T_initial=5.0,
        T_final=0.01,
        n_steps=2000,
        output_dir="outputs/antiferromagnetic"
    )
