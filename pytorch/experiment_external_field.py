"""
External Field Ising Model Experiment
======================================

Demonstrates the effect of external magnetic field on spin alignment.
Features binary field patterns: maze and Fibonacci spiral.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import IsingMachine
from typing import Union, Optional


def create_maze_pattern(size: int, field_strength: float = 0.5) -> np.ndarray:
    """
    Create a binary maze pattern using recursive backtracking
    
    Args:
        size: Lattice size (adjusted to odd number for maze generation)
        field_strength: Field strength for walls
        
    Returns:
        Binary field: walls (+field_strength), paths (-field_strength)
    """
    maze_size = size if size % 2 == 1 else size - 1
    maze = np.ones((maze_size, maze_size), dtype=int)
    
    directions = [(-2, 0), (0, 2), (2, 0), (0, -2)]
    
    def carve_passages(cx, cy):
        dirs = directions.copy()
        np.random.shuffle(dirs)
        
        for dx, dy in dirs:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < maze_size and 0 <= ny < maze_size and maze[ny, nx] == 1:
                maze[cy + dy//2, cx + dx//2] = 0
                maze[ny, nx] = 0
                carve_passages(nx, ny)
    
    maze[1, 1] = 0
    carve_passages(1, 1)
    
    h_maze = np.where(maze == 0, -field_strength, field_strength)
    
    if size != maze_size:
        h_maze = np.pad(h_maze, ((0, 1), (0, 1)), mode='edge')
    
    return h_maze


def create_fibonacci_spiral_pattern(size: int, field_strength: float = 0.5) -> np.ndarray:
    """
    Create a binary Fibonacci spiral pattern using golden angle
    
    Args:
        size: Lattice size
        field_strength: Field strength magnitude
        
    Returns:
        Binary field based on Fibonacci spiral
    """
    center = size / 2
    y, x = np.ogrid[:size, :size]
    
    x_centered = x - center
    y_centered = y - center
    r = np.sqrt(x_centered**2 + y_centered**2)
    theta = np.arctan2(y_centered, x_centered)
    
    num_arms = 8
    spiral_value = (theta + np.pi) / (2 * np.pi)
    spiral_value = (spiral_value + r / (size/2) * num_arms) % 1
    
    h_spiral = np.where(spiral_value < 0.5, field_strength, -field_strength)
    
    # Add Fibonacci radial bands
    fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34]
    fib_radii = np.array(fib_sequence) * (size / 80)
    
    for i, radius in enumerate(fib_radii):
        if radius < size:
            mask = (r > radius - 1) & (r < radius + 1)
            h_spiral[mask] = field_strength if i % 2 == 0 else -field_strength
    
    return h_spiral


def visualize_field_pattern(field: np.ndarray, save_path: str) -> None:
    """Visualize the 2D external field pattern"""
    plt.figure(figsize=(8, 7))
    vmax = max(abs(field.min()), abs(field.max()))
    im = plt.imshow(field, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
    plt.colorbar(im, label='Field Strength h(x,y)')
    plt.title('External Magnetic Field Pattern')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def run_external_field_experiment(
    lattice_size: int = 64,
    J_strength: float = 0.01,
    h_field: Union[float, np.ndarray, torch.Tensor] = 0.5,
    T_initial: float = 5.0,
    T_final: float = 0.1,
    n_steps: int = 2000,
    output_dir: str = "outputs/external_field"
) -> None:
    """
    Run Ising model with external field experiment
    
    Args:
        lattice_size: Size of the square lattice
        J_strength: Ferromagnetic coupling strength
        h_field: External field (float, numpy array, or torch tensor)
        T_initial: Initial temperature
        T_final: Final temperature
        n_steps: Number of annealing steps
        output_dir: Directory to save outputs
    """
    print("=" * 70)
    print("EXTERNAL FIELD ISING MODEL EXPERIMENT")
    print("=" * 70)
    
    # Determine field type
    is_uniform = isinstance(h_field, (int, float))
    
    if is_uniform:
        field_desc = f"Uniform h={h_field:.3f}"
    else:
        if isinstance(h_field, np.ndarray):
            h_array = h_field
        elif isinstance(h_field, torch.Tensor):
            h_array = h_field.cpu().numpy() if h_field.is_cuda else h_field.numpy()
        else:
            raise TypeError(f"h_field must be float, numpy array, or torch tensor")
        field_desc = f"2D field (min={h_array.min():.3f}, max={h_array.max():.3f})"
    
    print(f"Lattice: {lattice_size} × {lattice_size}")
    print(f"Coupling J: +{J_strength:.3f}")
    print(f"External field: {field_desc}")
    print(f"Temperature: {T_initial:.3f} → {T_final:.3f}")
    print(f"Steps: {n_steps}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Ising machine (cpu works as well as gpu)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Using device: {device}\n")
    
    # Create model
    if is_uniform:
        model = IsingMachine(
            shape=(lattice_size, lattice_size),
            J_strength=J_strength,
            h_strength=h_field,
            device=device
        )
    else:
        model = IsingMachine(
            shape=(lattice_size, lattice_size),
            J_strength=J_strength,
            h_field=h_field,
            device=device
        )
    
    # Visualize field if 2D
    if not is_uniform:
        field_viz_path = os.path.join(output_dir, "field_pattern.png")
        visualize_field_pattern(model.h.cpu().numpy(), field_viz_path)
        print(f"Field pattern saved to: {field_viz_path}")
    
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
    output_path = os.path.join(output_dir, "external_field_final.png")
    model.visualize(title="External Field Final State", save_path=output_path)
    print(f"Image saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    if is_uniform:
        print("\nExpected: Spins aligned with field direction")
    else:
        print("\nExpected: Spins follow the field pattern")
    print()


if __name__ == "__main__":
    size = 64
    
    # Experiment 1: Maze pattern
    print("\n" + "="*70)
    print("MAZE PATTERN")
    print("="*70)
    h_maze = create_maze_pattern(size, field_strength=0.5)
    
    run_external_field_experiment(
        lattice_size=size,
        J_strength=0.01,
        h_field=h_maze,
        T_initial=5.0,
        T_final=0.1,
        n_steps=2000,
        output_dir="outputs/external_field_maze"
    )
    
    # Experiment 2: Fibonacci spiral
    print("\n" + "="*70)
    print("FIBONACCI SPIRAL PATTERN")
    print("="*70)
    h_fibonacci = create_fibonacci_spiral_pattern(size, field_strength=0.5)
    
    run_external_field_experiment(
        lattice_size=size,
        J_strength=0.01,
        h_field=h_fibonacci,
        T_initial=5.0,
        T_final=0.1,
        n_steps=2000,
        output_dir="outputs/external_field_fibonacci"
    )
    
    print("\n" + "="*70)
    print("ALL BINARY PATTERN EXPERIMENTS COMPLETE")
    print("="*70)
    print("\nOutputs saved to:")
    print("  - outputs/external_field_maze/")
    print("  - outputs/external_field_fibonacci/")
    print()
