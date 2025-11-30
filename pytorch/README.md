# PyTorch Ising Machine

PyTorch implementation of the Ising model with simulated annealing.

## Features

- **Object-Oriented Design**: Clean `IsingMachine` class for easy experimentation
- **N-Dimensional Lattices**: Support for 1D, 2D, 3D, and higher dimensions
- **External Fields**: Optional magnetic field bias (uniform or 2D patterns)
- **Binary Field Patterns**: Maze and Fibonacci spiral patterns
- **GPU Acceleration**: Automatic CUDA support when available

## Files

- `model.py` - Core `IsingMachine` class implementation
- `experiment_ferromagnetic.py` - Ferromagnetic behavior (J > 0)
- `experiment_antiferromagnetic.py` - Antiferromagnetic behavior (J < 0)
- `experiment_external_field.py` - External field effects (h ≠ 0)
- `run_all_experiments.py` - Convenience script to run all experiments

## Quick Start

### Run All Experiments

```bash
python run_all_experiments.py
```

This generates final state images in the `outputs/` directory:

- `outputs/ferromagnetic/`
- `outputs/antiferromagnetic/`
- `outputs/external_field/`

### Run Individual Experiments

```bash
# Ferromagnetic (spins align)
python experiment_ferromagnetic.py

# Antiferromagnetic (checkerboard pattern)
python experiment_antiferromagnetic.py

# External field (biased spins)
python experiment_external_field.py
```

### Use the Model Directly

```python
from model import IsingMachine
import numpy as np

# Example 1: Ferromagnetic system with no field
model = IsingMachine(
    shape=(64, 64),
    J_strength=1.0,      # Positive = ferromagnetic
    h_strength=None,     # No external field
    device='cuda'
)

# Example 2: Uniform external field
model = IsingMachine(
    shape=(64, 64),
    J_strength=1.0,
    h_strength=0.5,      # Uniform field (all spins feel same field)
    device='cuda'
)

# Example 3: Binary maze pattern
from experiment_external_field import create_maze_pattern
h_maze = create_maze_pattern(64, field_strength=0.5)

model = IsingMachine(
    shape=(64, 64),
    J_strength=1.0,
    h_field=h_maze,  # Binary maze pattern
    device='cuda'
)

# Example 4: Binary Fibonacci spiral
from experiment_external_field import create_fibonacci_spiral_pattern
h_fib = create_fibonacci_spiral_pattern(64, field_strength=0.5)

model = IsingMachine(
    shape=(64, 64),
    J_strength=1.0,
    h_field=h_fib,  # Binary Fibonacci spiral
    device='cuda'
)

# Run simulated annealing
energy_history, temp_history = model.anneal(
    T_initial=5.0,
    T_final=0.01,
    n_steps=2000
)

# Check final state
print(f"Energy: {model.compute_energy():.2f}")
print(f"Magnetization: {model.compute_magnetization():.3f}")

# Visualize
model.visualize(title="My Ising Model", save_path="my_result.png")

# Access spin configuration
spins = model.get_spins()  # Returns numpy array
```

## Model Parameters

### Coupling Strength (J)

- **J > 0**: Ferromagnetic - spins prefer to align
- **J < 0**: Antiferromagnetic - spins prefer to anti-align
- **|J|**: Controls interaction strength

### External Field (h)

The external field can be specified in two ways:

**Option 1: Uniform field (h_strength)**

- **h > 0**: Biases spins upward (+1)
- **h < 0**: Biases spins downward (-1)
- **None**: No bias

**Option 2: Spatially varying field (h_field)**

- **2D numpy array**: Custom field pattern
- **h_field[i,j]**: Field strength at position (i,j)
- Enables gradients, localized fields, patterns, etc.

```python
# Uniform field
model = IsingMachine(shape=(64, 64), J_strength=1.0, h_strength=0.5)

# Gradient field
h_grad = np.linspace(-0.5, 0.5, 64)
h_2d = np.tile(h_grad, (64, 1))
model = IsingMachine(shape=(64, 64), J_strength=1.0, h_field=h_2d)

# Gaussian bump
center = 32
y, x = np.ogrid[:64, :64]
r2 = (x - center)**2 + (y - center)**2
h_gauss = np.exp(-r2 / (2 * 16**2))
model = IsingMachine(shape=(64, 64), J_strength=1.0, h_field=h_gauss)
```

### Temperature Schedule

- **T_initial**: High temperature for exploration (typical: 5.0)
- **T_final**: Low temperature for refinement (typical: 0.01)
- **n_steps**: Number of annealing steps (typical: 2000-5000)

## Video Generation

Each experiment creates an MP4 video showing:

1. **Left panel**: Spin configuration evolving over time
2. **Middle panel**: Energy decreasing (blue line)
3. **Right panel**: Magnetization evolution (red line)

Videos use FFMpeg. Make sure it's installed:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/
```

## Customization

You can modify experiment parameters in each script:

```python
run_ferromagnetic_with_recording(
    lattice_size=64,      # Size of lattice (N×N)
    J_strength=1.0,       # Coupling strength
    T_initial=5.0,        # Starting temperature
    T_final=0.01,         # Final temperature
    n_steps=2000,         # Number of annealing steps
    record_every=20,      # Frame recording frequency
    output_dir="outputs/ferromagnetic"
)
```

## Expected Behaviors

### Ferromagnetic (J > 0)

- Large domains of same-spin regions
- High magnetization (|M| ≈ 1)
- Few domain boundaries in final state

### Antiferromagnetic (J < 0)

- Checkerboard pattern emerges
- Near-zero magnetization (M ≈ 0)
- Perfect alternating pattern at low temperature

### External Field (h ≠ 0)

- Spins align with field direction
- Magnetization biased toward field sign
- Stronger field → stronger bias

## Performance Tips

1. **Use GPU**: Automatic if CUDA available, ~10× faster
2. **Reduce lattice size**: Start with 32×32 for quick tests
3. **Fewer steps**: 500-1000 steps for quick previews
4. **Record less often**: Increase `record_every` for smaller videos

## Troubleshooting

### "No module named 'model'"

Make sure you're in the `pytorch/` directory or adjust imports:

```python
from pytorch.model import IsingMachine
```

### "FFMpegWriter not available"

Install FFMpeg (see Video Generation section above)

### "CUDA out of memory"

Reduce `lattice_size` or switch to CPU:

```python
device='cpu'
```

## Learn More

See the main [README](../README.md) for:

- Physics background
- Comparison with CUDA implementation
- Application to optimization problems
- Performance benchmarks
