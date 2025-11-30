# Ising Machine Tutorial

Minimal implementations of Ising machines for computational optimization using simulated annealing.

**Blog post**: [From Error Correction to Energy Minimization: The Ising Machine](https://ataakbari.github.io/posts/ising-machines.html)

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all PyTorch experiments (generates videos!)
cd pytorch
python run_all_experiments.py
```

**Output**: Evolution videos and final state images in `outputs/` folder

📖 **New here?** Start with [`GETTING_STARTED.md`](GETTING_STARTED.md)

## What is this?

The Ising model describes magnetic materials through interacting spins. It turns out to be a universal framework for solving NP-hard optimization problems by mapping them to energy minimization.

This repository provides clean, educational implementations in:

- **PyTorch** (GPU-accelerated, easy to experiment with)
- **Pure CUDA** (maximum performance, closer to hardware)

Both support:

- N-dimensional lattices (1D, 2D, 3D, ...)
- With and without external magnetic fields
- Simulated annealing optimization

## Quick Start

### PyTorch Version

```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments (ferromagnetic, antiferromagnetic, external field)
cd pytorch
python run_all_experiments.py

# Or run individual experiments
python experiment_ferromagnetic.py
python experiment_antiferromagnetic.py
python experiment_external_field.py
```

**Output**: Final state images (.png) for each experiment.

### CUDA Version

```bash
# Compile (requires CUDA toolkit)
cd cuda
make

# Run examples
./ising_cuda
```

**Output**: Console output showing energy minimization progress.

## Physics Background

### The Ising Model

Energy function:

```
E = -Σ J_ij s_i s_j - Σ h_i s_i
```

Where:

- `s_i ∈ {-1, +1}`: spin states
- `J_ij`: coupling between spins i and j
  - `J > 0`: ferromagnetic (spins align)
  - `J < 0`: antiferromagnetic (spins anti-align)
- `h_i`: external magnetic field (bias on spin i)

### Simulated Annealing

Find low-energy configurations by gradually cooling:

1. Start hot (high temperature T) → random exploration
2. Gradually cool → settle into low-energy states
3. Use Metropolis criterion:
   - Accept moves that lower energy
   - Accept energy-raising moves with probability `exp(-ΔE/T)`

## Code Structure

```
ising-machine-tutorial/
├── pytorch/
│   ├── model.py                      # IsingMachine class
│   ├── experiment_ferromagnetic.py   # J > 0 (spins align)
│   ├── experiment_antiferromagnetic.py  # J < 0 (spins anti-align)
│   ├── experiment_external_field.py  # h ≠ 0 (field bias)
│   └── run_all_experiments.py        # Run all experiments
├── cuda/
│   ├── ising_cuda.cu                 # CUDA implementation
│   ├── Makefile                      # Build system
│   └── README.md                     # CUDA-specific docs
├── examples/                          # Coming soon: MaxCut, TSP, etc.
├── README.md
└── requirements.txt
```

## Experiments Included

Each PyTorch experiment generates:

- **Video**: Evolution of spin configuration during annealing (.mp4)
- **Image**: Final spin state (.png)

### 1. Ferromagnetic (J > 0, no field)

Spins want to align → forms large domains.

```bash
cd pytorch
python experiment_ferromagnetic.py
```

**Expected**: Large regions of uniform spin orientation, high |magnetization|.

### 2. Antiferromagnetic (J < 0, no field)

Spins want to anti-align → forms checkerboard pattern.

```bash
cd pytorch
python experiment_antiferromagnetic.py
```

**Expected**: Alternating up/down spins, magnetization ≈ 0.

### 3. External Field (h ≠ 0)

Field biases spins → non-uniform ground state. Supports spatially varying 2D binary field patterns.

```bash
cd pytorch
python experiment_external_field.py
```

This runs two binary pattern experiments:

- **Maze pattern**: Recursive backtracking maze (walls: +0.5, paths: -0.5)
- **Fibonacci spiral**: Golden angle spiral pattern (arms: +0.5, spaces: -0.5)

**Expected**: Spins follow binary field patterns. Competition between field constraints and ferromagnetic coupling creates interesting domain structures.

### Using the Model Directly

```python
from pytorch.model import IsingMachine
from pytorch.experiment_external_field import create_maze_pattern, create_fibonacci_spiral_pattern

# Example 1: No field
model = IsingMachine(shape=(32, 32), J_strength=1.0, h_strength=None)

# Example 2: Uniform field
model = IsingMachine(shape=(32, 32), J_strength=1.0, h_strength=0.5)

# Example 3: Binary maze pattern
h_maze = create_maze_pattern(32, field_strength=0.5)
model = IsingMachine(shape=(32, 32), J_strength=1.0, h_field=h_maze)

# Example 4: Binary Fibonacci spiral
h_fib = create_fibonacci_spiral_pattern(32, field_strength=0.5)
model = IsingMachine(shape=(32, 32), J_strength=1.0, h_field=h_fib)

# Run annealing
energy_history, temp_history = model.anneal(
    T_initial=5.0, T_final=0.01, n_steps=2000
)

# Visualize result
model.visualize(title="My Ising Model")

# Get final properties
energy = model.compute_energy()
magnetization = model.compute_magnetization()
```

## Implementation Notes

### PyTorch Version

- **Pros**: Easy to modify, visualize, integrate with ML
- **Cons**: Python overhead, sequential Metropolis updates
- **Best for**: Prototyping, learning, small-medium problems

### CUDA Version

- **Pros**: Maximum performance, true parallelism (checkerboard)
- **Cons**: Harder to modify, requires CUDA toolkit
- **Best for**: Large-scale problems, production use

## Mapping Problems to Ising Models

### Example: Graph MaxCut

**Problem**: Partition graph vertices to maximize edges between partitions.

**Mapping**:

- Vertex i → spin s_i
- s_i = +1 → partition 1, s_i = -1 → partition 2
- Edge (i,j) → coupling J_ij = +1 (antiferromagnetic)

Minimizing energy maximizes the cut!

### Example: Number Partitioning

**Problem**: Partition numbers {a₁, a₂, ..., aₙ} into two equal-sum sets.

**Mapping**:

- Number aᵢ → spin s_i
- Coupling: J_ij = aᵢ aⱼ
- Minimize: (Σ sᵢ aᵢ)²

More examples coming in `/examples` directory.

## Requirements

- **PyTorch**: Python 3.8+, PyTorch 2.0+, matplotlib, numpy
- **CUDA**: CUDA toolkit 11.0+, nvcc compiler

See `requirements.txt` for exact versions.

## Performance

Tested on NVIDIA RTX 4090:

| System      | Method  | Spins  | Time (5000 steps) |
| ----------- | ------- | ------ | ----------------- |
| 2D 32×32    | PyTorch | 1,024  | ~2 seconds        |
| 2D 64×64    | PyTorch | 4,096  | ~8 seconds        |
| 2D 64×64    | CUDA    | 4,096  | ~0.3 seconds      |
| 3D 32×32×32 | CUDA    | 32,768 | ~5 seconds        |

CUDA is ~25× faster due to parallel checkerboard updates.

## Learn More

- **Blog post**: [Ising Machines Explained](https://ataakbari.github.io/posts/ising-machines.html)
- **Next post**: Quantum Annealing (coming soon)
- **Original paper**: E. Ising, "Beitrag zur Theorie des Ferromagnetismus" (1925)

## Citation

If you use this code in research, please cite:

```bibtex
@misc{akbari2025ising,
  author = {Akbari Asanjan, Ata},
  title = {Ising Machine Tutorial},
  year = {2025},
  url = {https://github.com/ataakbari/ising-machine-tutorial}
}
```

## License

MIT License - feel free to use and modify!

## Contributing

Found a bug? Have a cool optimization problem to add? Open an issue or PR!

---

**Author**: Ata Akbari Asanjan  
**Website**: [ataakbari.github.io](https://ataakbari.github.io)  
**Blog**: [Quantum & Classical Computing](https://ataakbari.github.io/quantum-blog.html)
