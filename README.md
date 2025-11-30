# Ising Machine Tutorial

PyTorch implementation of Ising machines for computational optimization using simulated annealing.

**Blog post**: [From Error Correction to Energy Minimization: The Ising Machine](https://ataakbari.github.io/posts/ising-machines.html)

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments
cd pytorch
python run_all_experiments.py
```

**Output**: Final state images in `outputs/` folder

## What is this?

The Ising model describes magnetic materials through interacting spins. It turns out to be a universal framework for solving NP-hard optimization problems by mapping them to energy minimization.

This repository provides a clean, educational PyTorch implementation with:

- N-dimensional lattices (1D, 2D, 3D, ...)
- External magnetic fields (uniform and 2D patterns)
- Simulated annealing optimization
- Binary field patterns (maze and Fibonacci spiral)

## Running Experiments

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

**Output**: Final state images (.png) for each experiment in `outputs/` directory.

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
│   ├── model.py                         # IsingMachine class
│   ├── experiment_ferromagnetic.py      # J > 0 (spins align)
│   ├── experiment_antiferromagnetic.py  # J < 0 (spins anti-align)
│   ├── experiment_external_field.py     # h ≠ 0 (field bias, maze & Fibonacci)
│   └── run_all_experiments.py           # Run all experiments
├── README.md
└── requirements.txt
```

## Experiments Included

Each experiment generates:

- **Image**: Final spin state (.png)
- **Field visualization**: For 2D external field patterns (.png)

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

- **Easy to modify**: Clean object-oriented design with `IsingMachine` class
- **Sequential Metropolis**: Standard Metropolis-Hastings algorithm (inherently sequential)
- **GPU support**: Automatic CUDA/MPS detection (best for lattices ≥128×128)
- **Best for**: Learning, prototyping, and medium-scale problems

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

- Python 3.8+
- PyTorch 2.0+
- matplotlib
- numpy

See `requirements.txt` for exact versions.

## Performance

Typical performance on modern hardware:

| Lattice Size | Spins  | Time (2000 steps) |
| ------------ | ------ | ----------------- |
| 32×32        | 1,024  | ~1-2 seconds      |
| 64×64        | 4,096  | ~5-8 seconds      |
| 128×128      | 16,384 | ~20-30 seconds    |

**Note**: The Metropolis-Hastings algorithm is inherently sequential, limiting GPU parallelization benefits for small lattices.

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
