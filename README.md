# Ising Machine Tutorial

Minimal implementations of Ising machines for computational optimization using simulated annealing.

**Blog post**: [From Error Correction to Energy Minimization: The Ising Machine](https://ataakbari.github.io/posts/ising-machines.html)

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

# Run examples
cd src
python ising_pytorch.py
```

**Output**: Visualizations of spin configurations and energy evolution.

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
├── src/
│   └── ising_pytorch.py      # PyTorch implementation
├── cuda/
│   ├── ising_cuda.cu          # CUDA implementation
│   └── Makefile               # Build system
├── examples/                   # Coming soon: MaxCut, TSP, etc.
├── README.md
└── requirements.txt
```

## Examples Included

### 1. Ferromagnetic 2D Lattice (J > 0, no field)

Spins want to align → forms large domains.

```python
from src.ising_pytorch import run_ferromagnetic_2d_example
spins, energy = run_ferromagnetic_2d_example()
```

### 2. Antiferromagnetic 2D Lattice (J < 0, no field)

Spins want to anti-align → forms checkerboard pattern.

```python
from src.ising_pytorch import run_antiferromagnetic_2d_example
spins, energy = run_antiferromagnetic_2d_example()
```

### 3. With External Field

Field biases spins → non-uniform ground state.

```python
from src.ising_pytorch import run_with_external_field_example
spins, energy = run_with_external_field_example()
```

### 4. 3D Lattice (CUDA only)

Larger systems benefit from CUDA parallelism.

```bash
cd cuda && ./ising_cuda
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

| System | Method | Spins | Time (5000 steps) |
|--------|--------|-------|-------------------|
| 2D 32×32 | PyTorch | 1,024 | ~2 seconds |
| 2D 64×64 | PyTorch | 4,096 | ~8 seconds |
| 2D 64×64 | CUDA | 4,096 | ~0.3 seconds |
| 3D 32×32×32 | CUDA | 32,768 | ~5 seconds |

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

