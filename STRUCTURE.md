# Project Structure

## Overview

The ising-machine-tutorial repository is now organized into two main implementation folders:

```
ising-machine-tutorial/
│
├── pytorch/               # PyTorch implementation with experiments
│   ├── model.py          # IsingMachine class definition
│   ├── experiment_ferromagnetic.py
│   ├── experiment_antiferromagnetic.py
│   ├── experiment_external_field.py
│   ├── run_all_experiments.py
│   ├── __init__.py
│   └── README.md
│
├── cuda/                  # Pure CUDA implementation
│   ├── ising_cuda.cu
│   ├── Makefile
│   └── README.md
│
├── examples/              # Future: MaxCut, TSP, etc.
├── README.md             # Main documentation
├── requirements.txt      # Python dependencies
└── STRUCTURE.md         # This file
```

## PyTorch Implementation

### Core Model (`model.py`)

Defines the `IsingMachine` class with:

- Flexible N-dimensional lattice support
- Nearest-neighbor coupling configuration
- External field support
- Simulated annealing with Metropolis-Hastings
- Energy and magnetization computation
- Visualization utilities

**Key Methods:**

- `__init__(shape, J_strength, h_strength, device)` - Initialize system
- `anneal(T_initial, T_final, n_steps)` - Run simulated annealing
- `compute_energy()` - Calculate total energy
- `compute_magnetization()` - Calculate average spin
- `visualize(title, save_path)` - Create visualization
- `metropolis_step(temperature)` - Single MC sweep

### Experiments

Each experiment:

1. Initializes an `IsingMachine` with specific parameters
2. Records spin configurations during annealing
3. Generates evolution video (MP4)
4. Saves final state image (PNG)
5. Plots energy and magnetization over time

#### Experiment 1: Ferromagnetic (`experiment_ferromagnetic.py`)

- **Parameters**: J = +1.0, h = 0
- **Expected**: Large aligned domains, high |M|
- **Output**: `outputs/ferromagnetic/`

#### Experiment 2: Antiferromagnetic (`experiment_antiferromagnetic.py`)

- **Parameters**: J = -1.0, h = 0
- **Expected**: Checkerboard pattern, M ≈ 0
- **Output**: `outputs/antiferromagnetic/`

#### Experiment 3: External Field (`experiment_external_field.py`)

- **Parameters**: J = +1.0, h = +0.5
- **Expected**: Field-biased alignment, high M
- **Output**: `outputs/external_field/`

### Running Experiments

**All at once:**

```bash
cd pytorch
python run_all_experiments.py
```

**Individual:**

```bash
python experiment_ferromagnetic.py
python experiment_antiferromagnetic.py
python experiment_external_field.py
```

**Custom usage:**

```python
from pytorch.model import IsingMachine

model = IsingMachine(shape=(32, 32), J_strength=1.0)
model.anneal(T_initial=5.0, T_final=0.01, n_steps=2000)
model.visualize()
```

## CUDA Implementation

High-performance implementation with:

- Checkerboard decomposition for parallelism
- cuRAND for GPU-based random number generation
- Parallel reduction for energy calculation
- Multiple built-in examples

**Build and run:**

```bash
cd cuda
make
./ising_cuda
```

## Migration Notes

### Changes from Original Structure

**Before:**

```
src/
└── ising_pytorch.py  # Monolithic script
```

**After:**

```
pytorch/
├── model.py                         # Class-based design
├── experiment_ferromagnetic.py      # Separate experiments
├── experiment_antiferromagnetic.py
├── experiment_external_field.py
└── run_all_experiments.py          # Orchestration
```

### Benefits of New Structure

1. **Modularity**: `IsingMachine` class is reusable
2. **Clarity**: Each experiment in its own file
3. **Visualization**: Evolution videos show dynamics
4. **Organization**: Clear separation of concerns
5. **Extensibility**: Easy to add new experiments
6. **Documentation**: README in each folder

## Output Structure

When running experiments, outputs are organized as:

```
outputs/
├── ferromagnetic/
│   ├── ferromagnetic_evolution.mp4
│   └── ferromagnetic_final.png
├── antiferromagnetic/
│   ├── antiferromagnetic_evolution.mp4
│   └── antiferromagnetic_final.png
└── external_field/
    ├── external_field_evolution.mp4
    └── external_field_final.png
```

## Dependencies

**PyTorch experiments require:**

- Python 3.8+
- torch >= 2.0
- numpy
- matplotlib
- ffmpeg (for video generation)

**CUDA implementation requires:**

- CUDA toolkit 11.0+
- nvcc compiler

See `requirements.txt` for exact Python versions.

## Future Additions

Planned for `examples/` directory:

- MaxCut problem mapping
- Traveling Salesman Problem (TSP)
- Number Partitioning
- Graph Coloring
- Portfolio Optimization

Each will demonstrate how to map NP-hard problems to Ising model energy functions.
