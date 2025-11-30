# Getting Started with Ising Machine Tutorial

Quick start guide for running the experiments.

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:

- PyTorch (GPU-accelerated if CUDA available)
- NumPy
- Matplotlib

### 2. Install FFMpeg (for video generation)

**macOS:**

```bash
brew install ffmpeg
```

**Ubuntu/Debian:**

```bash
sudo apt install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### 3. Test Installation

```bash
cd pytorch
python test_installation.py
```

If you see "ALL TESTS PASSED ✓", you're ready to go!

## Running Experiments

### Option 1: Run All Experiments (Recommended)

```bash
cd pytorch
python run_all_experiments.py
```

This runs all three experiments in sequence:

1. Ferromagnetic (J > 0)
2. Antiferromagnetic (J < 0)
3. External Field (h ≠ 0)

**Time**: ~5-10 minutes total (depending on hardware)
**Output**: Videos and images in `outputs/` directory

### Option 2: Run Individual Experiments

```bash
cd pytorch

# Experiment 1: Ferromagnetic
python experiment_ferromagnetic.py

# Experiment 2: Antiferromagnetic
python experiment_antiferromagnetic.py

# Experiment 3: External Field
python experiment_external_field.py
```

## What You'll Get

Each experiment produces:

### 1. Evolution Video (`*_evolution.mp4`)

Shows three panels:

- **Left**: Spin configuration evolving during annealing
- **Middle**: Energy decreasing over time (blue)
- **Right**: Magnetization evolution (red)

### 2. Final State Image (`*_final.png`)

Shows the final spin configuration after annealing completes

### Output Structure

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

## Expected Results

### Ferromagnetic (J > 0)

- **Pattern**: Large domains of uniform spin
- **Magnetization**: High (|M| ≈ 1)
- **Explanation**: Spins prefer to align

### Antiferromagnetic (J < 0)

- **Pattern**: Checkerboard (alternating up/down)
- **Magnetization**: Near zero (M ≈ 0)
- **Explanation**: Spins prefer to anti-align

### External Field (h = 0.5)

- **Pattern**: Most spins aligned with field
- **Magnetization**: High positive (M ≈ +0.9)
- **Explanation**: Field biases spins upward

## Using the Model Directly

For custom experiments, use the `IsingMachine` class:

```python
from model import IsingMachine

# Create model
model = IsingMachine(
    shape=(64, 64),      # 64×64 lattice
    J_strength=1.0,      # Ferromagnetic
    h_strength=None,     # No field
    device='cuda'        # Use GPU if available
)

# Run annealing
energy_hist, temp_hist = model.anneal(
    T_initial=5.0,
    T_final=0.01,
    n_steps=2000
)

# Get results
final_energy = model.compute_energy()
final_mag = model.compute_magnetization()

print(f"Energy: {final_energy:.2f}")
print(f"Magnetization: {final_mag:.3f}")

# Visualize
model.visualize(save_path="my_result.png")
```

## Customizing Experiments

Edit parameters in experiment files:

```python
run_ferromagnetic_with_recording(
    lattice_size=64,      # Size of square lattice
    J_strength=1.0,       # Coupling strength
    T_initial=5.0,        # Start hot
    T_final=0.01,         # End cold
    n_steps=2000,         # Annealing steps
    record_every=20,      # Video frame rate
    output_dir="outputs/ferromagnetic"
)
```

## Performance Tips

### For Faster Experiments

- Reduce `lattice_size` (e.g., 32)
- Reduce `n_steps` (e.g., 1000)
- Increase `record_every` (e.g., 50)

### For Better Quality

- Increase `lattice_size` (e.g., 128)
- Increase `n_steps` (e.g., 5000)
- Decrease `record_every` (e.g., 10)

### GPU vs CPU

- **GPU (CUDA)**: ~10× faster, recommended
- **CPU**: Works but slower, use for small systems

## Troubleshooting

### ImportError: No module named 'model'

**Solution**: Make sure you're in the `pytorch/` directory

```bash
cd pytorch
python experiment_ferromagnetic.py
```

### RuntimeError: CUDA out of memory

**Solution**: Reduce lattice size or use CPU

```python
lattice_size=32  # Instead of 64
# or
device='cpu'
```

### FFMpegWriter not available

**Solution**: Install FFMpeg (see Installation section)

### Video generation is slow

**Solution**:

- Increase `record_every` to create fewer frames
- Reduce `lattice_size`
- This is normal for CPU; GPU helps significantly

## CUDA Implementation

For maximum performance, try the CUDA version:

```bash
cd cuda
make
./ising_cuda
```

**Note**: Requires NVIDIA GPU and CUDA toolkit

## Next Steps

1. ✓ Run the experiments
2. ✓ Watch the videos to understand the physics
3. ✓ Try customizing parameters
4. ✓ Modify the model for your own problems
5. ✓ Read the blog post: [ataakbari.github.io](https://ataakbari.github.io/posts/ising-machines.html)

## Need Help?

- Check `pytorch/README.md` for detailed API docs
- Check `STRUCTURE.md` for project organization
- See main `README.md` for physics background
- Open an issue on GitHub

---

**Happy experimenting! 🧲**
