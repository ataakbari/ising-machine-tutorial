"""
Quick Test Script
=================

Verifies that the PyTorch Ising Machine installation is working correctly.
Runs a minimal test without generating videos.
"""

import torch
from model import IsingMachine


def test_basic_functionality():
    """Test basic IsingMachine functionality"""
    
    print("=" * 60)
    print("ISING MACHINE INSTALLATION TEST")
    print("=" * 60)
    print()
    
    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ PyTorch installed")
    print(f"✓ Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Test model creation
    print("Testing model creation...")
    model = IsingMachine(
        shape=(16, 16),
        J_strength=1.0,
        h_strength=None,
        device=device
    )
    print(f"✓ Model created: {model}")
    print()
    
    # Test energy computation
    print("Testing energy computation...")
    initial_energy = model.compute_energy()
    print(f"✓ Initial energy: {initial_energy.item():.2f}")
    print()
    
    # Test magnetization
    print("Testing magnetization...")
    initial_mag = model.compute_magnetization()
    print(f"✓ Initial magnetization: {initial_mag:.3f}")
    print()
    
    # Test annealing (short run)
    print("Testing annealing (100 steps)...")
    energy_history, temp_history = model.anneal(
        T_initial=5.0,
        T_final=0.1,
        n_steps=100,
        record_interval=10
    )
    final_energy = model.compute_energy()
    final_mag = model.compute_magnetization()
    print(f"✓ Annealing completed")
    print(f"  Final energy: {final_energy.item():.2f}")
    print(f"  Final magnetization: {final_mag:.3f}")
    print(f"  Energy change: {final_energy.item() - initial_energy.item():.2f}")
    print()
    
    # Test spin access
    print("Testing spin access...")
    spins = model.get_spins()
    print(f"✓ Spins shape: {spins.shape}")
    print(f"  Min: {spins.min()}, Max: {spins.max()}")
    print()
    
    # Test reset
    print("Testing reset...")
    model.reset()
    reset_energy = model.compute_energy()
    print(f"✓ Model reset")
    print(f"  Energy after reset: {reset_energy.item():.2f}")
    print()
    
    # Summary
    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    print()
    print("Installation is working correctly!")
    print("You can now run experiments:")
    print("  - python experiment_ferromagnetic.py")
    print("  - python experiment_antiferromagnetic.py")
    print("  - python experiment_external_field.py")
    print("  - python run_all_experiments.py")
    print()


if __name__ == "__main__":
    try:
        test_basic_functionality()
    except Exception as e:
        print()
        print("=" * 60)
        print("TEST FAILED ✗")
        print("=" * 60)
        print(f"\nError: {e}")
        print()
        import traceback
        traceback.print_exc()
        print()
        print("Please check that all dependencies are installed:")
        print("  pip install -r requirements.txt")
        print()

