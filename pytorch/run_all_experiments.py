"""
Run All Ising Model Experiments
================================

Convenience script to run all three experiments in sequence:
1. Ferromagnetic (J > 0)
2. Antiferromagnetic (J < 0)
3. External Field - Maze and Fibonacci patterns
"""


def run_all_experiments():
    """Run all three Ising model experiments"""
    
    print("\n" + "=" * 70)
    print("ISING MACHINE EXPERIMENTS - FULL SUITE")
    print("=" * 70)
    print("\nThis will run three experiments:")
    print("  1. Ferromagnetic (spins align)")
    print("  2. Antiferromagnetic (spins anti-align)")
    print("  3. External Field (maze + Fibonacci patterns)")
    print("\n" + "=" * 70 + "\n")
    
    # Import experiments
    from experiment_ferromagnetic import run_ferromagnetic_experiment
    from experiment_antiferromagnetic import run_antiferromagnetic_experiment
    from experiment_external_field import run_external_field_experiment, create_maze_pattern, create_fibonacci_spiral_pattern
    
    # Experiment parameters
    lattice_size = 64
    n_steps = 2000
    
    # Experiment 1: Ferromagnetic
    print("\n" + "█" * 70)
    print("EXPERIMENT 1/4: FERROMAGNETIC")
    print("█" * 70 + "\n")
    try:
        run_ferromagnetic_experiment(
            lattice_size=lattice_size,
            J_strength=1.0,
            T_initial=5.0,
            T_final=0.01,
            n_steps=n_steps,
            output_dir="outputs/ferromagnetic"
        )
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
    
    # Experiment 2: Antiferromagnetic
    print("\n" + "█" * 70)
    print("EXPERIMENT 2/4: ANTIFERROMAGNETIC")
    print("█" * 70 + "\n")
    try:
        run_antiferromagnetic_experiment(
            lattice_size=lattice_size,
            J_strength=-1.0,
            T_initial=5.0,
            T_final=0.01,
            n_steps=n_steps,
            output_dir="outputs/antiferromagnetic"
        )
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
    
    # Experiment 3: External Field - Maze
    print("\n" + "█" * 70)
    print("EXPERIMENT 3/4: EXTERNAL FIELD (MAZE)")
    print("█" * 70 + "\n")
    try:
        h_maze = create_maze_pattern(lattice_size, field_strength=0.5)
        run_external_field_experiment(
            lattice_size=lattice_size,
            J_strength=0.01,
            h_field=h_maze,
            T_initial=5.0,
            T_final=0.1,
            n_steps=n_steps,
            output_dir="outputs/external_field_maze"
        )
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
    
    # Experiment 4: External Field - Fibonacci
    print("\n" + "█" * 70)
    print("EXPERIMENT 4/4: EXTERNAL FIELD (FIBONACCI)")
    print("█" * 70 + "\n")
    try:
        h_fib = create_fibonacci_spiral_pattern(lattice_size, field_strength=0.5)
        run_external_field_experiment(
            lattice_size=lattice_size,
            J_strength=0.01,
            h_field=h_fib,
            T_initial=5.0,
            T_final=0.1,
            n_steps=n_steps,
            output_dir="outputs/external_field_fibonacci"
        )
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
    
    # Summary
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 70)
    print("\nOutputs saved to:")
    print("  - outputs/ferromagnetic/")
    print("  - outputs/antiferromagnetic/")
    print("  - outputs/external_field_maze/")
    print("  - outputs/external_field_fibonacci/")
    print("\nEach folder contains final state images (.png)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_experiments()
