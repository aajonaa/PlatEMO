"""
Example: Using TrainedNeuroEA optimizer with CEC2017 problems

This demonstrates how to use the pure Python implementation of NeuroEA
that was trained via transfer learning on CEC2017 problems.
"""

import numpy as np
from neuroea_python import TrainedNeuroEA

# For CEC2017 problems, you can use the PlatEMO or other benchmark suites
# This is a simplified example with Rastrigin function

def rastrigin_function(solution):
    """Rastrigin benchmark function (similar to CEC2017 F9 structure)"""
    A = 10
    n = len(solution)
    return A * n + sum(solution**2 - A * np.cos(2 * np.pi * solution))


def sphere_function(solution):
    """Simple sphere function for testing"""
    return np.sum(solution**2)


class SimpleProblem:
    """Simple problem wrapper for testing"""
    def __init__(self, name, func, n_vars=30, lb=-10, ub=10):
        self.name = name
        self.obj_func = func
        self.n_vars = n_vars
        self.n_dims = n_vars
        self.lb = np.array([lb] * n_vars)
        self.ub = np.array([ub] * n_vars)
        self.minmax = "min"
    
    def generate_solution(self, encoded=True):
        return np.random.uniform(self.lb, self.ub)


def run_neuroea_example():
    """Run NeuroEA on a test problem"""
    
    print("\n" + "="*80)
    print("TRAINED NEUROEA - PURE PYTHON IMPLEMENTATION")
    print("="*80)
    
    # Create optimizer with trained parameters
    optimizer = TrainedNeuroEA(epoch=100, pop_size=30)
    
    # Try to load trained parameters from MATLAB
    try:
        optimizer.load_trained_parameters('trained_NeuroEA_F9_D30_stage2_from_f1.mat')
    except:
        print("Note: Could not load MATLAB trained parameters. Using defaults.")
    
    # Create simple problem
    problem = SimpleProblem("Rastrigin-30D", rastrigin_function, n_vars=30)
    
    print(f"\nOptimizer: TrainedNeuroEA")
    print(f"  Epochs: {optimizer.epoch}")
    print(f"  Population size: {optimizer.pop_size}")
    print(f"  Crossover rate (c1): {optimizer.c1:.4f}")
    print(f"  Mutation rate (m1): {optimizer.m1:.4f}")
    print(f"  Tournament size: {optimizer.tournament_size}")
    
    print(f"\nProblem: {problem.name}")
    print(f"  Dimension: {problem.n_vars}")
    print(f"  Search space: [{problem.lb[0]}, {problem.ub[0]}]^{problem.n_vars}")
    
    # Simulate optimization (would use mealpy framework in practice)
    print(f"\nRunning optimization (simulated)...")
    print("  This would use mealpy framework in production")
    print("  Structure is compatible with:")
    print("    - Mealpy Optimizer base class")
    print("    - Standard optimization problem definition")
    print("    - Multi-threading support via is_parallelizable flag")
    
    return optimizer


if __name__ == "__main__":
    optimizer = run_neuroea_example()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
The TrainedNeuroEA optimizer is now available as a pure Python class that:

1. Maintains the 11-block NeuroEA architecture from training
2. Uses parameters learned via transfer learning (F1 → F9)
3. Is compatible with Mealpy framework
4. Can load trained parameters from MATLAB .mat files
5. Supports custom hyperparameter tuning

To use in production:
    from neuroea_python import TrainedNeuroEA
    from mealpy import FloatVar
    
    model = TrainedNeuroEA(epoch=1000, pop_size=30)
    g_best = model.solve(problem_dict)
    """)
