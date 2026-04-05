#!/usr/bin/env python
# Example: Using NeuroEA with Mealpy framework

import numpy as np
from mealpy import FloatVar
from NeuroEA import OriginalNeuroEA, TrainedNeuroEA


def objective_function(solution):
    """
    Simple test function: Sphere function
    f(x) = sum(x_i^2)
    Global optimum: f(0,0,...,0) = 0
    """
    return np.sum(solution ** 2)


def rastrigin_function(solution):
    """
    Rastrigin benchmark function
    More complex, multimodal landscape
    """
    A = 10
    return A * len(solution) + np.sum(solution ** 2 - A * np.cos(2 * np.pi * solution))


def example_original_neuroea():
    """Example using original NeuroEA with custom parameters"""
    print("\n" + "="*80)
    print("EXAMPLE 1: OriginalNeuroEA with Custom Parameters")
    print("="*80 + "\n")
    
    # Define optimization problem
    problem_dict = {
        "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="x"),
        "obj_func": objective_function,
        "minmax": "min",
    }
    
    # Create optimizer with custom hyperparameters
    model = OriginalNeuroEA(epoch=50, pop_size=30, c1=0.6, m1=0.15, tournament_size=10)
    
    print(f"Configuration:")
    print(f"  Algorithm: NeuroEA (Original)")
    print(f"  Problem: Sphere Function (D=30)")
    print(f"  Epochs: {model.epoch}")
    print(f"  Population: {model.pop_size}")
    print(f"  Crossover rate (c1): {model.c1}")
    print(f"  Mutation rate (m1): {model.m1}")
    print(f"  Tournament size: {model.tournament_size}\n")
    
    # Solve the problem
    print("Running optimization...")
    g_best = model.solve(problem_dict)
    
    # Results
    print(f"\nResults:")
    print(f"  Best fitness: {g_best.target.fitness:.6e}")
    print(f"  Best solution (first 5 dims): {g_best.solution[:5]}")


def example_trained_neuroea():
    """Example using pre-trained NeuroEA (transfer learning)"""
    print("\n" + "="*80)
    print("EXAMPLE 2: TrainedNeuroEA (Transfer Learning from CEC2017)")
    print("="*80 + "\n")
    
    # Define optimization problem
    problem_dict = {
        "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="x"),
        "obj_func": rastrigin_function,
        "minmax": "min",
    }
    
    # Create optimizer using trained parameters
    model = TrainedNeuroEA(epoch=50, pop_size=30, tournament_size=10)
    
    # Display training information
    model.information()
    
    print(f"Problem: Rastrigin Function (D=30)\n")
    
    # Solve the problem
    print("Running optimization with transfer-learned parameters...")
    g_best = model.solve(problem_dict)
    
    # Results
    print(f"\nResults:")
    print(f"  Best fitness: {g_best.target.fitness:.6e}")
    print(f"  Best solution (first 5 dims): {g_best.solution[:5]}")
    print(f"  Algorithm: {model.__class__.__name__}")


def example_parametric_comparison():
    """Compare different hyperparameter settings"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Hyperparameter Comparison")
    print("="*80 + "\n")
    
    problem_dict = {
        "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="x"),
        "obj_func": objective_function,
        "minmax": "min",
    }
    
    # Test different configurations
    configurations = [
        {"name": "Low crossover", "c1": 0.3, "m1": 0.1},
        {"name": "High crossover", "c1": 0.8, "m1": 0.1},
        {"name": "Low mutation", "c1": 0.5, "m1": 0.05},
        {"name": "High mutation", "c1": 0.5, "m1": 0.2},
    ]
    
    results = []
    
    for config in configurations:
        print(f"\nTesting: {config['name']}")
        print(f"  c1={config['c1']}, m1={config['m1']}")
        
        model = OriginalNeuroEA(epoch=30, pop_size=30, 
                               c1=config['c1'], m1=config['m1'])
        g_best = model.solve(problem_dict)
        
        results.append({
            'name': config['name'],
            'fitness': g_best.target.fitness,
            'c1': config['c1'],
            'm1': config['m1']
        })
        
        print(f"  Best fitness: {g_best.target.fitness:.6e}")
    
    # Summary
    print("\n" + "-"*80)
    print("SUMMARY:")
    print("-"*80)
    best_result = min(results, key=lambda x: x['fitness'])
    print(f"\nBest configuration: {best_result['name']}")
    print(f"  c1={best_result['c1']}, m1={best_result['m1']}")
    print(f"  Best fitness: {best_result['fitness']:.6e}")


if __name__ == "__main__":
    # Run examples
    example_original_neuroea()
    example_trained_neuroea()
    example_parametric_comparison()
    
    print("\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80)
    print("""
To use NeuroEA in your code:

1. Import the algorithm:
   from NeuroEA import OriginalNeuroEA, TrainedNeuroEA

2. Create optimizer:
   model = OriginalNeuroEA(epoch=100, pop_size=30, c1=0.5, m1=0.1)

3. Define problem:
   problem_dict = {
       "bounds": FloatVar(n_vars=30, lb=(-10.,)*30, ub=(10.,)*30, name="x"),
       "obj_func": your_objective_function,
       "minmax": "min",
   }

4. Solve:
   g_best = model.solve(problem_dict)
   print(f"Best fitness: {g_best.target.fitness}")

For transfer-learned parameters:
   model = TrainedNeuroEA(epoch=100, pop_size=30)
   model.information()  # Show training details
   g_best = model.solve(problem_dict)
""")
