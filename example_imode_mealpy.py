#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IMODE Quick-Start Examples
Created: 06/04/2026

Three simple usage options for quick testing and integration.
Perfect for rapid prototyping and benchmarking.
"""

import numpy as np
from mealpy import FloatVar
from IMODE import OriginalIMODE, TrainedIMODE


# ============================================================================
# Option 1: Quick Start with Default Parameters
# ============================================================================

def quick_start_default():
    """
    Option 1: Quick Start with Default Parameters
    
    Simplest usage: just define a problem and run IMODE.
    Uses default hyperparameters optimized for general problems.
    """
    print("\n" + "="*80)
    print("Option 1: Quick Start with Default Parameters")
    print("="*80)
    
    # Define benchmark functions
    def sphere(solution):
        """Sphere: f(x) = sum(x_i^2), optimum at 0"""
        return np.sum(solution**2)
    
    def rosenbrock(solution):
        """Rosenbrock: challenging valley function"""
        return sum(100*(solution[i+1] - solution[i]**2)**2 
                  + (1 - solution[i])**2 
                  for i in range(len(solution)-1))
    
    def rastrigin(solution):
        """Rastrigin: highly multimodal function"""
        A = 10
        return A*len(solution) + sum(solution**2 - A*np.cos(2*np.pi*solution))
    
    # Test each function
    functions = {
        'Sphere': sphere,
        'Rosenbrock': rosenbrock,
        'Rastrigin': rastrigin
    }
    
    for name, func in functions.items():
        print(f"\nOptimizing {name}...")
        
        # Define problem
        problem = {
            "bounds": FloatVar(n_vars=20, lb=(-100.,)*20, ub=(100.,)*20, name="x"),
            "obj_func": func,
            "minmax": "min",
        }
        
        # Run IMODE
        model = OriginalIMODE(epoch=50, pop_size=50)
        best = model.solve(problem)
        
        print(f"  Result: {best.target.fitness:.6e}")


# ============================================================================
# Option 2: Customized Configuration for Faster Convergence
# ============================================================================

def quick_start_custom():
    """
    Option 2: Customized Configuration
    
    Adjust hyperparameters for better performance on your specific problem:
      - Increase population size for harder problems
      - Adjust aRate for diversity control
      - Tune minN for population reduction speed
    """
    print("\n" + "="*80)
    print("Option 2: Customized Configuration")
    print("="*80)
    
    def ackley(solution):
        """Ackley function - multimodal, harder than Sphere"""
        n = len(solution)
        s1 = np.sum(solution**2)
        s2 = np.sum(np.cos(2*np.pi*solution))
        return -20*np.exp(-0.2*np.sqrt(s1/n)) - np.exp(s2/n) + 20 + np.e
    
    problem = {
        "bounds": FloatVar(n_vars=20, lb=(-32.768,)*20, ub=(32.768,)*20, name="x"),
        "obj_func": ackley,
        "minmax": "min",
    }
    
    print("\nConfiguration A: Balanced (Default)")
    model_a = OriginalIMODE(epoch=100, pop_size=50, minN=4, aRate=2.6)
    best_a = model_a.solve(problem)
    print(f"  Fitness: {best_a.target.fitness:.6e}")
    
    print("\nConfiguration B: Larger Population (for harder problems)")
    model_b = OriginalIMODE(epoch=100, pop_size=100, minN=10, aRate=2.6)
    best_b = model_b.solve(problem)
    print(f"  Fitness: {best_b.target.fitness:.6e}")
    
    print("\nConfiguration C: Small Population (for quick testing)")
    model_c = OriginalIMODE(epoch=100, pop_size=30, minN=4, aRate=1.5)
    best_c = model_c.solve(problem)
    print(f"  Fitness: {best_c.target.fitness:.6e}")
    
    print("\nBest configuration for this problem:")
    results = [
        ('A (Default)', best_a.target.fitness),
        ('B (Large pop)', best_b.target.fitness),
        ('C (Small pop)', best_c.target.fitness),
    ]
    best_config = min(results, key=lambda x: x[1])
    print(f"  {best_config[0]}: {best_config[1]:.6e}")


# ============================================================================
# Option 3: Algorithm Comparison
# ============================================================================

def quick_start_comparison():
    """
    Option 3: Algorithm Comparison
    
    Compare OriginalIMODE vs TrainedIMODE with different mutation operators
    and parameter configurations on the same problem.
    """
    print("\n" + "="*80)
    print("Option 3: OriginalIMODE vs TrainedIMODE Comparison")
    print("="*80)
    
    def sphere(solution):
        return np.sum(solution**2)
    
    problem = {
        "bounds": FloatVar(n_vars=30, lb=(-100.,)*30, ub=(100.,)*30, name="x"),
        "obj_func": sphere,
        "minmax": "min",
    }
    
    print("\nProblem: Sphere function, D=30, [-100, 100]")
    print("Configuration: epoch=100, pop_size=50")
    
    # Test OriginalIMODE
    print("\n[1/2] Running OriginalIMODE...")
    model1 = OriginalIMODE(epoch=100, pop_size=50)
    best1 = model1.solve(problem)
    
    # Test TrainedIMODE
    print("[2/2] Running TrainedIMODE...")
    model2 = TrainedIMODE(epoch=100, pop_size=50)
    best2 = model2.solve(problem)
    
    # Compare results
    print("\n" + "="*80)
    print("Comparison Results:")
    print("="*80)
    print(f"\n{'Algorithm':<25} {'Fitness':<20} {'Status':<20}")
    print("-"*65)
    print(f"{'OriginalIMODE':<25} {best1.target.fitness:<20.6e} {'Baseline':<20}")
    print(f"{'TrainedIMODE':<25} {best2.target.fitness:<20.6e} {'Default params':<20}")
    
    if best1.target.fitness < best2.target.fitness:
        improvement = (best2.target.fitness - best1.target.fitness) / best2.target.fitness * 100
        print(f"\nWinner: OriginalIMODE ({improvement:.1f}% better)")
    else:
        improvement = (best1.target.fitness - best2.target.fitness) / best1.target.fitness * 100
        print(f"\nWinner: TrainedIMODE ({improvement:.1f}% better)")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("IMODE Quick-Start Examples")
    print("Improved Multi-Operator Differential Evolution in mealpy")
    print("="*80)
    
    # Run all options
    quick_start_default()
    quick_start_custom()
    quick_start_comparison()
    
    print("\n" + "="*80)
    print("Quick-start examples completed!")
    print("="*80 + "\n")
