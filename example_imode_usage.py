#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IMODE Examples - Comprehensive Usage Patterns
Created: 06/04/2026

This module demonstrates:
  1. Basic IMODE usage with default parameters
  2. Custom problem definition
  3. Hyperparameter tuning
  4. Multi-run comparative analysis
  5. Detailed configuration and monitoring
  6. Algorithm information display
"""

import numpy as np
from mealpy import FloatVar
from IMODE import OriginalIMODE, TrainedIMODE


# ============================================================================
# Example 1: Basic IMODE Usage with Default Parameters
# ============================================================================

def example_1_basic_usage():
    """
    Example 1: Basic IMODE Usage
    
    This example demonstrates the simplest way to use IMODE on a standard
    optimization problem (Sphere function in 30 dimensions).
    """
    print("\n" + "="*80)
    print("Example 1: Basic IMODE Usage")
    print("="*80)
    
    # Define objective function
    def sphere_function(solution):
        """Sphere function - simple quadratic, optimal at origin"""
        return np.sum(solution**2)
    
    # Define problem
    problem_dict = {
        "bounds": FloatVar(n_vars=30, lb=(-100.,)*30, ub=(100.,)*30, name="x"),
        "obj_func": sphere_function,
        "minmax": "min",
    }
    
    # Create and run IMODE with default parameters
    print("\nAlgorithm: OriginalIMODE")
    print("Problem: Sphere function, D=30, [-100, 100]")
    print("Parameters: epoch=50, pop_size=50, minN=4, aRate=2.6 (defaults)")
    
    model = OriginalIMODE(epoch=50, pop_size=50)
    g_best = model.solve(problem_dict)
    
    print(f"\nOptimization Results:")
    print(f"  Best solution: {g_best.solution[:3]}... (first 3 dims)")
    print(f"  Best fitness: {g_best.target.fitness:.6e}")
    print(f"  Expected: ~0.0 (optimum at origin)")
    print(f"  Status: {'✓ Converged' if g_best.target.fitness < 1e2 else '✗ Needs more epochs'}")


# ============================================================================
# Example 2: Custom Problem Definition
# ============================================================================

def example_2_custom_problem():
    """
    Example 2: Custom Problem Definition
    
    Define and optimize a custom problem: Rosenbrock function
    f(x) = sum(100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2)
    Optimum at x = [1, 1, ..., 1] with f = 0
    """
    print("\n" + "="*80)
    print("Example 2: Custom Problem - Rosenbrock Function")
    print("="*80)
    
    def rosenbrock_function(solution):
        """Rosenbrock function - challenging, narrow valley"""
        sum_val = 0
        for i in range(len(solution) - 1):
            sum_val += 100 * (solution[i+1] - solution[i]**2)**2
            sum_val += (1 - solution[i])**2
        return sum_val
    
    # Define problem with custom bounds [-2, 2]
    problem_dict = {
        "bounds": FloatVar(n_vars=20, lb=(-2.,)*20, ub=(2.,)*20, name="x"),
        "obj_func": rosenbrock_function,
        "minmax": "min",
    }
    
    print("\nAlgorithm: OriginalIMODE")
    print("Problem: Rosenbrock function, D=20, [-2, 2]")
    print("Difficulty: High (narrow, curved valley)")
    print("Parameters: epoch=100, pop_size=50 (increased budget)")
    
    model = OriginalIMODE(epoch=100, pop_size=50)
    g_best = model.solve(problem_dict)
    
    print(f"\nOptimization Results:")
    print(f"  Best solution: {g_best.solution[:3]}... (first 3 dims)")
    print(f"  Best fitness: {g_best.target.fitness:.6e}")
    print(f"  Expected: ~0.0 (optimum at [1,1,...,1])")
    print(f"  Note: Rosenbrock is very difficult; residual > 0 is normal")


# ============================================================================
# Example 3: Hyperparameter Study - Population Size
# ============================================================================

def example_3_hyperparam_study():
    """
    Example 3: Hyperparameter Study
    
    Compare different population sizes to understand IMODE's behavior
    with varying amounts of parallelism.
    """
    print("\n" + "="*80)
    print("Example 3: Hyperparameter Study - Population Size Effect")
    print("="*80)
    
    def ackley_function(solution):
        """Ackley function - multimodal, prone to local optima"""
        n = len(solution)
        sum_sq = np.sum(solution**2)
        sum_cos = np.sum(np.cos(2*np.pi*solution))
        return (-20 * np.exp(-0.2 * np.sqrt(sum_sq/n)) 
                - np.exp(sum_cos/n) + 20 + np.e)
    
    problem_dict = {
        "bounds": FloatVar(n_vars=15, lb=(-32.768,)*15, ub=(32.768,)*15, name="x"),
        "obj_func": ackley_function,
        "minmax": "min",
    }
    
    # Test different population sizes
    pop_sizes = [30, 50, 100]
    results = {}
    
    print("\nProblem: Ackley function, D=15, [-32.768, 32.768]")
    print("Difficulty: Multimodal (many local optima)")
    print("\nTesting different population sizes...")
    
    for pop_size in pop_sizes:
        print(f"\n  Testing pop_size={pop_size}...")
        model = OriginalIMODE(epoch=50, pop_size=pop_size)
        g_best = model.solve(problem_dict)
        results[pop_size] = g_best.target.fitness
        print(f"    Fitness: {g_best.target.fitness:.6e}")
    
    print(f"\nComparison Results:")
    print(f"  {'Population Size':<20} {'Fitness':<20} {'Rank':<10}")
    print(f"  {'-'*50}")
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    for rank, (pop_size, fitness) in enumerate(sorted_results, 1):
        print(f"  {pop_size:<20} {fitness:<20.6e} {'#' + str(rank):<10}")
    
    print(f"\nConclusion:")
    print(f"  Best configuration: pop_size={sorted_results[0][0]} with fitness={sorted_results[0][1]:.6e}")


# ============================================================================
# Example 4: Archive Size Ratio Study
# ============================================================================

def example_4_archive_ratio():
    """
    Example 4: Archive Size Ratio (aRate) Study
    
    IMODE maintains an archive for diversity. The aRate parameter controls
    archive size relative to population size. Higher aRate = more diversity.
    """
    print("\n" + "="*80)
    print("Example 4: Archive Size Ratio (aRate) Study")
    print("="*80)
    
    def rastrigin_function(solution):
        """Rastrigin function - highly multimodal"""
        A = 10
        n = len(solution)
        return A*n + np.sum(solution**2 - A*np.cos(2*np.pi*solution))
    
    problem_dict = {
        "bounds": FloatVar(n_vars=10, lb=(-5.12,)*10, ub=(5.12,)*10, name="x"),
        "obj_func": rastrigin_function,
        "minmax": "min",
    }
    
    # Test different archive ratios
    aRates = [1.0, 2.0, 2.6, 4.0]
    results = {}
    
    print("\nProblem: Rastrigin function, D=10, [-5.12, 5.12]")
    print("Difficulty: Highly multimodal (9801 local optima in 2D)")
    print("\nTesting different archive ratios...")
    
    for aRate in aRates:
        print(f"\n  Testing aRate={aRate}...")
        model = OriginalIMODE(epoch=100, pop_size=50, aRate=aRate)
        g_best = model.solve(problem_dict)
        results[aRate] = g_best.target.fitness
        print(f"    Fitness: {g_best.target.fitness:.6e}")
    
    print(f"\nComparison Results:")
    print(f"  {'Archive Ratio':<20} {'Fitness':<20} {'Conclusion':<30}")
    print(f"  {'-'*70}")
    
    for aRate in aRates:
        fitness = results[aRate]
        if aRate <= 1.5:
            conc = "Low diversity"
        elif aRate <= 2.6:
            conc = "Balanced (recommended)"
        else:
            conc = "High diversity"
        print(f"  {aRate:<20} {fitness:<20.6e} {conc:<30}")


# ============================================================================
# Example 5: Multi-run Analysis with Statistics
# ============================================================================

def example_5_multi_run():
    """
    Example 5: Multi-run Analysis
    
    Run IMODE multiple times and compute statistics to assess algorithm
    reliability and consistency.
    """
    print("\n" + "="*80)
    print("Example 5: Multi-run Analysis and Statistics")
    print("="*80)
    
    def sphere_function(solution):
        return np.sum(solution**2)
    
    problem_dict = {
        "bounds": FloatVar(n_vars=30, lb=(-100.,)*30, ub=(100.,)*30, name="x"),
        "obj_func": sphere_function,
        "minmax": "min",
    }
    
    # Run multiple times
    num_runs = 5
    results = []
    
    print(f"\nProblem: Sphere function, D=30")
    print(f"Number of runs: {num_runs}")
    print(f"Configuration: epoch=50, pop_size=50")
    print(f"\nRunning...")
    
    for run in range(num_runs):
        print(f"  Run {run+1}/{num_runs}...", end=" ", flush=True)
        model = OriginalIMODE(epoch=50, pop_size=50)
        g_best = model.solve(problem_dict)
        results.append(g_best.target.fitness)
        print(f"fitness={g_best.target.fitness:.6e}")
    
    # Compute statistics
    results = np.array(results)
    mean_fitness = np.mean(results)
    std_fitness = np.std(results)
    min_fitness = np.min(results)
    max_fitness = np.max(results)
    median_fitness = np.median(results)
    
    print(f"\nStatistical Results:")
    print(f"  Mean fitness:   {mean_fitness:.6e}")
    print(f"  Std deviation:  {std_fitness:.6e}")
    print(f"  Minimum:        {min_fitness:.6e}")
    print(f"  Maximum:        {max_fitness:.6e}")
    print(f"  Median:         {median_fitness:.6e}")
    print(f"  Range:          {max_fitness - min_fitness:.6e}")
    print(f"  CV (Std/Mean):  {std_fitness/mean_fitness:.4f} (consistency)")


# ============================================================================
# Example 6: TrainedIMODE with Information Display
# ============================================================================

def example_6_trained_imode():
    """
    Example 6: TrainedIMODE with Algorithm Information
    
    Use the TrainedIMODE class which extends OriginalIMODE with
    additional features like information display and metadata tracking.
    """
    print("\n" + "="*80)
    print("Example 6: TrainedIMODE with Information Display")
    print("="*80)
    
    # Display algorithm information
    model = TrainedIMODE(epoch=50, pop_size=50)
    model.information()
    
    # Run optimization
    def sphere_function(solution):
        return np.sum(solution**2)
    
    problem_dict = {
        "bounds": FloatVar(n_vars=30, lb=(-100.,)*30, ub=(100.,)*30, name="x"),
        "obj_func": sphere_function,
        "minmax": "min",
    }
    
    print("Running TrainedIMODE optimization...")
    g_best = model.solve(problem_dict)
    
    print(f"\nOptimization Results:")
    print(f"  Best fitness: {g_best.target.fitness:.6e}")
    print(f"  Best solution (first 5): {g_best.solution[:5]}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("IMODE - Improved Multi-Operator Differential Evolution")
    print("Comprehensive Usage Examples")
    print("="*80)
    
    # Run all examples
    example_1_basic_usage()
    example_2_custom_problem()
    example_3_hyperparam_study()
    example_4_archive_ratio()
    example_5_multi_run()
    example_6_trained_imode()
    
    print("\n" + "="*80)
    print("All examples completed successfully!")
    print("="*80 + "\n")
