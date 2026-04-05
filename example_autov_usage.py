#!/usr/bin/env python3
"""
AutoV.py Usage Guide and Examples
==================================

This guide shows how to use the AutoV Python implementation with trained operators.
"""

import numpy as np
from mealpy import FloatVar
import sys
import os

# Add PlatEMO directory to path if needed
# sys.path.insert(0, '/home/jona/github/PlatEMO')

from AutoV import OriginalAutoV, TrainedAutoV, load_trained_operator_from_mat, load_training_info_from_mat

# ============================================================================
# EXAMPLE 1: Using TrainedAutoV with hardcoded operator (quick start)
# ============================================================================

def example_1_hardcoded():
    """
    Run TrainedAutoV with hardcoded operator from stage 2 training
    This is the simplest way to use the trained operator.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Using Hardcoded Trained Operator")
    print("="*80)
    
    # Define test problem: Sphere function
    def sphere_func(solution):
        return np.sum(solution**2)
    
    problem = {
        "bounds": FloatVar(n_vars=30, lb=(-100.,)*30, ub=(100.,)*30, name="x"),
        "obj_func": sphere_func,
        "minmax": "min",
    }
    
    # Create model
    print("\nCreating TrainedAutoV model with hardcoded parameters...")
    model = TrainedAutoV(epoch=50, pop_size=30, tournament_size=2)
    
    # Display information about the trained operator
    model.information()
    
    # Solve
    print("\nRunning optimization...")
    g_best = model.solve(problem)
    
    print(f"\nResults:")
    print(f"  Best fitness: {g_best.target.fitness:.6e}")
    print(f"  Solution (first 5 dims): {g_best.solution[:5]}")
    
    return g_best


# ============================================================================
# EXAMPLE 2: Loading trained operator from .mat file
# ============================================================================

def example_2_load_from_mat():
    """
    Load trained operator from MATLAB .mat file and use it
    Run this AFTER training is complete in MATLAB.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Loading Operator from .mat File")
    print("="*80)
    
    mat_filepath = 'trained_AutoV_CEC2017_F9_D30_stage2_from_f1.mat'
    
    print(f"\nAttempting to load operator from: {mat_filepath}")
    
    # Load operator matrix
    operator_params = load_trained_operator_from_mat(mat_filepath)
    
    if operator_params is None:
        print("\nWarning: Could not load operator from .mat file")
        print("Make sure to run MATLAB training first:")
        print("  train_AutoV_cec2017_stage2_f9_D30_from_f1.m")
        print("\nSkipping this example.")
        return None
    
    # Load training metadata
    training_info = load_training_info_from_mat(mat_filepath)
    print(f"\nTraining metadata loaded:")
    for key, value in training_info.items():
        print(f"  {key}: {value}")
    
    # Create model with loaded operator
    print("\nCreating TrainedAutoV with loaded operator...")
    model = TrainedAutoV(epoch=50, pop_size=30, operator_params=operator_params)
    
    # Define test problem
    def rastrigin_func(solution):
        """Rastrigin function - multimodal benchmark"""
        A = 10
        return A * len(solution) + np.sum(solution**2 - A*np.cos(2*np.pi*solution))
    
    problem = {
        "bounds": FloatVar(n_vars=30, lb=(-5.12,)*30, ub=(5.12,)*30, name="x"),
        "obj_func": rastrigin_func,
        "minmax": "min",
    }
    
    print("\nRunning optimization on Rastrigin function...")
    g_best = model.solve(problem)
    
    print(f"\nResults with loaded operator:")
    print(f"  Best fitness: {g_best.target.fitness:.6e}")
    
    return g_best


# ============================================================================
# EXAMPLE 3: Custom problem with AutoV
# ============================================================================

def example_3_custom_problem():
    """
    Use TrainedAutoV on a custom optimization problem
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Custom Problem Definition")
    print("="*80)
    
    # Define custom Rosenbrock function
    def rosenbrock(solution):
        """Rosenbrock function"""
        return sum(100.0*(solution[i+1]-solution[i]**2)**2 + (1-solution[i])**2 
                   for i in range(len(solution)-1))
    
    problem = {
        "bounds": FloatVar(n_vars=30, lb=(-5.,)*30, ub=(10.,)*30, name="x"),
        "obj_func": rosenbrock,
        "minmax": "min",
    }
    
    print("\nOptimizing Rosenbrock function with TrainedAutoV...")
    
    model = TrainedAutoV(epoch=50, pop_size=30)
    g_best = model.solve(problem)
    
    print(f"\nResults on Rosenbrock:")
    print(f"  Best fitness: {g_best.target.fitness:.6e}")
    print(f"  Expected optimum: 0.0")
    print(f"  Distance to optimum: {g_best.target.fitness:.6e}")
    
    return g_best


# ============================================================================
# EXAMPLE 4: Comparing different population sizes
# ============================================================================

def example_4_parameter_study():
    """
    Compare performance with different population sizes
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Parameter Study - Different Population Sizes")
    print("="*80)
    
    def sphere_func(solution):
        return np.sum(solution**2)
    
    problem = {
        "bounds": FloatVar(n_vars=30, lb=(-100.,)*30, ub=(100.,)*30, name="x"),
        "obj_func": sphere_func,
        "minmax": "min",
    }
    
    pop_sizes = [15, 30, 50]
    results = {}
    
    for pop_size in pop_sizes:
        print(f"\nTesting with population size: {pop_size}")
        model = TrainedAutoV(epoch=50, pop_size=pop_size)
        g_best = model.solve(problem)
        results[pop_size] = g_best.target.fitness
        print(f"  Best fitness: {g_best.target.fitness:.6e}")
    
    print("\nParameter Study Results:")
    print("-" * 40)
    for pop_size, fitness in sorted(results.items()):
        print(f"  Pop size {pop_size}: {fitness:.6e}")
    
    return results


# ============================================================================
# EXAMPLE 5: Using OriginalAutoV with custom operator
# ============================================================================

def example_5_custom_operator():
    """
    Create OriginalAutoV with custom operator parameters
    Useful for experimenting with different operators
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: OriginalAutoV with Custom Operator")
    print("="*80)
    
    # Define custom operator: all parameters set to 0.5
    custom_operator = np.ones((10, 4)) * 0.5
    custom_operator[:, 3] = 1.0 / 10  # Equal probability weights
    
    print(f"\nCustom operator matrix (10 x 4):")
    print(custom_operator)
    
    def sphere_func(solution):
        return np.sum(solution**2)
    
    problem = {
        "bounds": FloatVar(n_vars=30, lb=(-100.,)*30, ub=(100.,)*30, name="x"),
        "obj_func": sphere_func,
        "minmax": "min",
    }
    
    print("\nRunning OriginalAutoV with custom operator...")
    model = OriginalAutoV(epoch=50, pop_size=30, operator_params=custom_operator)
    g_best = model.solve(problem)
    
    print(f"\nResults with custom operator:")
    print(f"  Best fitness: {g_best.target.fitness:.6e}")
    
    return g_best


# ============================================================================
# EXAMPLE 6: Retrieving operator details
# ============================================================================

def example_6_operator_details():
    """
    Access and display detailed operator information
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Operator Details and Configuration")
    print("="*80)
    
    model = TrainedAutoV(epoch=10, pop_size=30)
    
    # Get trained parameters
    params = model.get_trained_parameters()
    print(f"\nTrained operator parameters shape: {params.shape}")
    print(f"Parameters (first 3 sets):")
    print(params[:3])
    
    # Get operator details
    details = model.get_operator_details()
    print(f"\nOperator Details:")
    for key, value in details.items():
        if key != 'parameter_matrix':  # Skip the large matrix for display
            print(f"  {key}: {value}")
    
    # Get trained configuration
    config = model.trained_config
    print(f"\nTraining Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    return model


# ============================================================================
# MAIN: Run selected examples
# ============================================================================

def main():
    """Run all examples or specific ones"""
    
    print("\n")
    print("█" * 80)
    print("█" + " "*78 + "█")
    print("█" + "AutoV.py - Automated Design of Variation Operators".center(78) + "█")
    print("█" + "Usage Examples and Demonstrations".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█" * 80)
    
    # Run all examples
    try:
        # Example 1: Hardcoded operator
        result_1 = example_1_hardcoded()
        
        # Example 2: Load from mat
        result_2 = example_2_load_from_mat()
        
        # Example 3: Custom problem
        result_3 = example_3_custom_problem()
        
        # Example 4: Parameter study
        result_4 = example_4_parameter_study()
        
        # Example 5: Custom operator
        result_5 = example_5_custom_operator()
        
        # Example 6: Operator details
        result_6 = example_6_operator_details()
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY OF EXAMPLES")
        print("="*80)
        print("\nExample 1: Hardcoded operator - Best fitness: {:.6e}".format(result_1.target.fitness))
        if result_2:
            print("Example 2: Loaded from .mat - Best fitness: {:.6e}".format(result_2.target.fitness))
        print("Example 3: Rosenbrock function - Best fitness: {:.6e}".format(result_3.target.fitness))
        print("Example 4: Parameter study completed - See results above")
        print("Example 5: Custom operator - Best fitness: {:.6e}".format(result_5.target.fitness))
        print("Example 6: Operator details - See configuration above")
        
        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
