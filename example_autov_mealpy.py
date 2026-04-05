#!/usr/bin/env python3
"""
example_autov_mealpy.py - Simple example of using AutoV with mealpy
====================================================================

This example shows the most straightforward way to use the trained AutoV operator
with the mealpy framework for continuous optimization.

Requirements:
    pip install mealpy scipy numpy

Usage:
    python example_autov_mealpy.py
"""

import numpy as np
from mealpy import FloatVar
import sys
import os

# Ensure AutoV can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from AutoV import TrainedAutoV, load_trained_operator_from_mat


def benchmark_sphere():
    """Sphere function: f(x) = sum(x_i^2), optimum at x=0, f=0"""
    def func(solution):
        return np.sum(solution**2)
    
    return {
        "bounds": FloatVar(n_vars=30, lb=(-100.,)*30, ub=(100.,)*30, name="x"),
        "obj_func": func,
        "minmax": "min",
    }


def benchmark_rosenbrock():
    """Rosenbrock function: f(x) = sum(100(x_{i+1}-x_i^2)^2 + (1-x_i)^2)"""
    def func(solution):
        return sum(100.0*(solution[i+1]-solution[i]**2)**2 + (1-solution[i])**2 
                   for i in range(len(solution)-1))
    
    return {
        "bounds": FloatVar(n_vars=30, lb=(-5.,)*30, ub=(10.,)*30, name="x"),
        "obj_func": func,
        "minmax": "min",
    }


def benchmark_rastrigin():
    """Rastrigin function: f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))"""
    def func(solution):
        A = 10
        return A * len(solution) + np.sum(solution**2 - A*np.cos(2*np.pi*solution))
    
    return {
        "bounds": FloatVar(n_vars=30, lb=(-5.12,)*30, ub=(5.12,)*30, name="x"),
        "obj_func": func,
        "minmax": "min",
    }


def main():
    """
    Main function demonstrating AutoV usage
    """
    
    print("\n" + "="*80)
    print("AutoV with mealpy - Simple Usage Example")
    print("="*80)
    
    # ========================================================================
    # OPTION 1: Using hardcoded trained operator (default)
    # ========================================================================
    
    print("\n[Option 1] Using hardcoded trained operator")
    print("-" * 80)
    
    problem = benchmark_sphere()
    
    print("Creating TrainedAutoV model with hardcoded stage 2 operator...")
    model = TrainedAutoV(epoch=100, pop_size=30, tournament_size=2)
    
    print("Solving Sphere function (n_vars=30)...")
    g_best = model.solve(problem)
    
    print(f"\nResults with hardcoded operator:")
    print(f"  Best fitness: {g_best.target.fitness:.6e}")
    print(f"  Solution (first 5 dims): {g_best.solution[:5]}")
    
    # ========================================================================
    # OPTION 2: Loading operator from trained .mat file
    # ========================================================================
    
    print("\n[Option 2] Loading operator from .mat file")
    print("-" * 80)
    
    mat_filepath = 'trained_AutoV_CEC2017_F9_D30_stage2_from_f1.mat'
    
    if os.path.exists(mat_filepath):
        print(f"Loading operator from: {mat_filepath}")
        operator_params = load_trained_operator_from_mat(mat_filepath)
        
        if operator_params is not None:
            print("Creating model with loaded operator...")
            model_loaded = TrainedAutoV(epoch=100, pop_size=30, operator_params=operator_params)
            
            print("Solving Rosenbrock function...")
            problem2 = benchmark_rosenbrock()
            g_best2 = model_loaded.solve(problem2)
            
            print(f"\nResults with loaded operator:")
            print(f"  Best fitness: {g_best2.target.fitness:.6e}")
    else:
        print(f"Note: {mat_filepath} not found.")
        print(f"To use a loaded operator, first run MATLAB training:")
        print(f"  train_AutoV_cec2017_stage2_f9_D30_from_f1.m")
    
    # ========================================================================
    # OPTION 3: Testing on different benchmarks
    # ========================================================================
    
    print("\n[Option 3] Testing on multiple benchmarks")
    print("-" * 80)
    
    benchmarks = {
        "Sphere": benchmark_sphere(),
        "Rosenbrock": benchmark_rosenbrock(),
        "Rastrigin": benchmark_rastrigin(),
    }
    
    print("Running TrainedAutoV on different benchmark functions...")
    
    results = {}
    for name, problem in benchmarks.items():
        print(f"\n  Testing on {name}...", end=" ", flush=True)
        model = TrainedAutoV(epoch=50, pop_size=30)
        g_best = model.solve(problem)
        results[name] = g_best.target.fitness
        print(f"fitness = {g_best.target.fitness:.6e}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\nBenchmark Results (50 epochs, pop_size=30):")
    for name, fitness in results.items():
        print(f"  {name:15s}: {fitness:.6e}")
    
    print("\n" + "="*80)
    print("Key Points:")
    print("  ✓ TrainedAutoV includes hardcoded stage 2 trained operator")
    print("  ✓ Can load custom operators from .mat files")
    print("  ✓ Compatible with mealpy for continuous optimization")
    print("  ✓ TSRI operator maintains translation/scale/rotation invariance")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
