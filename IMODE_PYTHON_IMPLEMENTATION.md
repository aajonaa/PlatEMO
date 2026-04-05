#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IMODE Python Implementation Documentation
=============================================

Created: 06/04/2026
Adaptation: MATLAB IMODE.m → Python mealpy framework
Status: Complete and tested

================================================================================
TABLE OF CONTENTS
================================================================================

1. Overview
2. Algorithm Details
3. Class Hierarchy
4. Method Reference
5. Hyperparameter Guide
6. Usage Examples
7. Performance Tips
8. Algorithm Comparison
9. Troubleshooting
10. References

================================================================================
1. OVERVIEW
================================================================================

IMODE (Improved Multi-Operator Differential Evolution) is a differential
evolution variant that adaptively combines three mutation operators with
adaptive control parameters (CR and F values) and archive-based diversity
maintenance.

Key Features:
  - Three complementary mutation operators (DE/current-to-pbest/2 variants)
  - Adaptive crossover rate (CR) and scaling factor (F)
  - Archive-based diversity preservation
  - Adaptive population size reduction
  - Success-based operator probability adaptation

Framework: mealpy (Evolutionary Algorithms Library)
  - Based on mealpy.optimizer.Optimizer base class
  - Compatible with mealpy problem definitions
  - Integrates with mealpy solution management

Repository Structure:
  - IMODE.py: Main algorithm implementation
  - example_imode_usage.py: Comprehensive 6-example tutorial
  - example_imode_mealpy.py: Quick-start 3-option guide
  - IMODE_PYTHON_IMPLEMENTATION.md: This documentation file

================================================================================
2. ALGORITHM DETAILS
================================================================================

2.1 Mutation Operators (Three-Operator Strategy)
================================================

IMODE uses three complementary mutation operators with adaptive selection:

Operator 1: DE/current-to-pbest/2
  Formula: v = x + F*(p_best - x) + F*(r1 - r2)
  
  Where:
    x: Current individual
    p_best: Best individual from top 25% (pbest set)
    r1, r2: Random individuals from population/archive
    F: Scaling factor (search step size)
  
  Characteristics:
    - Uses current vector and pbest information
    - Combines two differential vectors
    - Good for exploitative search
    - Adaptive based on success


Operator 2: DE/current-to-pbest-archive/2
  Formula: v = x + F*(p_best - x) + F*(r1 - r3)
  
  Where:
    x: Current individual
    p_best: Best individual from top 25%
    r1: Random from current population
    r3: Random from current population (different pool)
    F: Scaling factor
  
  Characteristics:
    - Similar to Operator 1 but different vector pool
    - Provides search diversity exploration
    - Archive-aware variant


Operator 3: DE/rand-pbest/2
  Formula: v = F*(r1 + p_best - r3)
  
  Where:
    r1: Random individual
    p_best: Best from pbest set
    r3: Different random individual
    F: Scaling factor
  
  Characteristics:
    - No current vector (pure differential)
    - Maximum exploration capability
    - Best for escaping local optima


2.2 Adaptive Control Parameters
================================

Crossover Rate (CR) Adaptation:
  - Memory-based: MCR[i] stores successful CR values
  - Generated from: CR ~ N(MCR[i], sqrt(0.1))
  - Updated by: Weighted average of successful CR values
  - Range: CR ∈ [0, 1]
  
  Formula for update:
    MCR[k] = sum(w_i * CR_i^2) / sum(w_i * CR_i)
    where w_i = success_rate[i] / sum(success_rates)


Scaling Factor (F) Adaptation:
  - Memory-based: MF[i] stores successful F values
  - Generated from: F ~ Cauchy(MF[i], sqrt(0.1))
  - Updated by: Weighted average of successful F values
  - Range: F ∈ (0.001, 1.0]
  
  Formula for update:
    MF[k] = sum(w_i * F_i^2) / sum(w_i * F_i)
    where weights based on improvement magnitude


2.3 Operator Probability Adaptation
====================================

Each operator's selection probability is adapted based on success:

  MOP[i] = success_rate[i] / sum(success_rates)
  
  Where:
    success_rate[i] = average improvement for operator i
    Clipped to [0.1, 0.9] for exploration balance


2.4 Population Size Reduction
=============================

Population size decreases linearly from initial N to minimum minN:

  N(t) = max(minN, ceil((minN - N0) * FE / maxFE) + N0)
  
  Where:
    N0: Initial population size
    minN: Minimum allowed population size
    FE: Current function evaluations
    maxFE: Maximum allowed function evaluations
  
  Benefits:
    - Computational efficiency (fewer evaluations later)
    - Transition from exploration to exploitation
    - Adaptive budget allocation


2.5 Archive Management
======================

Diversity archive stores successful old solutions:

  - Size: Archive = ceil(aRate * current_population_size)
  - Updated: Old solutions replaced by successful offspring
  - Used for: Selecting random individuals (r2, r3)
  - Benefits: Prevents premature convergence, increases diversity


2.6 Crossover Modes
===================

Two different crossover strategies with probability control:

Uniform Crossover (40% probability):
  Site = random(N, D) > CR
  Offspring[Site] = Parent[Site]  (keep original)
  
  Characteristics:
    - Each dimension independently controlled
    - More exploration
    - Better for separable problems


Segmented Crossover (60% probability):
  Select random segment length and position
  Copy continuous segment from original to offspring
  
  Characteristics:
    - Preserves building blocks
    - Better for non-separable problems
    - Reduces dimensionality of CR effect


================================================================================
3. CLASS HIERARCHY
================================================================================

3.1 OriginalIMODE
=================

Base class implementing IMODE algorithm with configurable operators.

Inheritance:
  OriginalIMODE(Optimizer)  <- mealpy.optimizer.Optimizer

Key Methods:
  - __init__(**kwargs): Initialize algorithm with hyperparameters
  - initialize_variables(): Set up memory and archive
  - generate_agent(solution): Create and evaluate agent
  - update_population_size(): Calculate current N based on budget
  - select_parent_set(population, fraction): Get pbest set
  - generate_cr_values(N): Sample CR from memory
  - generate_f_values(N): Sample F from memory
  - select_operator(N): Choose operators for each individual
  - crossover(pop_dec, off_dec, cr_values): Apply recombination
  - de_current_pbest_2(...): Operator 1 implementation
  - de_current_pbest_archive_2(...): Operator 2 implementation
  - de_rand_pbest_2(...): Operator 3 implementation
  - update_memory(...): Adapt CR and F values
  - update_operator_prob(...): Adapt operator probabilities
  - evolve(epoch): Main evolution loop

Properties:
  - self.MCR: Crossover rate memory (size: 20*D)
  - self.MF: Scaling factor memory (size: 20*D)
  - self.MOP: Operator probabilities (3 elements)
  - self.archive: Diversity archive
  - self.k: Current memory index


3.2 TrainedIMODE
================

Extended class with training metadata and information display.

Inheritance:
  TrainedIMODE(OriginalIMODE)

Key Methods:
  - __init__(**kwargs): Same parameters as OriginalIMODE
  - information(): Display algorithm configuration and details

Properties:
  - self.trained_config: Metadata dictionary
    {
      'algorithm': 'IMODE',
      'framework': 'mealpy',
      'status': 'default_parameters'
    }

Usage:
  model = TrainedIMODE(epoch=100, pop_size=50)
  model.information()  # Display detailed info
  best = model.solve(problem_dict)


================================================================================
4. METHOD REFERENCE
================================================================================

4.1 Public Methods
==================

__init__(epoch, pop_size, minN=4, aRate=2.6, cr_mean=0.2, f_mean=0.2)
  Initialize IMODE algorithm
  
  Parameters:
    - epoch (int): [1, 100000] Maximum iterations, default=100
    - pop_size (int): [5, 10000] Population size, default=50
    - minN (int): [2, 20] Minimum population size, default=4
    - aRate (float): [1.0, 5.0] Archive size ratio, default=2.6
    - cr_mean (float): (0, 1.0) Initial mean CR, default=0.2
    - f_mean (float): (0, 1.0) Initial mean F, default=0.2
  
  Returns: None


initialize_variables()
  Set up algorithm-specific variables before optimization
  
  Initializes:
    - MCR: Crossover rate memory
    - MF: Scaling factor memory
    - MOP: Operator probabilities
    - archive: Empty list
  
  Called by: Optimizer base class automatically


evolve(epoch)
  Main evolution loop for one generation
  
  Parameters:
    - epoch (int): Current epoch number
  
  Process:
    1. Update population size
    2. Trim population if needed
    3. Maintain archive
    4. Generate CR and F values
    5. Select operators
    6. Generate offspring via differential evolution
    7. Apply crossover
    8. Evaluate offspring
    9. Comparison and selection
    10. Update memory and probabilities
  
  Called by: Optimizer base class in main solve loop


4.2 Protected Methods
=====================

generate_empty_agent(solution)
  Create Agent without fitness evaluation
  
  Parameters:
    - solution (np.ndarray): Decision variables
  
  Returns: Agent object


generate_agent(solution)
  Create Agent with fitness evaluation
  
  Parameters:
    - solution (np.ndarray): Decision variables
  
  Returns: Agent with target (fitness) computed


amend_solution(solution)
  Ensure solution respects problem bounds
  
  Parameters:
    - solution (np.ndarray): Potentially out-of-bounds solution
  
  Returns: Amended solution within bounds


update_population_size()
  Calculate population size for current generation
  
  Formula: N(t) = max(minN, ceil((minN - N0) * FE / maxFE) + N0)
  
  Returns: int, updated population size


select_parent_set(population, fraction)
  Select pbest set from current population
  
  Parameters:
    - population (list): List of Agent objects
    - fraction (float): Top fraction to select (e.g., 0.25)
  
  Returns: np.ndarray of shape (pbest_size, D)


generate_cr_values(N)
  Sample CR values from memory-based normal distribution
  
  Parameters:
    - N (int): Number of CR values to generate
  
  Formula: CR ~ N(MCR[i], sqrt(0.1)), clipped to [0, 1]
  
  Returns: np.ndarray of shape (N,)


generate_f_values(N)
  Sample F values from memory-based Cauchy distribution
  
  Parameters:
    - N (int): Number of F values to generate
  
  Formula: F ~ Cauchy(MF[i], sqrt(0.1)), clipped to (0.001, 1]
  
  Returns: np.ndarray of shape (N,)


select_operator(N)
  Assign operators to individuals based on adaptive probabilities
  
  Parameters:
    - N (int): Population size
  
  Returns: list of 3 lists, each containing indices for that operator


crossover(pop_dec, off_dec, cr_values)
  Apply recombination operation
  
  Parameters:
    - pop_dec (np.ndarray): Parent decisions
    - off_dec (np.ndarray): Mutant decisions
    - cr_values (np.ndarray): Crossover rates
  
  Returns: np.ndarray of recombined solutions


de_current_pbest_2(x, pbest, r1, r2, f)
  Implement DE/current-to-pbest/2 mutation
  
  Formula: v = x + F*(pbest - x) + F*(r1 - r2)
  
  Returns: Mutant vector


de_current_pbest_archive_2(x, pbest, r1, r3, f)
  Implement DE/current-to-pbest-archive/2 mutation
  
  Formula: v = x + F*(pbest - x) + F*(r1 - r3)
  
  Returns: Mutant vector


de_rand_pbest_2(pbest, r1, r3, f)
  Implement DE/rand-pbest/2 mutation
  
  Formula: v = F*(r1 + pbest - r3)
  
  Returns: Mutant vector


update_memory(success_indices, cr_values, f_values, success_rates)
  Adapt CR and F memories based on successful solutions
  
  Parameters:
    - success_indices (np.ndarray): Indices of successful offspring
    - cr_values (np.ndarray): Current CR values
    - f_values (np.ndarray): Current F values
    - success_rates (np.ndarray): Improvement magnitudes
  
  Updates: self.MCR[self.k], self.MF[self.k], self.k


update_operator_prob(success_indices, op_groups, success_rates)
  Adapt operator selection probabilities
  
  Parameters:
    - success_indices (np.ndarray): Successful solution indices
    - op_groups (list): Operator assignment
    - success_rates (np.ndarray): Improvement magnitudes
  
  Updates: self.MOP


4.3 Inherited Methods
======================

solve(problem_dict)
  Main optimization loop (from Optimizer base class)
  
  Parameters:
    - problem_dict (dict): Problem definition with 'bounds', 'obj_func', 'minmax'
  
  Returns: Agent with best solution found


get_target(solution)
  Evaluate objective function (from Optimizer base class)
  
  Parameters:
    - solution (np.ndarray): Decision variables
  
  Returns: Target object with fitness


================================================================================
5. HYPERPARAMETER GUIDE
================================================================================

5.1 epoch (Maximum Iterations)
==============================
Role: Controls total computational budget

Recommended Range: [50, 200]
  - Small (50):   Quick testing, fast iteration
  - Medium (100): Balanced, good for single-objective
  - Large (200):  For difficult landscapes

Problem-dependent:
  - Unimodal/simple: 50 epochs sufficient
  - Multimodal: 100-150 epochs recommended
  - Highly complex: 150-200+ epochs

Trade-off: More epochs = better solutions but longer runtime


5.2 pop_size (Population Size)
==============================
Role: Controls population diversity and parallelism

Recommended Range: [20, 200]
  - Small (20-30):    Quick execution, less exploration
  - Medium (50-100):  Default, good balance
  - Large (100+):     More diversity, slower per generation

Guidelines:
  - Dimension rule: pop_size ≥ 10 * dimension
  - Diversity rule: pop_size ≥ 50 for complex problems
  - Budget rule: Total FE = epoch * pop_size * ~3 (evals per gen)

Typical configurations:
  - D=10: pop_size=30-50
  - D=30: pop_size=50-100
  - D=100: pop_size=100-200


5.3 minN (Minimum Population Size)
==================================
Role: Sets floor for population size reduction

Recommended Range: [2, 20]
  - Small (2-5):   Aggressive reduction, fast convergence
  - Medium (4-8):  Balanced reduction
  - Large (8-20):  Slow reduction, more exploration

Guidance:
  - minN >= 2 (always)
  - minN <= min(pop_size/2, 25)
  - Higher minN = longer exploration phase
  - Lower minN = faster convergence

Typical choice: minN ≈ 0.1 * initial_pop_size


5.4 aRate (Archive Size Ratio)
==============================
Role: Controls diversity through archive management

Recommended Range: [1.0, 4.0]
  - Low (1.0-1.5):   Less diversity, faster convergence
  - Medium (2.0-3.0): Balanced, recommended
  - High (3.0-4.0):  More diversity, wider search

Guidance:
  - Simple problems: aRate = 1.5-2.0
  - Complex problems: aRate = 2.5-3.5
  - Multimodal problems: aRate = 3.0-4.0

Effect:
  - Higher aRate = larger archive = more diversity = slower convergence
  - Lower aRate = smaller archive = faster convergence = risk of local optima


5.5 cr_mean and f_mean (Initial Memories)
==========================================
Role: Set initial CR and F adaptation values

Recommended Range:
  - cr_mean: [0.05, 0.5]
  - f_mean: [0.05, 0.5]

Guidance:
  - cr_mean = 0.2 is well-tested default
  - f_mean = 0.2 is well-tested default
  - Higher f_mean = larger initial steps
  - Lower f_mean = smaller initial steps

Typical combinations:
  - Balanced: cr_mean=0.2, f_mean=0.2 (default)
  - Explorative: cr_mean=0.3, f_mean=0.3
  - Exploitative: cr_mean=0.1, f_mean=0.1


5.6 Preset Configurations
==========================

Configuration A: General Purpose (Default)
  epoch=100, pop_size=50, minN=4, aRate=2.6, cr_mean=0.2, f_mean=0.2
  Use for: Unknown problems, balanced approach
  Budget: ~15000 function evaluations

Configuration B: Large-scale Optimization
  epoch=200, pop_size=100, minN=10, aRate=2.6, cr_mean=0.2, f_mean=0.2
  Use for: Difficult, multimodal problems
  Budget: ~60000 function evaluations

Configuration C: Quick Testing
  epoch=50, pop_size=30, minN=4, aRate=1.5, cr_mean=0.3, f_mean=0.3
  Use for: Fast prototyping, quick benchmarking
  Budget: ~4500 function evaluations

Configuration D: Fine-tuning
  epoch=150, pop_size=75, minN=6, aRate=3.0, cr_mean=0.15, f_mean=0.15
  Use for: Final optimization, high precision needed
  Budget: ~33750 function evaluations


================================================================================
6. USAGE EXAMPLES
================================================================================

6.1 Simple Usage
================

from mealpy import FloatVar
from IMODE import OriginalIMODE
import numpy as np

def sphere(solution):
    return np.sum(solution**2)

problem = {
    "bounds": FloatVar(n_vars=30, lb=(-100.,)*30, ub=(100.,)*30),
    "obj_func": sphere,
    "minmax": "min",
}

model = OriginalIMODE(epoch=100, pop_size=50)
best = model.solve(problem)
print(f"Best fitness: {best.target.fitness}")


6.2 Custom Problem
===================

from mealpy import FloatVar
from IMODE import OriginalIMODE
import numpy as np

def rosenbrock(solution):
    return sum(100*(solution[i+1] - solution[i]**2)**2 + 
              (1 - solution[i])**2 for i in range(len(solution)-1))

problem = {
    "bounds": FloatVar(n_vars=20, lb=(-2.,)*20, ub=(2.,)*20),
    "obj_func": rosenbrock,
    "minmax": "min",
}

# Tune for harder problem
model = OriginalIMODE(epoch=200, pop_size=100, minN=10, aRate=3.0)
best = model.solve(problem)


6.3 Algorithm Information
==========================

from IMODE import TrainedIMODE

model = TrainedIMODE(epoch=100, pop_size=50)
model.information()  # Print detailed algorithm info

# Then run optimization
problem = {...}
best = model.solve(problem)


6.4 Run statistics
==================

from IMODE import OriginalIMODE
import numpy as np

results = []
for run in range(10):
    model = OriginalIMODE(epoch=100, pop_size=50)
    best = model.solve(problem)
    results.append(best.target.fitness)

results = np.array(results)
print(f"Mean: {np.mean(results):.6e}")
print(f"Std:  {np.std(results):.6e}")
print(f"Min:  {np.min(results):.6e}")
print(f"Max:  {np.max(results):.6e}")


================================================================================
7. PERFORMANCE TIPS
================================================================================

7.1 Computational Efficiency
=============================

1. Population Size Scaling:
   - Don't use pop_size > 2*dimension for simple problems
   - Can use pop_size = 3-5*dimension for very hard problems
   - Larger populations = diminishing returns after pop_size > 100

2. Epoch Budgeting:
   - Total FE ≈ epoch × pop_size × 3 (rough estimate)
   - For 100K budget: use epoch=50-100 with pop_size=50
   - For 10K budget: use epoch=50 with pop_size=30

3. Memory Efficiency:
   - Memory complexity: O(pop_size × D + archive_size × D)
   - Archive size = ceil(aRate × pop_size)
   - Total: O((1 + aRate) × pop_size × D)


7.2 Solution Quality Tips
==========================

1. Problem Analysis First:
   - Is it separable? (use smaller pop_size)
   - Is it multimodal? (use larger pop_size, higher aRate)
   - Is it constrained? (verify bounds are tight)

2. Parameter Tuning:
   - Start with default config, measure baseline
   - If not converging: increase pop_size or epoch
   - If converging too slowly: decrease minN
   - If getting stuck: increase aRate (more diversity)

3. Post-processing:
   - Run local search on final solution
   - Use results as initialization for deterministic solver
   - Consider ensemble of multiple runs


7.3 Debugging and Validation
=============================

1. Check convergence:
   - Plot fitness over time
   - Look for monotonic improvement
   - Plateau indicates convergence reached

2. Verify implementation:
   - Run on known benchmark (Sphere, Rosenbrock)
   - Compare fitness values across functions
   - Test with very small population (N=5)

3. Monitor adaptation:
   - Print MCR, MF values over time
   - Check if operators adapt to problem
   - Verify archive size constraints


================================================================================
8. ALGORITHM COMPARISON
================================================================================

IMODE vs Standard DE:
  - Operators: 3 (IMODE) vs 1 (standard DE)
  - Parameter adaptation: Explicit (IMODE) vs Fixed (DE)
  - Archiving: Yes (IMODE) vs No (standard DE)
  - Expected advantage: Better adaptation, higher success rate

IMODE vs jDE (Self-adaptive DE):
  - Parameter adaptation: Memory-based (IMODE) vs Individual (jDE)
  - Operator selection: Adaptive (IMODE) vs Fixed (jDE)
  - Population reduction: Yes (IMODE) vs No (jDE)
  - IMODE typically better on CEC benchmarks

IMODE vs CMA-ES:
  - Problem type: Black-box (both)
  - Covariance matrix: No (IMODE) vs Yes (CMA-ES)
  - Population size: 50-100 (IMODE) vs 10-20 (CMA-ES)
  - CMA-ES better for continuous optimization
  - IMODE better for multimodal problems


================================================================================
9. TROUBLESHOOTING
================================================================================

9.1 Solution Not Improving
===========================

Problem: Fitness plateaus early
Solutions:
  1. Increase pop_size (less chance of premature convergence)
  2. Increase aRate (more diversity from archive)
  3. Increase epoch (more generations for search)
  4. Verify bounds are realistic

Problem: Very slow convergence
Solutions:
  1. Decrease minN (faster population reduction)
  2. Decrease aRate (less diversity overhead)
  3. Decrease epoch (fewer slow generations)
  4. Reduce dimension if possible


9.2 Out of Memory Errors
=========================

Problem: MemoryError during large optimization
Solutions:
  1. Reduce pop_size
  2. Reduce aRate (smaller archive)
  3. Reduce epoch (fewer generations)
  4. Reduce problem dimension
  5. Use dtype=float32 if precision allows


9.3 Unexpected Results
======================

Problem: Results vary widely between runs
Solutions:
  1. Increase epoch and pop_size for stability
  2. Run multiple times and take average
  3. Check random seed (set via problem_dict)

Problem: Different results from MATLAB version
Solutions:
  1. Python/MATLAB use different random libraries
  2. Same algorithms may produce different results
  3. Compare on averages, not single runs
  4. Verify both implementations give consistent trends


================================================================================
10. REFERENCES
================================================================================

[1] K. M. Sallam, S. M. Elsayed, R. K. Chakrabortty, and M. J. Ryan. 
    "Improved multi-operator differential evolution algorithm for solving 
    unconstrained problems." IEEE Congress on Evolutionary Computation (CEC), 2020.

[2] Storn, Rainer, and Kenneth Price. 
    "Differential evolution–a simple and efficient heuristic for global 
    optimization over continuous spaces." Journal of global optimization 11.4 (1997): 341-359.

[3] Zhang, Jing, and Arthur C. Sanderson. 
    "Adaptive differential evolution with optional external archive." 
    IEEE Transactions on Evolutionary Computation 13.5 (2009): 945-958.

[4] Das, Swagatam, and Ponnuthurai Nagaratnam Suganthan. 
    "Differential evolution: A survey of the state-of-the-art." 
    IEEE Transactions on Evolutionary Computation 15.1 (2011): 4-31.

Git Repository:
  - IMODE.py: Full algorithm implementation
  - example_imode_usage.py: 6 comprehensive examples
  - example_imode_mealpy.py: 3 quick-start options
  - Test suite: Sphere, Rosenbrock, Rastrigin, Ackley functions

================================================================================
End of Documentation
================================================================================
"""

if __name__ == "__main__":
    print(__doc__)
