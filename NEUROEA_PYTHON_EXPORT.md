# NeuroEA Python Export

Pure Python implementation of the trained NeuroEA algorithm. Transfer learned from CEC2017 Stage 2 training (F1 → F9).

## Overview

This directory contains three approaches to using the trained NeuroEA in Python:

### 1. **neuroea_python_standalone.py** - RECOMMENDED 
   - **Zero external dependencies** (only NumPy)
   - **Pure Python implementation** of NeuroEA
   - Loads parameters from JSON file
   - Can be used standalone or integrated with Mealpy
   - Best for deployment and cross-platform use

### 2. **neuroea_python.py** 
   - Full Mealpy framework integration
   - Complete optimization wrapper
   - Can load directly from MATLAB .mat files (requires scipy)
   - Advanced features for benchmarking

### 3. **Example Usage Files**
   - `example_neuroea_usage.py` - Show how to use the optimizer
   - `export_trained_parameters_to_json.m` - MATLAB script to export parameters

## Quick Start

### Step 1: Export trained parameters from MATLAB
```matlab
% Run this in MATLAB with the workspace in PlatEMO directory:
export_trained_parameters_to_json
```

This creates: `trained_neuroea_params.json`

### Step 2: Use in Python
```python
import numpy as np
from neuroea_python_standalone import TrainedNeuroEA

# Create optimizer
optimizer = TrainedNeuroEA(epoch=100, pop_size=30)

# Now you can use it:
# - In production
# - With Mealpy framework
# - In your own optimization pipeline
```

## Architecture

The NeuroEA consists of 11 blocks:

```
Population (P)
    ↓
Tournament Selection (T1, T2, T3)
    ↓
Information Exchange (E1, E2, E3, E4)
    ↓
Crossover (C)
    ↓
Mutation (M)
    ↓
Selection (S)
    ↓
Population (P)  [cycle back]
```

## Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `epoch` | 100 | [1, 100000] | Number of generations |
| `pop_size` | 30 | [5, 10000] | Population size |
| `c1` | 0.5 | [0, 1] | Crossover rate |
| `m1` | 0.1 | [0, 1] | Mutation rate |
| `tournament_size` | 10 | [2, 100] | Tournament selection size |

## Training Configuration

The trained model was created using:

- **Stage 1**: Train on CEC2017 F1
  - Pop = 30, Generations = 100, Max FE = 3000
  - Best fitness reached: ~710

- **Stage 2**: Transfer learn on CEC2017 F9
  - Pop = 30, Generations = 100, Max FE = 3000  
  - Best fitness reached: ~0.39 (offset from optimum across 30 dimensions)

## Files Generated

### By MATLAB (export_trained_parameters_to_json.m):
- `trained_neuroea_params.json` - All parameters in portable JSON format

### Python files:
- `neuroea_python_standalone.py` - Recommended implementation
- `neuroea_python.py` - Mealpy-integrated version
- `example_neuroea_usage.py` - Usage examples

## Integration with Mealpy

### Option 1: Direct use with standalone version
```python
from neuroea_python_standalone import TrainedNeuroEA
import numpy as np
from mealpy import FloatVar

# Define problem
def objective_function(solution):
    return np.sum(solution**2)

problem_dict = {
    "bounds": FloatVar(n_vars=30, lb=(-10.,)*30, ub=(10.,)*30, name="x"),
    "obj_func": objective_function,
    "minmax": "min",
}

# Create and solve
model = TrainedNeuroEA(epoch=100, pop_size=30)
# Note: For Mealpy integration, wrap in Mealpy's Optimizer class
g_best = model.solve(problem_dict)
print(f"Best fitness: {g_best.target.fitness}")
```

### Option 2: Use full Mealpy-integrated version
```python
from neuroea_python import TrainedNeuroEA
# Requires: mealpy, scipy, numpy

model = TrainedNeuroEA(epoch=100, pop_size=30)
# Can be used directly with Mealpy benchmarking tools
```

## Performance

Benchmark results on CEC2017 problems:

| Problem | Dimension | Fitness | Status |
|---------|-----------|---------|--------|
| F1 (Stage 1) | 30 | 1.195e+03 | Training |
| F9 (Stage 2) | 30 | 3.928e-01 | Transfer trained |

## Dependencies

### Standalone version (neuroea_python_standalone.py)
- NumPy (for numerical operations)

### Full Mealpy version (neuroea_python.py)
- NumPy
- Mealpy
- SciPy (optional, for loading .mat files directly)

## Usage Examples

### Simple Optimization
```python
from neuroea_python_standalone import TrainedNeuroEA
import numpy as np

optimizer = TrainedNeuroEA(epoch=50, pop_size=30, c1=0.6, m1=0.1)

# Define problem
def sphere(x):
    return np.sum(x**2)

# Initialize population
D = 30
bounds = [(-5.0, 5.0) for _ in range(D)]
population = [np.random.uniform(-5.0, 5.0, D) for _ in range(optimizer.pop_size)]
fitness = np.array([sphere(x) for x in population])

# Optimize
for gen in range(optimizer.epoch):
    new_pop = optimizer.evolve_generation(population, fitness, bounds)
    new_fitness = np.array([sphere(x) for x in new_pop])
    
    # Survivor selection (μ+λ)
    combined = population + new_pop
    combined_fit = np.concatenate([fitness, new_fitness])
    best_idx = np.argsort(combined_fit)[:optimizer.pop_size]
    
    population = [combined[i] for i in best_idx]
    fitness = combined_fit[best_idx]
    
    if (gen+1) % 10 == 0:
        print(f"Gen {gen+1}: best={fitness.min():.6e}")

print(f"Final best: {fitness.min():.6e}")
```

### Custom Hyperparameters
```python
# Fine-tune for your problem
optimizer = TrainedNeuroEA(
    epoch=200,           # More generations
    pop_size=50,         # Larger population
    c1=0.7,              # Higher crossover rate
    m1=0.15,             # Higher mutation rate
    tournament_size=15   # Larger tournaments
)
```

## Troubleshooting

### Q: "Warning: trained_neuroea_params.json not found"
**A**: Run the MATLAB export script first:
```matlab
export_trained_parameters_to_json
```

### Q: ImportError for scipy
**A**: Use `neuroea_python_standalone.py` instead - it has no scipy dependency.

### Q: How do I load the original .mat file?
**A**: Use `neuroea_python.py` with scipy installed:
```bash
pip install scipy
```

## References

- **NeuroEA**: Neural network-guided evolutionary algorithm
- **CEC2017**: Constrained real-parameter optimization benchmark suite
- **Transfer Learning**: Parameters trained on F1, applied to F9

## Notes

- The trained model uses 30-dimensional problems with bounds [-10, 10]
- Parameters are dimensionality-agnostic and can be applied to other problem dimensions
- The architecture is preserved from the MATLAB training process
- All 11 blocks are fully implemented in Python

## Future Enhancements

- [ ] Automatic Mealpy integration wrapper
- [ ] Multi-objective optimization version
- [ ] GPU acceleration support
- [ ] Distributed/parallel population management
- [ ] Real-time visualization during optimization

## License

Licensed under PlatEMO framework terms. See main README.md for details.

---

For more information, see the main training scripts:
- `train_NeuroEA_cec2017_stage1_f1_D30.m`
- `train_NeuroEA_cec2017_stage2_f9_D30_from_f1.m`
