# NeuroEA Training Setup for CEC2017 (D=30)

## Overview

This package provides a complete setup for training NeuroEA models on CEC2017 single-objective optimization problems, following the architecture and methodology from the Tian et al. paper with hyperparameters scaled for population size 30.

**Key Features:**
- Paper-faithful architecture: 11 blocks with specific structure
- Scaled hyperparameters optimized for pop=30 (not paper's pop=100)
- GA-based outer loop to tune 55 block parameters
- 3 independent trainings: one each for f1, f4, and f9
- Saved models for later inference/reuse
- Easy-to-modify settings at top of scripts

## Architecture

### Block Structure (11 blocks)
```
[P, T1, T2, T3, E1, E2, E3, E4, C, M, S]
```

Where:
- **P** = Block_Population() - Initial population (0 parameters)
- **T1, T2, T3** = Block_Tournament(60, 10) each - Tournament selection (1 param each)
- **E1-E4** = Block_Exchange(3) each - Parent exchange (3 params each)
- **C** = Block_Crossover(2, 5) - Crossover operator (30 parameters)
- **M** = Block_Mutation(5) - Mutation operator (10 parameters)
- **S** = Block_Selection(30) - Environmental selection (0 parameters)

**Total tunable parameters: 55**

### Data Flow (Adjacency Matrix)
```
P → T1, T2, T3, S
T1, T2, T3 → E1-E4 (each tournament sends 0.25 fraction)
E1-E4 → C
C → M
M → S
S → P (feedback)
```

## Usage

### 1. Training Phase

Train NeuroEA on f1:
```matlab
cd /path/to/PlatEMO
train_NeuroEA_cec2017_f1_D30
```

This will:
1. Create a GA trainer with pop_size=50 and max_evals=5000 (configurable at top of file)
2. Each GA generation: evaluate 50 candidates (inner GA population)
3. For each candidate: run 3 independent NeuroEA optimizations, average fitness
4. Save trained model to `trained_NeuroEA_CEC2017_F1_D30.mat`

Inside, the inner NeuroEA runs with:
- Population size: 30
- Max function evaluations: 3000
- Dimension: 30

Similarly for f4 and f9:
```matlab
train_NeuroEA_cec2017_f4_D30
train_NeuroEA_cec2017_f9_D30
```

### 2. Training Configuration (Optional)

Edit the following in `train_NeuroEA_cec2017_common.m` to customize:
```matlab
% Lines 16-21:
TRAINER_POP_SIZE = 50;          % Outer GA population size (paper: 50)
TRAINER_MAX_EVALS = 5000;       % Total GA evaluations (paper: 5000)
NUM_RUNS_PER_CANDIDATE = 3;     % Runs to average per candidate
SEED_BASE = 12345;              % Reproducibility

% Lines 24-27:
INNER_POP_SIZE = 30;            % NeuroEA population size
INNER_MAX_FE = 3000;            % NeuroEA max function evals
INNER_D = D;                    % Dimension (passed as argument)
```

### 3. Inference Phase: Using Trained Models

Load a trained model and run on a test problem:
```matlab
% Test a trained f1 model on different problem
[fitness, pop, details] = load_trained_NeuroEA_and_run(...
    'trained_NeuroEA_CEC2017_F1_D30.mat', ...
    @CEC2017_F4);
```

Run with custom settings:
```matlab
[fitness, pop, details] = load_trained_NeuroEA_and_run(...
    'trained_NeuroEA_CEC2017_F1_D30.mat', ...
    @CEC2017_F9, ...
    30,      % Dimension
    30,      % Population size (for display/info)
    3000,    % Max function evaluations
    12345);  % Random seed
```

## Scripts Included

### Training Scripts
- **train_NeuroEA_cec2017_f1_D30.m** - Train on F1
- **train_NeuroEA_cec2017_f4_D30.m** - Train on F4
- **train_NeuroEA_cec2017_f9_D30.m** - Train on F9

### Core Training Logic
- **train_NeuroEA_cec2017_common.m** - Main GA trainer (called by above)
  - Implements outer GA loop
  - Evaluates candidates by running NeuroEA
  - Saves history and best results

### Utilities
- **train_setup_utils.m** - Helper functions
  - `create_blocks_graph()` - Creates the 11-block architecture
  - `create_ga_trainer()` - Configuration helpers

### Inference
- **load_trained_NeuroEA_and_run.m** - Load model and run on new problem
  - Automatically reconstructs the architecture
  - Applies trained parameters
  - Runs on any CEC2017 problem

## Output Format

Saved .mat files contain:
```
Blocks              - Array of Block objects (template)
Graph               - 11x11 adjacency matrix
best_params         - Best 55 parameter values found
best_fitness        - Best fitness achieved during training
trainer_history     - Struct with generation/evaluation history
seeds               - Random seeds used for reproducibility
PROBLEM_NAME        - Training problem name ("CEC2017_F1", etc.)
PROBLEM_CLASS       - Training problem class handle
DIMENSION           - Training dimension (30)
```

## Important Notes

### Parameter Scaling
The hyperparameters **are not** the paper's values. They've been carefully scaled:

| Block Type | Paper (pop=100) | Scaled (pop=30) | Notes |
|---|---|---|---|
| Tournament | Block_Tournament(100,10)* | Block_Tournament(60,10) | nParents scaled 100→60 |
| Exchange | Block_Exchange(3) | Block_Exchange(3) | No scaling needed |
| Crossover | Block_Crossover(2,5) | Block_Crossover(2,5) | No scaling |
| Mutation | Block_Mutation(5) | Block_Mutation(5) | No scaling |
| Selection | Block_Selection(100) | Block_Selection(30) | Direct pop size |

*Estimated from paper description

### Reproducibility
- Each training script uses SEED_BASE = 12345
- GA generations logged with statistics
- All seeds saved in output .mat file
- Re-run with same model file: load and run_trained with explicit seed

### Assumptions Printed at Runtime
- Population size 30 vs paper size 100
- CEC2017 problem class names verified at runtime
- Block constructor signatures confirmed
- Graph structure and data flow validated

## Example Workflow

```matlab
% 1. Train three separate models
train_NeuroEA_cec2017_f1_D30   % ~1-2 hours depending on hardware
train_NeuroEA_cec2017_f4_D30
train_NeuroEA_cec2017_f9_D30

% 2. Test transfer learning: f1-trained model on f4, f9
[fitness_f4, ~] = load_trained_NeuroEA_and_run(...
    'trained_NeuroEA_CEC2017_F1_D30.mat', @CEC2017_F4);
[fitness_f9, ~] = load_trained_NeuroEA_and_run(...
    'trained_NeuroEA_CEC2017_F1_D30.mat', @CEC2017_F9);
fprintf('F1 model on F4: %.6e\n', fitness_f4);
fprintf('F1 model on F9: %.6e\n', fitness_f9);

% 3. Run f4-trained model
[fitness, pop, details] = load_trained_NeuroEA_and_run(...
    'trained_NeuroEA_CEC2017_F4_D30.mat', @CEC2017_F4);
fprintf('F4 model on F4: %.6e\n', fitness);
```

## File Locations

```
/home/jona/github/PlatEMO/
├── train_NeuroEA_cec2017_f1_D30.m
├── train_NeuroEA_cec2017_f4_D30.m
├── train_NeuroEA_cec2017_f9_D30.m
├── train_NeuroEA_cec2017_common.m
├── train_setup_utils.m
├── load_trained_NeuroEA_and_run.m
├── NEUROEA_TRAINING_README.md  (this file)
├── trained_NeuroEA_CEC2017_F1_D30.mat  (after training)
├── trained_NeuroEA_CEC2017_F4_D30.mat
└── trained_NeuroEA_CEC2017_F9_D30.mat
```

## Troubleshooting

**Q: "Cannot find block class Block_Tournament"**
A: Make sure you're in the PlatEMO directory and the class path includes the NeuroEA algorithms.

**Q: "Dimension mismatch"**
A: CEC2017 problems have min dimension 10. Default your D ≥ 10. Script uses D=30.

**Q: Training is very slow**
A: This is expected. 5000 GA evaluations × 3 runs each = 15,000 NeuroEA runs.
   With D=30, maxFE=3000 each, expect 1-4 hours on modern hardware.
   To speed up: reduce TRAINER_MAX_EVALS or NUM_RUNS_PER_CANDIDATE in common.m.

**Q: "Error: CEC2017_F1 not found"**
A: The problems are in `/PlatEMO/Problems/Single-objective optimization/CEC 2017/`
   Verify they exist and are accessible via `@CEC2017_F1` syntax.

## Paper Reference

This implementation is based on:
> Y. Tian, X. Qi, S. Yang, C. He, K. C. Tan, Y. Jin, and X. Zhang.
> "A universal framework for automatically generating single- and
> multi-objective evolutionary algorithms."
> IEEE Transactions on Evolutionary Computation, 2025.

The architecture and hyperparameters selection follow the paper's design principles,
but are scaled to the user's specified population size of 30.
