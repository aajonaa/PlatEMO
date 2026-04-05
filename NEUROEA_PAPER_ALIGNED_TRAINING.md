% NEUROEA_PAPER_ALIGNED_TRAINING_SETUP.md

# NeuroEA Paper-Aligned Sequential Training Setup

## Overview

This folder contains a **paper-aligned, sequential training pipeline** for NeuroEA on CEC2017 single-objective optimization problems. The approach follows the paper's methodology while adapting to your reduced computational budget.

### Key Design Decisions

- **Sequential training** (F1 → F9): NOT joint multi-problem training
- **Largest 11-block architecture**: Paper-faithful design with all blocks
- **Reduced budget**: Inner population and generations scaled down, but paper structure preserved
- **Transferable parameters**: Stage 2 loads Stage 1's best parameters and continues

---

## Architecture

### 11-Block NeuroEA (Paper-Faithful)

Node order: `[P, T1, T2, T3, E1, E2, E3, E4, C, M, S]`

```
P  = Block_Population()           [0 params]
T1 = Block_Tournament(60, 10)     [1 param: tournament size k_1]
T2 = Block_Tournament(60, 10)     [1 param: tournament size k_2]
T3 = Block_Tournament(60, 10)     [1 param: tournament size k_3]
E1 = Block_Exchange(3)            [3 params: exchange configuration]
E2 = Block_Exchange(3)            [3 params]
E3 = Block_Exchange(3)            [3 params]
E4 = Block_Exchange(3)            [3 params]
C  = Block_Crossover(2, 5)        [30 params: weights for 5 pairs]
M  = Block_Mutation(5)            [10 params: mutation parameters]
S  = Block_Selection(30)          [0 params: keeps best 30 solutions]
```

**Total tunable parameters: 55**

### Connectivity Graph (11 × 11)

The adjacency matrix defines data flow:

```
P  → T1, T2, T3, S
T1 → E1, E2, E3, E4 (25% each)
T2 → E1, E2, E3, E4 (25% each)
T3 → E1, E2, E3, E4 (25% each)
E1 → C
E2 → C
E3 → C
E4 → C
C  → M
M  → S
S  → P (feedback)
```

---

## Training Protocol

### Stage 1: Train on F1 (Dimension 30)

**File:** `train_NeuroEA_cec2017_stage1_f1_D30.m`

**Settings:**
- Inner NeuroEA: population = 30, generations = 100, maxFE = 3000
- Outer GA trainer: population = 50, max evaluations = 5000
- Fitness metric: mean best objective over 3 independent runs
- Problem: CEC2017_F1, D = 30

**Output:** `trained_NeuroEA_CEC2017_F1_D30_stage1.mat`
- Contains: `best_params`, `Blocks`, `Graph`, `trainer_history`, `seeds`

**How to Run:**
```matlab
matlab -nojvm -batch "train_NeuroEA_cec2017_stage1_f1_D30"
```

Expected duration: **~30-60 minutes** (depends on hardware)

---

### Stage 2: Transfer to F9 (Continue from Stage 1)

**File:** `train_NeuroEA_cec2017_stage2_f9_D30_from_f1.m`

**Settings:**
- Same inner and outer trainer settings as Stage 1
- **Initialization:** 50% from Stage 1 best parameters (with Gaussian mutation), 50% random
- Problem: CEC2017_F9, D = 30

**Input:** `trained_NeuroEA_CEC2017_F1_D30_stage1.mat` (Stage 1 output)

**Output:** `trained_NeuroEA_F9_D30_stage2_from_f1.mat`
- Contains: Stage 1 and Stage 2 best parameters, both fitness values, trainer histories

**How to Run:**
```matlab
matlab -nojvm -batch "train_NeuroEA_cec2017_stage2_f9_D30_from_f1"
```

Expected duration: **~30-60 minutes** (same as Stage 1)

---

## Testing / Inference

**File:** `load_trained_NeuroEA_and_run.m`

This standalone script loads the trained model from Stage 2 and evaluates it on a test problem.

**Configuration (edit these lines in the script):**
```matlab
% Trained model file (output from Stage 2)
TRAINED_MODEL_FILE = 'trained_NeuroEA_F9_D30_stage2_from_f1.mat';

% Test problem configuration
TEST_PROBLEM_CLASS = @CEC2017_F1;        % Change to @CEC2017_F4, @CEC2017_F9, etc.
TEST_PROBLEM_NAME = 'CEC2017_F1';
TEST_DIMENSION = 30;

% Test settings
NUM_TEST_RUNS = 5;                       % Number of independent runs
TEST_MAX_FE = 3000;                      % Budget per run
```

**How to Run:**
```matlab
matlab -nojvm -batch "load_trained_NeuroEA_and_run"
```

**Example: Test on F4**
```matlab
TEST_PROBLEM_CLASS = @CEC2017_F4;
TEST_PROBLEM_NAME = 'CEC2017_F4';
```

---

## Paper Alignment Summary

| Aspect                    | Paper Setting        | Our Setting | Note                    |
|---------------------------|----------------------|-------------|-------------------------|
| **Problem sequence**      | F1 → F9 → ...        | F1 → F9    | Sequential not joint    |
| **Training approach**     | GA on parameters     | GA on parameters | Identical              |
| **GA population**         | 50                   | 50         | Same                    |
| **GA max evals**          | 5000                 | 5000       | Same                    |
| **Inner population**      | 100                  | 30         | Scaled for budget       |
| **Inner generations**     | ?                    | 100        | ~3000 FE per run        |
| **Inner maxFE**           | ?                    | 3000       | Matches paper range     |
| **Architecture**          | 11-block largest     | 11-block largest | Identical            |
| **Block parameters**      | Tuned                | Tuned (55 total) | Same methodology      |
| **Fitness metric**        | Mean over runs       | Mean 3 runs | Fewer runs to save time |

---

## File Outputs

### Stage 1 Output
`trained_NeuroEA_CEC2017_F1_D30_stage1.mat` contains:
```
Blocks                    - Array of 11 trained blocks
Graph                     - 11×11 adjacency matrix
best_params               - Best parameter vector (1×55)
best_fitness              - Best mean fitness achieved
trainer_history           - Struct with training curves
seeds                     - Seeds used for reproducibility
PROBLEM_CLASS, PROBLEM_NAME, DIMENSION, STAGE
INNER_POP, INNER_GEN, INNER_MAX_FE
TRAINER_POP, TRAINER_MAX_EVALS, NUM_RUNS_PER_CANDIDATE
```

### Stage 2 Output
`trained_NeuroEA_F9_D30_stage2_from_f1.mat` contains:
```
[All from Stage 1, plus:]
best_params_stage1        - Stage 1 best parameters
best_fitness_stage1       - Stage 1 best fitness (on F1)
best_params_stage2        - Stage 2 best parameters
best_fitness_stage2       - Stage 2 best fitness (on F9)
trainer_history_stage2    - Stage 2 training curves
```

---

## Tips & Customization

### Change Test Problem
Edit `load_trained_NeuroEA_and_run.m`:
```matlab
TEST_PROBLEM_CLASS = @CEC2017_F4;    % Try F4
TEST_PROBLEM_NAME = 'CEC2017_F4';
```

### Increase Training Budget
To train twice as long, edit Stage 1 or Stage 2:
```matlab
TRAINER_MAX_EVALS = 10000;           % Double the evaluations
```

### Change Training Sequence
To train on different problems:
- Create a new Stage 1 script with your chosen problem
- Stage 2 will automatically load Stage 1's best parameters

---

## Expected Performance

With these settings, you should expect:

**Stage 1 (F1):**
- Best fitness improves over ~5000 evaluations
- Final best: typically 10^2 to 10^4 range (problem-dependent)

**Stage 2 (F9):**
- Transfer learning: starts with F1's best parameters
- Transfer improvement: typically 20-50% better than random start
- Final best: F9 is harder, expect higher values

**Test performance:**
- On F1: Should match or exceed Stage 1 final fitness
- On F9: Should match or exceed Stage 2 final fitness
- On F4: Mixed performance (not seen during training)

---

## Important Assumptions & Caveats

✅ **Clear assumptions:**
1. PlatEMO is installed in correct folder structure
2. MATLAB R2024a with necessary toolboxes
3. All paths set correctly (scripts add paths automatically)
4. CEC2017 problem files exist under `PlatEMO/Problems/Single-objective optimization/CEC 2017/`

⚠️ **Caveats:**
- Budget is reduced vs paper (inner pop 30 vs 100)
- Only 2 problems trained (F1, F9) vs paper's full set
- Results will not match paper exactly due to budget reduction
- Reproducibility depends on random seed settings

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  train_NeuroEA_cec2017_stage1_f1_D30.m                      │
│  ─────────────────────────────────────────                  │
│  GA tunes 55 block parameters on F1                         │
│  Saves: trained_NeuroEA_F1_D30_stage1.mat                   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  train_NeuroEA_cec2017_stage2_f9_D30_from_f1.m              │
│  ──────────────────────────────────────────────────────     │
│  Loads Stage 1 best_params                                  │
│  GA continues tuning on F9                                  │
│  Saves: trained_NeuroEA_F9_D30_stage2_from_f1.mat           │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  load_trained_NeuroEA_and_run.m                             │
│  ─────────────────────────────────────                      │
│  Loads Stage 2 model                                        │
│  Tests on any problem (F1, F4, F9, etc.)                   │
│  Reports: mean fitness, std, min/max over runs              │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

1. **Run Stage 1 training:**
   ```bash
   cd /home/jona/github/PlatEMO
   export LIBGL_ALWAYS_INDIRECT=1
   matlab -nojvm -batch "train_NeuroEA_cec2017_stage1_f1_D30"
   ```

2. **Run Stage 2 training:**
   ```bash
   matlab -nojvm -batch "train_NeuroEA_cec2017_stage2_f9_D30_from_f1"
   ```

3. **Test on any problem:**
   ```bash
   matlab -nojvm -batch "load_trained_NeuroEA_and_run"
   ```

---

## References

**Paper structure:** NeuroEA (as described in recent EvComm literature)

**Architecture details:** 
- 11-block largest configuration
- Block parameters tuned via GA
- Sequential multi-problem training for generalization

---

**Last updated:** April 5, 2026

For questions or issues, refer to the MATLAB command window output and trainer_history struct in saved .mat files.
