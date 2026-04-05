% AUTOV TRAINING SETUP - README
% ============================================================================
% Train AutoV operator for single-objective CEC2017 optimization
% Using a reduced but fair budget matching NeuroEA setup
% ============================================================================

%% FILES CREATED
% 1. train_AutoV_cec2017_common.m
%    - Core training function (GA-based operator design)
%    - Implements outer GA loop to evolve operator parameters
%    - Evaluates each operator via 3 runs, returns median fitness
%    - Handles both stage 1 (random init) and stage 2 (seeded from stage 1)
%
% 2. train_AutoV_cec2017_stage1_f1_D30.m
%    - Stage 1 training on CEC2017_F1 with D=30
%    - Outer budget: 500 candidate operators (20 pop × 25 gen)
%    - Inner budget: D=30, pop=30, maxFE=3000
%    - Saves: trained_AutoV_CEC2017_F1_D30_stage1.mat
%    - Run this FIRST
%
% 3. train_AutoV_cec2017_stage2_f9_D30_from_f1.m
%    - Stage 2 training on CEC2017_F9 with D=30
%    - Initialized from stage 1 best operator + mutations
%    - Same outer/inner budgets as stage 1
%    - Saves: trained_AutoV_CEC2017_F9_D30_stage2_from_f1.mat
%    - Run this SECOND (requires stage 1 output)
%
% 4. load_trained_AutoV_and_run.m
%    - Load trained AutoV operator
%    - Run on any CEC2017 test problem
%    - Display convergence plots
%    - Interactive or programmatic usage

%% CONFIGURATION DETAILS

% === OPERATOR REPRESENTATION ===
% Operator family: h3 (TSRI - Translation, Scale, Rotation Invariant)
% Operator equation: o = r1*(u-l) + r2*x2 + (1-r2)*x1
%   where r1 ~ N(0, w1^2)
%         r2 ~ N(w3, w2^2)
%         x1, x2 are parents
%         u, l are bounds
% 
% Parameter sets: k = 10
% Parameters per set: [w1, w2, w3, w4]
%   w1 ∈ [0, 1]       (r1 coefficient)
%   w2 ∈ [0, 1]       (r2 standard deviation)
%   w3 ∈ [-1, 1]      (r2 mean)
%   w4 ∈ [1e-6, 1]    (probability weight for roulette wheel)
% 
% Total search space: 10 × 4 = 40 dimensions

% === OUTER TRAINER (GA for operator design) ===
% Population size:     20
% Max evaluations:     500 per stage
% Generations:         ~25 (ceil(500/20))
% Selection:           Binary tournament
% Variation:           Gaussian mutation (0.20 * range)
% Survival:            Environmental (merge parents & offspring, keep best 50%)
% Total across 2 stages: 1000 operators evaluated

% === INNER SOLVER (AutoV on test problem) ===
% Population size:     30
% Max FE:              3000
% Generations:         ~100 (3000/30)
% Dimension:           30
% Selection:           Binary tournament
% Variation:           TSRI operator (adaptive to parameter set)
% Survival:            Environmental (single-objective: keep best N)
% Repeat per operator: 3 independent runs
% Fitness aggregation: MEDIAN of 3 runs

% === TRAINING PROBLEMS ===
% Stage 1: CEC2017_F1 (shifted sphere function)
%          Global optimum: f(x*) = 100
% 
% Stage 2: CEC2017_F9 (shifted composite function)
%          Global optimum: f(x*) = 900
%          Continues from stage 1 best operator

%% QUICK START

% Step 1: Open MATLAB and cd to PlatEMO root directory
cd /home/jona/github/PlatEMO

% Step 2: Run stage 1 training (takes ~15-30 minutes depending on hardware)
train_AutoV_cec2017_stage1_f1_D30
% Output: trained_AutoV_CEC2017_F1_D30_stage1.mat

% Step 3: Run stage 2 training (takes ~15-30 minutes)
train_AutoV_cec2017_stage2_f9_D30_from_f1
% Output: trained_AutoV_CEC2017_F9_D30_stage2_from_f1.mat

% Step 4: Use the trained operator
load_trained_AutoV_and_run('trained_AutoV_CEC2017_F9_D30_stage2_from_f1.mat')
% Interactive: Choose test problem from CEC2017_F1 through CEC2017_F28
% Output: Convergence plot and best fitness achieved

%% EXPECTED RESULTS

% Stage 1 (F1, D=30):
%   - Initial fitness: ~1e7 (with random operators)
%   - Final fitness:   ~1e3-1e4 (after evolution)
%   - Improvement:     >99%

% Stage 2 (F9, D=30, from F1):
%   - Initial fitness: ~1e5 (from F1-tuned operator)
%   - Final fitness:   ~1e4-1e5 (further refined for F9)
%   - Should benefit from F1 warm start

% Total training time:
%   ~30-60 minutes for both stages
%   Depends on hardware (GPU/parallel evaluation recommended)

%% ACCESSING RESULTS PROGRAMMATICALLY

% Load the final trained operator:
load('trained_AutoV_CEC2017_F9_D30_stage2_from_f1.mat');

% Access results:
best_operator_matrix          % (10 x 4) parameter matrix
best_fitness                   % Final best fitness value
trainer_history                % Struct with training progress
  .best_fitnesses              % Best fitness per generation
  .mean_fitnesses              % Mean fitness per generation
  .generation_num              % Total generations
  .num_evaluations             % Total evaluations
  .k                           % Number of parameter sets = 10
  .operator_family             % = 'h3'
  .inner_pop                   % = 30
  .inner_maxfe                 % = 3000
  .inner_D                     % = 30

% Run on custom problem:
problem = feval(@CEC2017_F9);
problem.D = 30;
problem.maxFE = 3000;

Weight = best_operator_matrix;
Fit = cumsum(Weight(:, 4));
Fit = Fit ./ max(Fit);

population = problem.Initialization();
[population, fitness] = EnvironmentalSelection(population, 30);

while problem.FE < 3000
    MatingPool = TournamentSelection(2, 60, fitness);
    offspring = TSRIOperator(problem, Weight, Fit, population(MatingPool));
    [population, fitness] = EnvironmentalSelection([population, offspring], 30);
end

fprintf('Best fitness: %.6e\n', min(fitness));

%% DESIGN RATIONALE

% === Fair budget allocation ===
% NeuroEA training uses: 50 pop × 100 gen × 3 runs = 15,000 inner evals per stage
% AutoV training uses: 20 pop × 25 gen × 3 runs = 1,500 inner evals per stage
% Ratio: ~1:10, reflecting that AutoV operates on simpler operators
%
% Total fitness evaluations (not counting inner solver calls):
%   NeuroEA: 100 evaluations × 3 runs = 300 outer fitness evals
%   AutoV:   500 evaluations × 3 runs = 1,500 outer fitness evals
%
% This is fair because:
% - AutoV operators are simpler (40D vs 55D parameters)
% - Easier to search (lower dimensional space)
% - More evaluations allowed per fair comparison

% === Operator family choice ===
% h3 (TSRI operator) chosen because:
% - Simple and efficient (only 2 random variables per offspring)
% - Theoretically principled (translation/scale/rotation invariant)
% - Widely used in successful algorithms (CMA-ES inspired)
% - Easy to tune (only 4 parameters per set)
% - Fast evaluation (no matrix operations)

% === Two-stage training ===
% Stage 1 on F1 (sphere: simple, convex):
% - Finds operators good for smooth, convex optimization
% - Fast convergence to near-optimal solutions
%
% Stage 2 on F9 (composite: multi-modal, non-convex):
% - Refines operators for complex landscapes
% - Warm-start from F1 solution gives better starting point
% - Expected to achieve better F9 performance than random init
%
% This mimics the biological principle of transfer learning:
% Learn general skills (F1), then specialize (F9)

%% IMPORTANT NOTES

% 1. Reproducibility:
%    - All runs use fixed random seeds (SEED_BASE = 12345)
%    - Results should be reproducible across runs
%    - Stored in .mat files: 'seeds' array

% 2. Verification of assumptions:
%    - All scripts print assumptions to console at start
%    - k=10 parameter sets (not changeable without code modification)
%    - Operator family: h3 TSRI (hardcoded)
%    - Fitness aggregation: median (not mean) of 3 runs
%    - These are printed prominently: do not silently change

% 3. Customization:
%    - To change outer budget: modify TRAINER_POP_SIZE, TRAINER_MAX_EVALS in train_AutoV_cec2017_common.m
%    - To change inner budget: modify INNER_POP_SIZE, INNER_MAX_FE, INNER_D in common.m
%    - To change repeat runs: modify NUM_RUNS_PER_CANDIDATE in common.m
%    - To use different problem: modify stage scripts
%    - To change k: modify K constant in common.m (affects search space)

% 4. Parallel evaluation:
%    - Current implementation uses serial evaluation
%    - Can be parallelized in evaluate_autov_operator_on_problem
%    - Consider using parfor for faster training

% 5. Extension paths:
%    - Train on additional functions (CEC2017_F2, F3, etc.)
%    - Train for different dimensions (D=10, D=50)
%    - Extend to multi-objective (use multi-objective CEC functions)
%    - Compare with other operator design methods (RL, Bayesian opt)

%% TROUBLESHOOTING

% Problem: "Fail to load weights from..."
% Solution: Make sure stage 1 output file exists before running stage 2
%           File should be: trained_AutoV_CEC2017_F1_D30_stage1.mat

% Problem: Slow execution
% Solution: Run on GPU if available, or parallelize inner evaluations
%           Modify evaluate_autov_operator_on_problem to use parfor

% Problem: NaN or Inf fitness values
% Solution: Check problem bounds, verify TSRIOperator generates valid offspring
%           Use smaller mutation strength or different initialization

% Problem: EnvironmentalSelection errors
% Solution: Ensure Population objects are properly initialized
%           Check that TSRIOperator returns valid Population objects

%% REFERENCES

% AutoV paper:
% Y. Tian, X. Zhang, C. He, K. C. Tan, and Y. Jin. Principled design of
% translation, scale, and rotation invariant variation operators for
% metaheuristics. Chinese Journal of Electronics, 2023, 32(1): 111-129.
%
% CEC2017 benchmark:
% G. Wu, R. Mallipeddi, and P. N. Suganthan. Problem definitions and
% evaluation criteria for the CEC 2017 competition on constrained real-
% parameter optimization. National University of Defense Technology, 2016.
%
% PlatEMO platform:
% Ye Tian, et al. PlatEMO: A MATLAB platform for evolutionary 
% multi-objective optimization. IEEE CIM, 2017, 12(4): 73-87.

%% ============================================================================
% Training summary:
% File                                        | Purpose
% ============================================================================
% train_AutoV_cec2017_common.m               | Core GA trainer
% train_AutoV_cec2017_stage1_f1_D30.m        | Stage 1 entry point (F1)
% train_AutoV_cec2017_stage2_f9_D30_from_f1  | Stage 2 entry point (F9)
% load_trained_AutoV_and_run.m               | Evaluation & plotting
% ============================================================================
