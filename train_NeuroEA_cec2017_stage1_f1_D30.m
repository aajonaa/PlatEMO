%% train_NeuroEA_cec2017_stage1_f1_D30.m
% Stage 1: Train NeuroEA on CEC2017 F1 (D=30)
%
% This script trains the paper-style 11-block NeuroEA on single-objective
% optimization task F1. The best parameters are saved for Stage 2.
%
% Paper-aligned settings (with reduced budget):
%   - Inner NeuroEA:  pop=30, generations=100, maxFE=3000
%   - Outer trainer:  GA, pop=50, max_evals=5000
%   - Fitness:        mean of 3 independent runs
%   - Architecture:   largest 11-block NeuroEA (paper-faithful)
%
% Output: trained_NeuroEA_F1_D30_stage1.mat
%   Contains: best_params, Blocks, Graph, trainer_history, DIMENSION, etc.

clear; clc;

%% Add PlatEMO to path
current_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(current_dir, 'PlatEMO'));
addpath(fullfile(current_dir, 'PlatEMO', 'Algorithms'));
addpath(fullfile(current_dir, 'PlatEMO', 'Algorithms', 'NeuroEA'));
addpath(fullfile(current_dir, 'PlatEMO', 'Metrics'));
addpath(fullfile(current_dir, 'PlatEMO', 'Problems'));
addpath(fullfile(current_dir, 'PlatEMO', 'Problems', 'Single-objective optimization'));
addpath(fullfile(current_dir, 'PlatEMO', 'Problems', 'Single-objective optimization', 'CEC 2017'));

%% ========================================================================
%% STAGE 1 CONFIGURATION
%% ========================================================================

PROBLEM_CLASS = @CEC2017_F1;
PROBLEM_NAME = 'CEC2017_F1';
DIMENSION = 30;
STAGE = 1;

% Inner NeuroEA (solver on each problem)
INNER_POP = 30;
INNER_GEN = 100;
INNER_MAX_FE = INNER_POP * INNER_GEN;  % = 3000

% Outer GA trainer (parameter tuner)
TRAINER_POP = 50;
TRAINER_MAX_EVALS = 5000;
NUM_RUNS_PER_CANDIDATE = 3;  % Mean fitness over 3 runs

% Output file
OUTPUT_FILE = sprintf('trained_NeuroEA_%s_D%d_stage%d.mat', PROBLEM_NAME, DIMENSION, STAGE);

fprintf('\n%s\n', repmat('=', 1, 80));
fprintf('NEUROEA STAGE 1: SINGLE-OBJECTIVE TRAINING ON %s (D=%d)\n', PROBLEM_NAME, DIMENSION);
fprintf('%s\n', repmat('=', 1, 80));
fprintf('\nPAPER-ALIGNED SETTINGS (reduced budget):\n');
fprintf('  Inner NeuroEA:   pop=%d, generations=%d, maxFE=%d\n', ...
    INNER_POP, INNER_GEN, INNER_MAX_FE);
fprintf('  Outer GA trainer: pop=%d, max_evals=%d\n', TRAINER_POP, TRAINER_MAX_EVALS);
fprintf('  Fitness metric:  mean best objective over %d independent runs\n', NUM_RUNS_PER_CANDIDATE);
fprintf('  Architecture:    11-block NeuroEA (paper-faithful)\n');
fprintf('  Node order:      [P, T1, T2, T3, E1, E2, E3, E4, C, M, S]\n');
fprintf('\nOutput will be saved to: %s\n', OUTPUT_FILE);
fprintf('%s\n\n', repmat('=', 1, 80));

%% ========================================================================
%% CREATE BLOCKS AND GRAPH (11-block NeuroEA, paper-aligned)
%% ========================================================================

fprintf('Creating 11-block NeuroEA architecture...\n\n');

% Create blocks in order [P, T1, T2, T3, E1, E2, E3, E4, C, M, S]
Blocks = [
    Block_Population()              % P
    Block_Tournament(60, 10)        % T1
    Block_Tournament(60, 10)        % T2
    Block_Tournament(60, 10)        % T3
    Block_Exchange(3)               % E1
    Block_Exchange(3)               % E2
    Block_Exchange(3)               % E3
    Block_Exchange(3)               % E4
    Block_Crossover(2, 5)           % C
    Block_Mutation(5)               % M
    Block_Selection(30)             % S
];

fprintf('Blocks created:\n');
fprintf('  P  = Block_Population()\n');
fprintf('  T1 = Block_Tournament(60, 10)  [1 param]\n');
fprintf('  T2 = Block_Tournament(60, 10)  [1 param]\n');
fprintf('  T3 = Block_Tournament(60, 10)  [1 param]\n');
fprintf('  E1 = Block_Exchange(3)         [3 params]\n');
fprintf('  E2 = Block_Exchange(3)         [3 params]\n');
fprintf('  E3 = Block_Exchange(3)         [3 params]\n');
fprintf('  E4 = Block_Exchange(3)         [3 params]\n');
fprintf('  C  = Block_Crossover(2, 5)     [30 params]\n');
fprintf('  M  = Block_Mutation(5)         [10 params]\n');
fprintf('  S  = Block_Selection(30)       [0 params]\n');

% Get total number of parameters
param_lowers_cell = {Blocks.lower};
param_uppers_cell = {Blocks.upper};
param_lowers = [];
param_uppers = [];
for i = 1:length(Blocks)
    param_lowers = [param_lowers; Blocks(i).lower];
    param_uppers = [param_uppers; Blocks(i).upper];
end
num_params = length(param_lowers);

fprintf('\nTotal tunable parameters: %d\n', num_params);
fprintf('Parameter bounds: [%.4f, %.4f]\n\n', min(param_lowers), max(param_uppers));

% Create adjacency graph (11 x 11)
% Rows/cols: [P, T1, T2, T3, E1, E2, E3, E4, C, M, S]
Graph = [
    0    1    1    1    0    0    0    0    0    0    1  ;  % P -> T1, T2, T3, S
    0    0    0    0    0.25 0.25 0.25 0.25 0    0    0  ;  % T1 -> E1, E2, E3, E4
    0    0    0    0    0.25 0.25 0.25 0.25 0    0    0  ;  % T2 -> E1, E2, E3, E4
    0    0    0    0    0.25 0.25 0.25 0.25 0    0    0  ;  % T3 -> E1, E2, E3, E4
    0    0    0    0    0    0    0    0    1    0    0  ;  % E1 -> C
    0    0    0    0    0    0    0    0    1    0    0  ;  % E2 -> C
    0    0    0    0    0    0    0    0    1    0    0  ;  % E3 -> C
    0    0    0    0    0    0    0    0    1    0    0  ;  % E4 -> C
    0    0    0    0    0    0    0    0    0    1    0  ;  % C -> M
    0    0    0    0    0    0    0    0    0    0    1  ;  % M -> S
    1    0    0    0    0    0    0    0    0    0    0  ;  % S -> P
];

fprintf('Connectivity graph created (11 x 11 adjacency matrix).\n\n');

%% ========================================================================
%% OUTER GA TRAINER LOOP
%% ========================================================================

fprintf('Starting outer GA trainer loop...\n');
fprintf('  Population size: %d\n', TRAINER_POP);
fprintf('  Max candidate evaluations: %d\n', TRAINER_MAX_EVALS);
fprintf('  Fitness evaluation: mean of %d runs on %s\n\n', ...
    NUM_RUNS_PER_CANDIDATE, PROBLEM_NAME);

% Initialize
SEED_BASE = 12345;
rng_stream = RandStream('mt19937ar', 'Seed', SEED_BASE);
seed_count = 0;
seeds = zeros(TRAINER_MAX_EVALS * NUM_RUNS_PER_CANDIDATE, 1);

% Storage for history
best_fitnesses_per_eval = [];
mean_fitnesses_per_eval = [];
generation_num = 0;
num_evaluations = 0;

% Initial population (random parameters within bounds)
population = repmat(param_lowers, TRAINER_POP, 1) + ...
    rand(rng_stream, TRAINER_POP, num_params) .* ...
    repmat((param_uppers - param_lowers), TRAINER_POP, 1);

% Evaluate initial population
fitness_values = inf(TRAINER_POP, 1);

fprintf('Evaluating initial population (candidate 1-%d)...\n', TRAINER_POP);

for candidate_idx = 1:TRAINER_POP
    % Evaluate this candidate over NUM_RUNS_PER_CANDIDATE independent runs
    run_fitnesses = [];
    
    for run_idx = 1:NUM_RUNS_PER_CANDIDATE
        % Get a seed for this run
        seed_count = seed_count + 1;
        current_seed = SEED_BASE + seed_count;
        seeds(seed_count) = current_seed;
        
        % Assign candidate parameters to blocks
        param_idx = 1;
        for block_idx = 1:length(Blocks)
            block_num_params = length(Blocks(block_idx).lower);
            if block_num_params > 0
                Blocks(block_idx).parameter = population(candidate_idx, param_idx:param_idx+block_num_params-1);
                Blocks(block_idx).ParameterAssign();
                param_idx = param_idx + block_num_params;
            end
        end
        
        % Run NeuroEA on the problem
        Problem = feval(PROBLEM_CLASS, DIMENSION);
        [best_obj, ~, ~, ~] = NeuroEA(Problem, Blocks, Graph, INNER_MAX_FE, 1);
        
        run_fitnesses = [run_fitnesses; best_obj];
    end
    
    % Candidate fitness = mean of run fitnesses
    fitness_values(candidate_idx) = mean(run_fitnesses);
    
    num_evaluations = num_evaluations + 1;
    [best_fit, best_idx] = min(fitness_values(1:candidate_idx));
    mean_fit = mean(fitness_values(1:candidate_idx));
    
    fprintf('  Candidate %3d: fitness=%.6e, best so far=%.6e (mean=%.6e)\n', ...
        candidate_idx, fitness_values(candidate_idx), best_fit, mean_fit);
    
    best_fitnesses_per_eval = [best_fitnesses_per_eval; best_fit];
    mean_fitnesses_per_eval = [mean_fitnesses_per_eval; mean_fit];
    
    % Stop if we reach max evaluations
    if num_evaluations >= TRAINER_MAX_EVALS
        fprintf('\nReached max evaluations (%d). Stopping trainer.\n', TRAINER_MAX_EVALS);
        break;
    end
end

%% ========================================================================
%% FINALIZE AND SAVE
%% ========================================================================

% Get best solution
[best_fitness, best_idx] = min(fitness_values(1:min(TRAINER_POP, num_evaluations)));
best_params = population(best_idx, :);

fprintf('\n%s\n', repmat('=', 1, 80));
fprintf('STAGE 1 TRAINING COMPLETE\n');
fprintf('%s\n', repmat('=', 1, 80));
fprintf('\nBest fitness found: %.6e\n', best_fitness);
fprintf('Best parameters (first 5): [%.4f, %.4f, %.4f, %.4f, %.4f, ...]\n', ...
    best_params(1), best_params(2), best_params(3), best_params(4), best_params(5));

% Assign best parameters to Blocks before saving
fprintf('\nAssigning best parameters to Blocks...\n');
param_idx = 1;
for block_idx = 1:length(Blocks)
    num_block_params = length(Blocks(block_idx).lower);
    if num_block_params > 0
        Blocks(block_idx).parameter = best_params(param_idx:param_idx+num_block_params-1);
        Blocks(block_idx).ParameterAssign();
    end
    param_idx = param_idx + num_block_params;
end

% Prepare history struct
trainer_history = struct();
trainer_history.best_fitnesses = best_fitnesses_per_eval;
trainer_history.mean_fitnesses = mean_fitnesses_per_eval;
trainer_history.num_evaluations = num_evaluations;
trainer_history.num_candidates = min(TRAINER_POP, num_evaluations);

% Save trained model
fprintf('\nSaving trained model to: %s\n', OUTPUT_FILE);

save(OUTPUT_FILE, ...
    'Blocks', 'Graph', 'best_params', 'best_fitness', ...
    'trainer_history', 'seeds', 'seed_count', ...
    'PROBLEM_CLASS', 'PROBLEM_NAME', 'DIMENSION', 'STAGE', ...
    'INNER_POP', 'INNER_GEN', 'INNER_MAX_FE', ...
    'TRAINER_POP', 'TRAINER_MAX_EVALS', 'NUM_RUNS_PER_CANDIDATE');

fprintf('Save complete.\n\n');

fprintf('STAGE 1 SUMMARY:\n');
fprintf('  Problem: %s (D=%d)\n', PROBLEM_NAME, DIMENSION);
fprintf('  Best fitness: %.6e\n', best_fitness);
fprintf('  Evaluations completed: %d / %d\n', num_evaluations, TRAINER_MAX_EVALS);
fprintf('  Output file: %s\n', OUTPUT_FILE);
fprintf('\nNext: Run train_NeuroEA_cec2017_stage2_f9_D30_from_f1.m\n');
fprintf('%s\n\n', repmat('=', 1, 80));
