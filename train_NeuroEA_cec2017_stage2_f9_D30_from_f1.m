%% train_NeuroEA_cec2017_stage2_f9_D30_from_f1.m
% Stage 2: Continue training NeuroEA on CEC2017 F9 (D=30)
%
% This script loads the best parameters from Stage 1 (trained on F1),
% then continues GA training on the F9 problem.
%
% Input: trained_NeuroEA_F1_D30_stage1.mat (from Stage 1)
% Output: trained_NeuroEA_F9_D30_stage2_from_f1.mat
%
% This implements the paper's sequential training approach:
%   Stage 1: Train on F1 -> save best_params
%   Stage 2: Load best_params, continue GA training on F9

clear; clc;

%% Disable figure visualization to avoid opening too many windows
set(0, 'DefaultFigureVisible', 'off');
close all;

%% Add PlatEMO to path
current_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(current_dir, 'PlatEMO'));
addpath(fullfile(current_dir, 'PlatEMO', 'Algorithms'));
addpath(fullfile(current_dir, 'PlatEMO', 'Algorithms', 'NeuroEA'));
addpath(fullfile(current_dir, 'PlatEMO', 'Algorithms', 'Utility functions'));
addpath(fullfile(current_dir, 'PlatEMO', 'Metrics'));
addpath(fullfile(current_dir, 'PlatEMO', 'Problems'));
addpath(fullfile(current_dir, 'PlatEMO', 'Problems', 'Single-objective optimization'));
addpath(fullfile(current_dir, 'PlatEMO', 'Problems', 'Single-objective optimization', 'CEC 2017'));
addpath(fullfile(current_dir, 'PlatEMO', 'GUI'));

%% ========================================================================
%% STAGE 2 CONFIGURATION
%% ========================================================================

PROBLEM_CLASS = @CEC2017_F9;
PROBLEM_NAME = 'CEC2017_F9';
DIMENSION = 30;
STAGE = 2;

% Inner NeuroEA (solver on each problem)
INNER_POP = 30;
INNER_GEN = 100;
INNER_MAX_FE = INNER_POP * INNER_GEN;  % = 3000

% Outer GA trainer (parameter tuner)
TRAINER_POP = 50;
TRAINER_MAX_EVALS = 5000;
NUM_RUNS_PER_CANDIDATE = 3;  % Mean fitness over 3 runs
SEED_BASE = 67890;            % Different seed base for Stage 2

% Input and output files
INPUT_FILE = 'trained_NeuroEA_CEC2017_F1_D30_stage1.mat';
OUTPUT_FILE = sprintf('trained_NeuroEA_%s_D%d_stage%d_from_f1.mat', PROBLEM_NAME, DIMENSION, STAGE);

fprintf('\n%s\n', repmat('=', 1, 80));
fprintf('NEUROEA STAGE 2: SEQUENTIAL TRANSFER TRAINING ON %s (D=%d)\n', PROBLEM_NAME, DIMENSION);
fprintf('%s\n', repmat('=', 1, 80));
fprintf('\nInput (Stage 1 best model):  %s\n', INPUT_FILE);
fprintf('Output (Stage 2 trained):    %s\n', OUTPUT_FILE);
fprintf('\nPAPER-ALIGNED SETTINGS:\n');
fprintf('  Inner NeuroEA:   pop=%d, generations=%d, maxFE=%d\n', ...
    INNER_POP, INNER_GEN, INNER_MAX_FE);
fprintf('  Outer GA trainer: pop=%d, max_evals=%d\n', TRAINER_POP, TRAINER_MAX_EVALS);
fprintf('  Fitness metric:  mean best objective over %d independent runs\n', NUM_RUNS_PER_CANDIDATE);
fprintf('  Initial population: from Stage 1 best parameters (mutation-based variation)\n');
fprintf('%s\n\n', repmat('=', 1, 80));

%% ========================================================================
%% LOAD STAGE 1 RESULTS
%% ========================================================================

fprintf('Loading Stage 1 trained model from: %s\n', INPUT_FILE);

if ~isfile(INPUT_FILE)
    error('Stage 1 output file not found: %s\nPlease run train_NeuroEA_cec2017_stage1_f1_D30.m first.', INPUT_FILE);
end

stage1_data = load(INPUT_FILE);

Blocks = stage1_data.Blocks;
Graph = stage1_data.Graph;
best_params_stage1 = stage1_data.best_params;
best_fitness_stage1 = stage1_data.best_fitness;

fprintf('  Stage 1 best fitness (on F1):  %.6e\n', best_fitness_stage1);
fprintf('  Stage 1 best params (first 5): [%.4f, %.4f, %.4f, %.4f, %.4f, ...]\n', ...
    best_params_stage1(1), best_params_stage1(2), best_params_stage1(3), ...
    best_params_stage1(4), best_params_stage1(5));

num_params = length(best_params_stage1);
param_lowers = Blocks.lowers();
param_uppers = Blocks.uppers();

fprintf('\n');

%% ========================================================================
%% INITIALIZE POPULATION FOR STAGE 2
%% ========================================================================

fprintf('Initializing GA population for Stage 2...\n');
fprintf('  50%% from Stage 1 best + Gaussian mutation\n');
fprintf('  50%% from random initialization\n\n');

rng_stream = RandStream('mt19937ar', 'Seed', SEED_BASE);
seed_count = stage1_data.seed_count;  % Continue from Stage 1 seed count
seeds = stage1_data.seeds;
seeds(seed_count+1:seed_count+TRAINER_MAX_EVALS*NUM_RUNS_PER_CANDIDATE) = 0;  % Allocate space

% Initialize population
population = [];

% 50% from Stage 1 best + Gaussian mutation
num_mutated = ceil(TRAINER_POP / 2);
for i = 1:num_mutated
    % Mutate Stage 1 best by adding small Gaussian noise
    mutation_std = 0.1 * (param_uppers - param_lowers);
    mutated_params = best_params_stage1 + mutation_std .* randn(rng_stream, 1, num_params);
    % Clip to bounds
    mutated_params = max(param_lowers, min(param_uppers, mutated_params));
    population = [population; mutated_params];
end

% 50% from random initialization
num_random = TRAINER_POP - num_mutated;
random_pop = repmat(param_lowers, num_random, 1) + ...
    rand(rng_stream, num_random, num_params) .* ...
    repmat((param_uppers - param_lowers), num_random, 1);
population = [population; random_pop];

fprintf('Population initialized with %d candidates.\n\n', TRAINER_POP);

%% ========================================================================
%% OUTER GA TRAINER LOOP (Stage 2)
%% ========================================================================

fprintf('Starting outer GA trainer loop on %s...\n', PROBLEM_NAME);
fprintf('  Population size: %d\n', TRAINER_POP);
fprintf('  Max candidate evaluations: %d\n', TRAINER_MAX_EVALS);
fprintf('  Fitness evaluation: mean of %d runs on %s\n\n', ...
    NUM_RUNS_PER_CANDIDATE, PROBLEM_NAME);

% Storage for history
best_fitnesses_per_eval = [];
mean_fitnesses_per_eval = [];
num_evaluations = 0;

% Evaluate initial population
fitness_values = inf(TRAINER_POP, 1);

fprintf('Evaluating initial population (candidate 1-%d on %s)...\n', TRAINER_POP, PROBLEM_NAME);

for candidate_idx = 1:TRAINER_POP
    % Evaluate this candidate over NUM_RUNS_PER_CANDIDATE independent runs
    run_fitnesses = [];
    
    for run_idx = 1:NUM_RUNS_PER_CANDIDATE
        % Get a seed for this run
        seed_count = seed_count + 1;
        current_seed = SEED_BASE + seed_count;
        seeds(seed_count) = current_seed;
        
        % Create blocks with candidate parameters
        candidate_blocks = [
            Block_Population()
            Block_Tournament(60, 10)
            Block_Tournament(60, 10)
            Block_Tournament(60, 10)
            Block_Exchange(3)
            Block_Exchange(3)
            Block_Exchange(3)
            Block_Exchange(3)
            Block_Crossover(2, 5)
            Block_Mutation(5)
            Block_Selection(30)
        ];
        
        % Assign candidate parameters to blocks using ParameterSet
        candidate_blocks.ParameterSet(population(candidate_idx, :));
        
        % Run NeuroEA on the F9 problem
        Problem = feval(PROBLEM_CLASS, DIMENSION);
        Problem.maxFE = INNER_MAX_FE;
        
        % Create NeuroEA with configured blocks and Graph
        algo = NeuroEA('parameter', {candidate_blocks, Graph});
        
        % Run via PlatEMO interface
        try
            algo.Solve(Problem);
            close all;  % Close any figures created during solve
            
            % Extract best fitness from result
            if ~isempty(algo.result) && size(algo.result, 2) >= 2
                final_pop = algo.result{end, 2};
                if ~isempty(final_pop)
                    best_obj = min(final_pop.objs);
                else
                    best_obj = inf;
                end
            else
                best_obj = inf;
            end
        catch err
            close all;  % Close any figures created during error
            fprintf('      Warning: NeuroEA error: %s\n', err.message);
            best_obj = inf;
        end
        
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
[best_fitness_stage2, best_idx] = min(fitness_values(1:min(TRAINER_POP, num_evaluations)));
best_params_stage2 = population(best_idx, :);

fprintf('\n%s\n', repmat('=', 1, 80));
fprintf('STAGE 2 TRAINING COMPLETE\n');
fprintf('%s\n', repmat('=', 1, 80));
fprintf('\nStage 2 best fitness on %s: %.6e\n', PROBLEM_NAME, best_fitness_stage2);
fprintf('Stage 1 best fitness on F1:    %.6e\n', best_fitness_stage1);
fprintf('Transfer improvement: %.2f%%\n', ...
    (1 - best_fitness_stage2/best_fitness_stage1) * 100);

fprintf('\nBest parameters found (first 5): [%.4f, %.4f, %.4f, %.4f, %.4f, ...]\n', ...
    best_params_stage2(1), best_params_stage2(2), best_params_stage2(3), ...
    best_params_stage2(4), best_params_stage2(5));

% Assign best parameters to Blocks before saving
fprintf('\nAssigning best parameters to Blocks...\n');
Blocks.ParameterSet(best_params_stage2);

% Prepare history struct
trainer_history_stage2 = struct();
trainer_history_stage2.best_fitnesses = best_fitnesses_per_eval;
trainer_history_stage2.mean_fitnesses = mean_fitnesses_per_eval;
trainer_history_stage2.num_evaluations = num_evaluations;
trainer_history_stage2.num_candidates = min(TRAINER_POP, num_evaluations);

% Save trained model
fprintf('\nSaving Stage 2 trained model to: %s\n', OUTPUT_FILE);

save(OUTPUT_FILE, ...
    'Blocks', 'Graph', ...
    'best_params_stage1', 'best_fitness_stage1', ...
    'best_params_stage2', 'best_fitness_stage2', ...
    'trainer_history_stage2', 'seeds', 'seed_count', ...
    'PROBLEM_CLASS', 'PROBLEM_NAME', 'DIMENSION', 'STAGE', ...
    'INNER_POP', 'INNER_GEN', 'INNER_MAX_FE', ...
    'TRAINER_POP', 'TRAINER_MAX_EVALS', 'NUM_RUNS_PER_CANDIDATE');

fprintf('Save complete.\n\n');

fprintf('STAGE 2 SUMMARY:\n');
fprintf('  Initial problem (Stage 1): F1 (D=%d)\n', DIMENSION);
fprintf('  Current problem (Stage 2):  %s (D=%d)\n', PROBLEM_NAME, DIMENSION);
fprintf('  Best fitness Stage 1: %.6e\n', best_fitness_stage1);
fprintf('  Best fitness Stage 2: %.6e\n', best_fitness_stage2);
fprintf('  Output file: %s\n', OUTPUT_FILE);
fprintf('\nTRAINED MODEL IS NOW READY FOR USE\n');
fprintf('Next: Use load_trained_NeuroEA_and_run.m to test on any problem\n');
fprintf('%s\n\n', repmat('=', 1, 80));
