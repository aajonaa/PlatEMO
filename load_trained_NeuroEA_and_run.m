%% load_trained_NeuroEA_and_run.m (Paper-aligned version)
% Load trained NeuroEA model (from Stage 2) and run on a test problem
%
% This script:
% 1. Loads the trained model from Stage 2 training
% 2. Runs the trained NeuroEA on a specified test problem
% 3. Displays performance statistics
%
% Configuration:
%   - TRAINED_MODEL_FILE: Path to trained model .mat file
%   - TEST_PROBLEM_CLASS: Problem handle (e.g., @CEC2017_F1)
%   - TEST_PROBLEM_NAME: Display name
%   - TEST_DIMENSION: Problem dimension
%   - NUM_TEST_RUNS: How many independent runs to execute
%   - TEST_MAX_FE: Budget per run

clear all; clc;

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
%% CONFIGURATION - EDIT THESE TO CHANGE TEST PROBLEM
%% ========================================================================

% Trained model file (output from Stage 2)
TRAINED_MODEL_FILE = 'trained_NeuroEA_CEC2017_F9_D30_stage2_from_f1.mat';

% Test problem configuration
TEST_PROBLEM_CLASS = @CEC2017_F1;                % Change to @CEC2017_F1, @CEC2017_F4, @CEC2017_F9, etc.
TEST_PROBLEM_NAME = 'CEC2017_F1';               % Display name
TEST_DIMENSION = 30;                            % Dimension

% Test settings
NUM_TEST_RUNS = 5;                              % Number of independent test runs
TEST_MAX_FE = 3000;                             % Budget per run

%% Load trained model
fprintf('\n%s\n', repmat('=', 1, 80));
fprintf('LOAD AND RUN TRAINED NEUROEA MODEL\n');
fprintf('%s\n', repmat('=', 1, 80));

fprintf('\nLoading trained model from: %s\n', TRAINED_MODEL_FILE);

if ~isfile(TRAINED_MODEL_FILE)
    error('Trained model file not found: %s\nPlease run Stage 1 and Stage 2 training first.', TRAINED_MODEL_FILE);
end

model_data = load(TRAINED_MODEL_FILE);

Blocks = model_data.Blocks;
Graph = model_data.Graph;
best_params_stage2 = model_data.best_params_stage2;
best_params_stage1 = model_data.best_params_stage1;
best_fitness_stage1 = model_data.best_fitness_stage1;
best_fitness_stage2 = model_data.best_fitness_stage2;

fprintf('  Stage 1 trained on: F1\n');
fprintf('    Best fitness: %.6e\n', best_fitness_stage1);
fprintf('  Stage 2 trained on: %s\n', model_data.PROBLEM_NAME);
fprintf('    Best fitness: %.6e\n', best_fitness_stage2);

% Assign trained parameters to blocks
fprintf('\nAssigning trained parameters to blocks...\n');
Blocks.ParameterSet(best_params_stage2);

fprintf('\n%s\n', repmat('=', 1, 80));
fprintf('TEST CONFIGURATION\n');
fprintf('%s\n', repmat('=', 1, 80));

fprintf('\nTest problem: %s (D=%d)\n', TEST_PROBLEM_NAME, TEST_DIMENSION);
fprintf('Number of test runs: %d\n', NUM_TEST_RUNS);
fprintf('Max function evaluations per run: %d\n\n', TEST_MAX_FE);

%% ========================================================================
%% RUN ON TEST PROBLEM
%% ========================================================================

fprintf('Running trained NeuroEA on %s...\n\n', TEST_PROBLEM_NAME);

test_best_objs = [];

for run_idx = 1:NUM_TEST_RUNS
    fprintf('  Run %d/%d: ', run_idx, NUM_TEST_RUNS);
    
    % Create problem
    TestProblem = feval(TEST_PROBLEM_CLASS, TEST_DIMENSION);
    TestProblem.maxFE = TEST_MAX_FE;
    
    % Run NeuroEA with trained parameters
    try
        algo = NeuroEA('parameter', {Blocks, Graph});
        algo.Solve(TestProblem);
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
        fprintf('Warning: NeuroEA error: %s\n', err.message);
        best_obj = inf;
    end
    
    test_best_objs = [test_best_objs; best_obj];
    
    fprintf('best=%.6e\n', best_obj);
end

%% ========================================================================
%% RESULTS SUMMARY
%% ========================================================================

mean_test_best = mean(test_best_objs);
std_test_best = std(test_best_objs);
median_test_best = median(test_best_objs);
min_test_best = min(test_best_objs);
max_test_best = max(test_best_objs);

fprintf('\n%s\n', repmat('=', 1, 80));
fprintf('TEST RESULTS ON %s\n', TEST_PROBLEM_NAME);
fprintf('%s\n', repmat('=', 1, 80));

fprintf('\nNumber of runs: %d\n', NUM_TEST_RUNS);
fprintf('Max FE per run: %d\n', TEST_MAX_FE);

fprintf('\nBest objective value statistics:\n');
fprintf('  Mean:   %.6e\n', mean_test_best);
fprintf('  Std:    %.6e\n', std_test_best);
fprintf('  Median: %.6e\n', median_test_best);
fprintf('  Min:    %.6e\n', min_test_best);
fprintf('  Max:    %.6e\n', max_test_best);

fprintf('\nTraining history (for reference):\n');
fprintf('  Stage 1 (on F1):               %.6e\n', best_fitness_stage1);
fprintf('  Stage 2 (on F9, transfer:      %.6e\n', best_fitness_stage2);
fprintf('  Test (on %s, mean):           %.6e\n', TEST_PROBLEM_NAME, mean_test_best);

fprintf('\n%s\n', repmat('=', 1, 80));
fprintf('Test complete.\n');
fprintf('%s\n\n', repmat('=', 1, 80));

