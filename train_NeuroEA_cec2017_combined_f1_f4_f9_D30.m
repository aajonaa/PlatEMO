%% train_NeuroEA_cec2017_combined_f1_f4_f9_D30.m
% Train NeuroEA on CEC2017 F1, F4, F9 simultaneously (combined)
%
% This script trains a SINGLE NeuroEA model that generalizes across
% three different problem types, following the paper's approach of
% multi-problem training.
%
% This script:
% 1. Trains a single NeuroEA model by tuning 55 block parameters via outer GA
% 2. Fitness is evaluated on all three problems: F1, F4, F9
% 3. Mean fitness across all three problems is used as the objective
% 4. Saves the trained model to a .mat file for later reuse

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

%% Configuration
PROBLEM_CLASSES = {@CEC2017_F1, @CEC2017_F4, @CEC2017_F9};
PROBLEM_NAMES = {'CEC2017_F1', 'CEC2017_F4', 'CEC2017_F9'};
DIMENSION = 30;
OUTPUT_FILE = sprintf('trained_NeuroEA_combined_F1_F4_F9_D%d.mat', DIMENSION);

fprintf('\n%s\n', repmat('#',1,70));
fprintf('# Training NeuroEA on COMBINED PROBLEMS: F1, F4, F9 (D=%d)\n', DIMENSION);
fprintf('# This single model will generalize across different problem types\n');
fprintf('# Output will be saved to: %s\n', OUTPUT_FILE);
fprintf('%s\n\n', repmat('#',1,70));

%% Run training
[best_params, trainer_history, seeds, best_fitness] = ...
    train_NeuroEA_cec2017_combined(PROBLEM_CLASSES, PROBLEM_NAMES, DIMENSION);

%% Recreate blocks and graph for saving
[Blocks, Graph] = train_setup_utils('create_blocks_graph');

%% Save results
fprintf('\nSaving trained model to: %s\n', OUTPUT_FILE);

save(OUTPUT_FILE, ...
    'Blocks', 'Graph', 'best_params', 'trainer_history', 'seeds', ...
    'best_fitness', 'PROBLEM_NAMES', 'PROBLEM_CLASSES', 'DIMENSION');

fprintf('Save complete.\n');
fprintf('\nTrained model details:\n');
fprintf('  Problems: %s, %s, %s\n', PROBLEM_NAMES{1}, PROBLEM_NAMES{2}, PROBLEM_NAMES{3});
fprintf('  Dimension: %d\n', DIMENSION);
fprintf('  Best mean fitness (across all 3 problems): %.6e\n', best_fitness);
fprintf('  Best parameters (55 values): [%.4f ... %.4f]\n', best_params(1), best_params(end));
fprintf('  Training history available in trainer_history struct\n');
fprintf('\nTo use this generalized model, call:\n');
fprintf('  [fitness, pop] = load_trained_NeuroEA_and_run(...\n');
fprintf('    ''%s'', @CEC2017_F1);  %% Test on any problem\n\n', OUTPUT_FILE);
