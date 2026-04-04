%% train_NeuroEA_cec2017_f9_D30.m
% Train NeuroEA on CEC2017 F9 (shifted rastrigin function) with D=30
%
% This script:
% 1. Trains a single NeuroEA model by tuning 55 block parameters via outer GA
% 2. Saves the trained model, blocks, graph, and history to a .mat file
% 3. Results can be loaded later with load_trained_NeuroEA_and_run.m

clear; clc;

% Add PlatEMO to path
addpath(genpath('/home/jona/github/PlatEMO/PlatEMO'));

%% Configuration
PROBLEM_CLASS = @CEC2017_F9;
PROBLEM_NAME = 'CEC2017_F9';
DIMENSION = 30;
OUTPUT_FILE = sprintf('trained_NeuroEA_%s_D%d.mat', PROBLEM_NAME, DIMENSION);

fprintf('\n%s\n', repmat('#',1,70));
fprintf('# Training NeuroEA on %s (D=%d)\n', PROBLEM_NAME, DIMENSION);
fprintf('# Output will be saved to: %s\n', OUTPUT_FILE);
fprintf('%s\n\n', repmat('#',1,70));

%% Run training
[best_params, trainer_history, seeds, best_fitness] = ...
    train_NeuroEA_cec2017_common(PROBLEM_CLASS, PROBLEM_NAME, DIMENSION);

%% Recreate blocks and graph for saving
[Blocks, Graph] = train_setup_utils('create_blocks_graph');

%% Save results
fprintf('\nSaving trained model to: %s\n', OUTPUT_FILE);

save(OUTPUT_FILE, ...
    'Blocks', 'Graph', 'best_params', 'trainer_history', 'seeds', ...
    'best_fitness', 'PROBLEM_NAME', 'PROBLEM_CLASS', 'DIMENSION');

fprintf('Save complete.\n');
fprintf('\nTrained model details:\n');
fprintf('  Problem: %s\n', PROBLEM_NAME);
fprintf('  Dimension: %d\n', DIMENSION);
fprintf('  Best fitness achieved: %.6e\n', best_fitness);
fprintf('  Best parameters (55 values): [%.4f ... %.4f]\n', best_params(1), best_params(end));
fprintf('  Training history available in trainer_history struct\n');
fprintf('\nTo use this model, call:\n');
fprintf('  load_trained_NeuroEA_and_run(''%s'', @YOUR_PROBLEM)\n\n', OUTPUT_FILE);
