%% train_AutoV_cec2017_stage2_f9_D30_from_f1.m
% Train AutoV on CEC2017 F9 (shifted composite function) with D=30
% Stage 2 of 2-stage training, initialized from Stage 1 (F1) results
%
% This script:
% 1. Loads the best AutoV operator from Stage 1 (F1 training)
% 2. Initializes the search population with Stage 1 result + mutations
% 3. Continues training on F9 with the same outer/inner budgets
% 4. Evaluates each operator by running inner solver 3 times, taking median
% 5. Uses outer budget: 500 candidate operators (20 pop × ~25 gen)
% 6. Uses inner budget: D=30, pop=30, maxFE=3000 per evaluation
% 7. Saves the trained operator matrix and full training history
%
% This 2-stage approach allows the operator to specialize first on F1,
% then refine for F9 starting from a good F1-tuned baseline.

clear; clc;

% Add PlatEMO to path (including AutoV helpers)
addpath(genpath('/home/jona/github/PlatEMO/PlatEMO'));

%% Configuration
PROBLEM_CLASS = @CEC2017_F9;
PROBLEM_NAME = 'CEC2017_F9';
DIMENSION = 30;

% Reference Stage 1 output - this initializes Stage 2 search
STAGE1_OUTPUT = sprintf('trained_AutoV_CEC2017_F1_D%d_stage1.mat', DIMENSION);
OUTPUT_FILE = sprintf('trained_AutoV_%s_D%d_stage2_from_f1.mat', PROBLEM_NAME, DIMENSION);

fprintf('\n%s\n', repmat('#',1,80));
fprintf('# Training AutoV Stage 2 on %s (D=%d)\n', PROBLEM_NAME, DIMENSION);
fprintf('# Initialized from Stage 1: %s\n', STAGE1_OUTPUT);
fprintf('# Output will be saved to: %s\n', OUTPUT_FILE);
fprintf('%s\n\n', repmat('#',1,80));

%% Check that Stage 1 output exists
if ~isfile(STAGE1_OUTPUT)
    error('Stage 1 output file not found: %s\n  Please run train_AutoV_cec2017_stage1_f1_D30 first.', STAGE1_OUTPUT);
end

%% Run training (Stage 2: initialized from Stage 1)
[best_operator, trainer_history, seeds, best_fitness, operator_family] = ...
    train_AutoV_cec2017_common(PROBLEM_CLASS, PROBLEM_NAME, DIMENSION, STAGE1_OUTPUT);

%% Save results
fprintf('Saving trained AutoV operator to: %s\n', OUTPUT_FILE);

best_operator_matrix = best_operator;  % Explicit naming for clarity

save(OUTPUT_FILE, ...
    'best_operator_matrix', ...
    'operator_family', ...
    'trainer_history', ...
    'seeds', ...
    'best_fitness', ...
    'PROBLEM_NAME', ...
    'PROBLEM_CLASS', ...
    'DIMENSION', ...
    'STAGE1_OUTPUT');

fprintf('Save complete.\n');
fprintf('\n%s\n', repmat('=',1,80));
fprintf('STAGE 2 COMPLETE\n');
fprintf('%s\n', repmat('=',1,80));
fprintf('Trained operator details:\n');
fprintf('  Operator family: %s\n', operator_family);
fprintf('  Parameter sets:  k=10\n');
fprintf('  Problem: %s (initialized from F1)\n', PROBLEM_NAME);
fprintf('  Dimension: %d\n', DIMENSION);
fprintf('  Best fitness achieved: %.6e\n', best_fitness);
fprintf('  Operator matrix (10x4): [w1, w2, w3, w4] for 10 parameter sets\n');
fprintf('\n');
fprintf('The operator is now trained and ready for use.\n');
fprintf('  Use load_trained_AutoV_and_run.m to apply it to other problems.\n');
fprintf('%s\n\n', repmat('=',1,80));
