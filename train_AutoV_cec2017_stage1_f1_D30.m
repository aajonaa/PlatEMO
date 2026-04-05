%% train_AutoV_cec2017_stage1_f1_D30.m
% Train AutoV on CEC2017 F1 (shifted sphere function) with D=30
% Stage 1 of 2-stage training
%
% This script:
% 1. Trains AutoV operator by tuning k=10 parameter sets via outer GA
% 2. Evaluates each operator by running inner solver 3 times, taking median
% 3. Uses outer budget: 500 candidate operators (20 pop × ~25 gen)
% 4. Uses inner budget: D=30, pop=30, maxFE=3000 per evaluation
% 5. Saves the trained operator matrix and full training history
%
% Results can be loaded and used with load_trained_AutoV_and_run.m
% Stage 2 continues from this stage 1 result on F9.

clear; clc;

% Add PlatEMO to path (including AutoV helpers)
addpath(genpath('/home/jona/github/PlatEMO/PlatEMO'));

%% Configuration
PROBLEM_CLASS = @CEC2017_F1;
PROBLEM_NAME = 'CEC2017_F1';
DIMENSION = 30;
OUTPUT_FILE = sprintf('trained_AutoV_%s_D%d_stage1.mat', PROBLEM_NAME, DIMENSION);

fprintf('\n%s\n', repmat('#',1,80));
fprintf('# Training AutoV Stage 1 on %s (D=%d)\n', PROBLEM_NAME, DIMENSION);
fprintf('# Output will be saved to: %s\n', OUTPUT_FILE);
fprintf('# This result will be used to initialize Stage 2 (F9 training)\n');
fprintf('%s\n\n', repmat('#',1,80));

%% Run training (Stage 1: no initialization file)
[best_operator, trainer_history, seeds, best_fitness, operator_family] = ...
    train_AutoV_cec2017_common(PROBLEM_CLASS, PROBLEM_NAME, DIMENSION);

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
    'DIMENSION');

fprintf('Save complete.\n');
fprintf('\n%s\n', repmat('=',1,80));
fprintf('STAGE 1 COMPLETE\n');
fprintf('%s\n', repmat('=',1,80));
fprintf('Trained operator details:\n');
fprintf('  Operator family: %s\n', operator_family);
fprintf('  Parameter sets:  k=10\n');
fprintf('  Problem: %s\n', PROBLEM_NAME);
fprintf('  Dimension: %d\n', DIMENSION);
fprintf('  Best fitness achieved: %.6e\n', best_fitness);
fprintf('  Operator matrix (10x4): [w1, w2, w3, w4] for 10 parameter sets\n');
fprintf('\n');
fprintf('Next step:\n');
fprintf('  Run: train_AutoV_cec2017_stage2_f9_D30_from_f1\n');
fprintf('  This will load the stage 1 best operator above and continue training on F9\n');
fprintf('%s\n\n', repmat('=',1,80));
