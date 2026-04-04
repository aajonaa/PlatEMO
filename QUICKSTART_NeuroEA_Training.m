%% QUICKSTART_NeuroEA_Training.m
% Quick start example for NeuroEA training and inference
%
% This script demonstrates the complete workflow:
% 1. Train NeuroEA on CEC2017 F1
% 2. Test the trained model on different problems
% 3. Show how to load and use saved models

clc; clear all;

fprintf('\n%s\n', repmat('=',1,70));
fprintf('NEUROEA TRAINING & INFERENCE QUICK START\n');
fprintf('%s\n\n', repmat('=',1,70));

%% =======================================================================
%% PART 1: TRAIN NEUROEA ON F1 (This takes 1-4 hours)
%% =======================================================================

fprintf('PART 1: Training phase\n');
fprintf('%s\n\n', repmat('-',1,70));

% Option A: Run full training (commented out - uncomment to train)
% fprintf('Warning: Full training takes 1-4 hours...\n');
% fprintf('Run: train_NeuroEA_cec2017_f1_D30\n\n');
% train_NeuroEA_cec2017_f1_D30

% Option B: Use pre-trained model (for demonstration)
% We'll create a dummy model file for this example
fprintf('For demonstration, we will:\n');
fprintf('1. Load architecture (no saved model file required)\n');
fprintf('2. Show how to run trained models\n\n');

%% =======================================================================
%% PART 2: LOAD AND RUN A TRAINED MODEL
%% =======================================================================

fprintf('PART 2: Inference example\n');
fprintf('%s\n\n', repmat('-',1,70));

fprintf('Assuming trained_NeuroEA_CEC2017_F1_D30.mat is available:\n\n');

fprintf('Example 1: Test f1-trained model on f4\n');
fprintf('  [fitness, pop, details] = load_trained_NeuroEA_and_run(...\n');
fprintf('      ''trained_NeuroEA_CEC2017_F1_D30.mat'', @CEC2017_F4);\n\n');

fprintf('Example 2: Test f1-trained model on f9 with custom settings\n');
fprintf('  [fitness, pop, details] = load_trained_NeuroEA_and_run(...\n');
fprintf('      ''trained_NeuroEA_CEC2017_F1_D30.mat'', @CEC2017_F9, ...\n');
fprintf('      30, 30, 3000, 12345);\n\n');

%% =======================================================================
%% PART 3: KEY CONFIGURATIONS
%% =======================================================================

fprintf('PART 3: Configuration details\n');
fprintf('%s\n\n', repmat('-',1,70));

fprintf('ARCHITECTURE (Paper-faithful, scaled for pop=30):\n');
fprintf('  11 blocks: [P, T1, T2, T3, E1, E2, E3, E4, C, M, S]\n');
fprintf('  55 tunable parameters total\n\n');

fprintf('OUTER TRAINER (configurable in train_NeuroEA_cec2017_common.m):\n');
fprintf('  Population size : 50\n');
fprintf('  Max evaluations : 5000\n');
fprintf('  Runs per candidate : 3 (average fitness)\n\n');

fprintf('INNER NEUROEA (per evaluation):\n');
fprintf('  Population size : 30\n');
fprintf('  Max function evals : 3000\n');
fprintf('  Dimension : 30\n\n');

%% =======================================================================
%% PART 4: TRAINING SCRIPT STATUS
%% =======================================================================

fprintf('PART 4: Available scripts\n');
fprintf('%s\n\n', repmat('-',1,70));

scripts = {
    'train_NeuroEA_cec2017_f1_D30.m', 'Train NeuroEA on CEC2017 F1 (sphere)'
    'train_NeuroEA_cec2017_f4_D30.m', 'Train NeuroEA on CEC2017 F4 (elliptic)'
    'train_NeuroEA_cec2017_f9_D30.m', 'Train NeuroEA on CEC2017 F9 (rastrigin)'
    'load_trained_NeuroEA_and_run.m', 'Load and run trained models'
    'train_NeuroEA_cec2017_common.m', 'Core training logic (called by above)'
    'train_setup_utils.m', 'Utility functions'
};

for i = 1:size(scripts,1)
    fprintf('  ✓ %s\n', scripts{i,1});
    fprintf('    → %s\n\n', scripts{i,2});
end

%% =======================================================================
%% PART 5: QUICK MANUAL TEST
%% =======================================================================

fprintf('PART 5: Manual test (verify setup)\n');
fprintf('%s\n\n', repmat('-',1,70));

fprintf('Testing block structure creation...\n');
try
    [Blocks, Graph] = train_setup_utils('create_blocks_graph');
    fprintf('✓ Blocks created: %d blocks\n', length(Blocks));
    fprintf('✓ Graph shape: %d x %d\n', size(Graph,1), size(Graph,2));
    fprintf('✓ Total parameters: %d\n', length(Blocks.lowers()));
    fprintf('✓ Status: Architecture ready for training\n\n');
catch err
    fprintf('✗ Error creating blocks: %s\n', err.message);
    fprintf('  Check that you are in the PlatEMO directory\n\n');
end

%% =======================================================================
%% SUMMARY
%% =======================================================================

fprintf('%s\n', repmat('=',1,70));
fprintf('SUMMARY\n');
fprintf('%s\n\n', repmat('=',1,70));

fprintf('To get started:\n\n');
fprintf('1. Review NEUROEA_TRAINING_README.md for complete documentation\n\n');

fprintf('2. For quick training on one problem:\n');
fprintf('   >> train_NeuroEA_cec2017_f1_D30\n');
fprintf('   (Saves trained_NeuroEA_CEC2017_F1_D30.mat)\n\n');

fprintf('3. To test the trained model:\n');
fprintf('   >> [fit, pop, details] = load_trained_NeuroEA_and_run(...\n');
fprintf('          ''trained_NeuroEA_CEC2017_F1_D30.mat'', @CEC2017_F4);\n\n');

fprintf('4. Common modifications (in train_NeuroEA_cec2017_common.m):\n');
fprintf('   - Change TRAINER_POP_SIZE (default 50)\n');
fprintf('   - Change TRAINER_MAX_EVALS (default 5000)\n');
fprintf('   - Change NUM_RUNS_PER_CANDIDATE (default 3)\n');
fprintf('   - Change SEED_BASE (default 12345)\n\n');

fprintf('Happy training!\n');
fprintf('%s\n\n', repmat('=',1,70));
