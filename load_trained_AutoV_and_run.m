%% load_trained_AutoV_and_run.m
% Load a trained AutoV operator and run it on a chosen test problem
%
% This script:
% 1. Loads a saved AutoV operator (trained via stage 1 and/or stage 2)
% 2. Displays operator details and statistics
% 3. Runs the operator on a test problem (user-selectable)
% 4. Shows convergence results
%
% Usage:
%    load_trained_AutoV_and_run('trained_AutoV_CEC2017_F9_D30_stage2_from_f1.mat')

function load_trained_AutoV_and_run(operator_file, optional_problem_class)

%% Input handling
if nargin < 1 || isempty(operator_file)
    % Interactive selection if no file provided
    [file, path] = uigetfile('*.mat', 'Select trained AutoV operator file');
    if isequal(file, 0)
        fprintf('No file selected. Exiting.\n');
        return;
    end
    operator_file = fullfile(path, file);
end

if ~isfile(operator_file)
    error('Operator file not found: %s', operator_file);
end

% Add PlatEMO to path
addpath(genpath('/home/jona/github/PlatEMO/PlatEMO'));

%% Load operator
fprintf('\n%s\n', repmat('=',1,80));
fprintf('LOADING TRAINED AUTOV OPERATOR\n');
fprintf('%s\n', repmat('=',1,80));

load(operator_file);  % Loads: best_operator_matrix, operator_family, trainer_history, etc.

fprintf('Loaded from: %s\n\n', operator_file);
fprintf('Operator metadata:\n');
fprintf('  Family: %s\n', operator_family);
fprintf('  Parameter sets (k): %d\n', size(best_operator_matrix, 1));
fprintf('  Training problem: %s\n', PROBLEM_NAME);
fprintf('  Training dimension: %d\n', DIMENSION);
fprintf('  Best fitness achieved: %.6e\n', best_fitness);
fprintf('\n');

fprintf('Training history:\n');
fprintf('  Total evaluations: %d\n', trainer_history.num_evaluations);
fprintf('  Total generations: %d\n', trainer_history.generation_num);
fprintf('  Outer population size: 20\n');
if ~isempty(trainer_history.best_fitnesses)
    fprintf('  Initial best fitness: %.6e\n', trainer_history.best_fitnesses(1));
    fprintf('  Final best fitness:   %.6e\n', trainer_history.best_fitnesses(end));
    fprintf('  Improvement:          %.2f%%\n', ...
        (trainer_history.best_fitnesses(1) - trainer_history.best_fitnesses(end)) / ...
        trainer_history.best_fitnesses(1) * 100);
end
fprintf('\n');

fprintf('Operator parameter matrix (10 x 4):\n');
fprintf('  [w1=r1_coeff, w2=r2_sigma, w3=r2_mu, w4=probability]\n');
disp(best_operator_matrix);

%% Select test problem
if nargin >= 2 && ~isempty(optional_problem_class)
    test_problem_class = optional_problem_class;
    test_problem_name = func2str(test_problem_class);
else
    % Interactive selection
    fprintf('%s\n', repmat('-',1,80));
    fprintf('SELECT TEST PROBLEM\n');
    fprintf('%s\n', repmat('-',1,80));
    fprintf('Available CEC2017 problems: F1, F2, ..., F28\n');
    fprintf('Enter problem name (e.g., CEC2017_F1, CEC2017_F9): ');
    
    test_problem_name = input('', 's');
    if isempty(test_problem_name)
        test_problem_name = 'CEC2017_F9';  % Default
    end
    
    % Build class handle
    test_problem_class = str2func(test_problem_name);
end

fprintf('\nTest problem selected: %s\n\n', test_problem_name);

%% Configuration for evaluation
TEST_DIMENSION = 30;  % Fixed to match training
TEST_POP_SIZE = 30;   % Fixed to match training
TEST_MAX_FE = 3000;   % Fixed to match training
TEST_SEED = 42;       % Reproducible evaluation

fprintf('%s\n', repmat('=',1,80));
fprintf('RUNNING AUTOV ON TEST PROBLEM\n');
fprintf('%s\n', repmat('=',1,80));
fprintf('Test configuration:\n');
fprintf('  Problem: %s\n', test_problem_name);
fprintf('  Dimension: %d\n', TEST_DIMENSION);
fprintf('  Population size: %d\n', TEST_POP_SIZE);
fprintf('  Max evaluations: %d\n', TEST_MAX_FE);
fprintf('  Random seed: %d\n', TEST_SEED);
fprintf('\n');

%% Run the operator
try
    rng(TEST_SEED);
    
    % Create problem
    test_problem = feval(test_problem_class);
    test_problem.D = TEST_DIMENSION;
    test_problem.maxFE = TEST_MAX_FE;
    test_problem.outputFcn = @(~,~)[];  % Suppress logging
    
    % Use the operator
    Weight = best_operator_matrix;
    Fit = cumsum(Weight(:, 4));
    Fit = Fit ./ max(Fit);
    
    % Initialization
    population = test_problem.Initialization();
    [population, fitness] = EnvironmentalSelection(population, TEST_POP_SIZE);
    
    % Main loop
    best_fitness_history = [];
    eval_history = [];
    
    while test_problem.FE < TEST_MAX_FE
        % Record progress
        best_fitness_history = [best_fitness_history; min(fitness)];
        eval_history = [eval_history; test_problem.FE];
        
        % Tournament selection
        MatingPool = TournamentSelection(2, 2 * TEST_POP_SIZE, fitness);
        
        % Variation
        offspring = TSRIOperator(test_problem, Weight, Fit, population(MatingPool));
        
        % Environmental selection
        [population, fitness] = EnvironmentalSelection([population, offspring], TEST_POP_SIZE);
        
        if test_problem.FE >= TEST_MAX_FE
            break;
        end
    end
    
    % Final result
    final_best = min(fitness);
    
    fprintf('RESULTS:\n');
    fprintf('%s\n', repmat('-',1,80));
    fprintf('Final best fitness: %.6e\n', final_best);
    fprintf('Total evaluations: %d\n', test_problem.FE);
    fprintf('Total generations: %d\n', ceil(test_problem.FE / TEST_POP_SIZE));
    if ~isempty(best_fitness_history)
        fprintf('Initial fitness:    %.6e\n', best_fitness_history(1));
        improvement = (best_fitness_history(1) - final_best) / abs(best_fitness_history(1)) * 100;
        fprintf('Improvement:        %.2f%%\n', improvement);
    end
    fprintf('%s\n\n', repmat('-',1,80));
    
    % Plot convergence if available
    if ~isempty(best_fitness_history)
        figure('Name', sprintf('AutoV Convergence on %s', test_problem_name), ...
               'NumberTitle', 'off');
        
        % Log scale plot for better visualization
        semilogy(eval_history, best_fitness_history, 'b-', 'LineWidth', 2);
        xlabel('Function Evaluations');
        ylabel('Best Fitness (log scale)');
        title(sprintf('AutoV Operator Convergence on %s (D=%d)', test_problem_name, TEST_DIMENSION));
        grid on;
        
        % Add reference lines
        yline(final_best, 'r--', sprintf('Final: %.2e', final_best), 'Alpha', 0.7);
        yline(best_fitness_history(1), 'g--', sprintf('Initial: %.2e', best_fitness_history(1)), 'Alpha', 0.7);
        
        legend('Best Fitness', 'Location', 'best');
    end
    
    fprintf('\n%s\n', repmat('=',1,80));
    fprintf('EXECUTION COMPLETE\n');
    fprintf('%s\n\n', repmat('=',1,80));
    
catch ME
    fprintf('ERROR during execution: %s\n', ME.message);
    fprintf('Stack trace:\n');
    fprintf('%s\n', getReport(ME, 'extended'));
end

end
