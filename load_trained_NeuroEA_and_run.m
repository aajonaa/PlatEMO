%% load_trained_NeuroEA_and_run.m
% Load a trained NeuroEA model and run it on a test problem
%
% Usage:
%   [best_fitness, final_pop] = load_trained_NeuroEA_and_run(model_file, test_problem_class, D, pop_size, max_fe, seed)
%
% Inputs:
%   model_file           - Path to .mat file with saved trained model
%   test_problem_class   - Class handle for test problem (e.g., @CEC2017_F1)
%   D                    - Dimension for test problem (optional, default=30)
%   pop_size             - Population size (optional, default=30)
%   max_fe               - Max function evaluations (optional, default=3000)
%   seed                 - Random seed (optional, default=42)
%
% Outputs:
%   best_fitness         - Best objective value achieved
%   final_pop            - Final population object
%
% Example 1: Run trained f1 model on f4 test problem
%   load_trained_NeuroEA_and_run('trained_NeuroEA_CEC2017_F1_D30.mat', @CEC2017_F4);
%
% Example 2: Run with custom settings
%   load_trained_NeuroEA_and_run('trained_NeuroEA_CEC2017_F1_D30.mat', @CEC2017_F9, 30, 30, 3000, 999);

function [best_fitness, final_pop, details] = load_trained_NeuroEA_and_run(model_file, test_problem_class, varargin)

%% Parse arguments
if nargin < 2
    error('Usage: load_trained_NeuroEA_and_run(model_file, test_problem_class, [D, pop_size, max_fe, seed])');
end

% Optional arguments with defaults
D = 30;
pop_size = 30;
max_fe = 3000;
seed = 42;

if nargin >= 3 && ~isempty(varargin{1})
    D = varargin{1};
end
if nargin >= 4 && ~isempty(varargin{2})
    pop_size = varargin{2};
end
if nargin >= 5 && ~isempty(varargin{3})
    max_fe = varargin{3};
end
if nargin >= 6 && ~isempty(varargin{4})
    seed = varargin{4};
end

%% Load trained model
fprintf('\n%s\n', repmat('=',1,70));
fprintf('LOADING TRAINED NEUROEA MODEL\n');
fprintf('%s\n', repmat('=',1,70));
fprintf('Model file: %s\n', model_file);

if ~exist(model_file, 'file')
    error('Model file not found: %s', model_file);
end

load(model_file, 'Blocks', 'Graph', 'best_params', 'PROBLEM_NAME', 'DIMENSION');

fprintf('Loaded from training on: %s (D=%d)\n', PROBLEM_NAME, DIMENSION);

%% Create test problem
test_problem = feval(test_problem_class);
test_problem.D = D;
test_problem.maxFE = max_fe;

fprintf('\nTest problem: %s\n', class(test_problem));
fprintf('Test dimension: %d\n', D);
fprintf('Test population size: %d\n', pop_size);
fprintf('Test max FE: %d\n', max_fe);
fprintf('Random seed: %d\n\n', seed);

%% Set block parameters from trained model
Blocks_test = [
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
Blocks_test.ParameterSet(best_params);

%% Run NeuroEA with trained parameters
rng(seed);

fprintf('%s\n', repmat('=',1,70));
fprintf('RUNNING TRAINED NEUROEA ON TEST PROBLEM\n');
fprintf('%s\n\n', repmat('=',1,70));

% Create algorithm
algo = NeuroEA('parameter', {Blocks_test, Graph});

% Solve
start_time = tic;
algo.Solve(test_problem);
elapsed_time = toc(start_time);

%% Extract results
if ~isempty(algo.result)
    final_pop = algo.result{end, 2};
    best_fitness = min(final_pop.objs);
else
    final_pop = [];
    best_fitness = inf;
end

%% Display results
fprintf('%s\n', repmat('=',1,70));
fprintf('INFERENCE COMPLETE\n');
fprintf('%s\n', repmat('=',1,70));
fprintf('Best fitness: %.6e\n', best_fitness);
fprintf('Function evaluations used: %d / %d\n', test_problem.FE, test_problem.maxFE);
fprintf('Runtime: %.2f seconds\n', elapsed_time);
fprintf('%s\n\n', repmat('=',1,70));

%% Build details structure
details = struct();
details.best_fitness = best_fitness;
details.final_population = final_pop;
details.function_evals = test_problem.FE;
details.runtime = elapsed_time;
details.trained_on = PROBLEM_NAME;
details.test_on = class(test_problem);
details.seed = seed;
details.D = D;

end
