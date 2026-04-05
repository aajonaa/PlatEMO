function [best_params, trainer_history, seeds, best_fitness] = train_NeuroEA_cec2017_combined(problem_classes, problem_names, D)
% train_NeuroEA_cec2017_combined - Train NeuroEA on multiple CEC2017 problems
%
% This function implements a GA outer loop to tune the 55 block parameters of NeuroEA.
% The fitness of each candidate solution is the MEAN best objective across:
%   - All three problems (F1, F4, F9)
%   - N independent NeuroEA runs per problem
%
% This creates a SINGLE model that generalizes well across different problem types,
% following the paper's multi-problem training approach.
%
% Inputs:
%   problem_classes - Cell array of class handles (e.g., {@CEC2017_F1, @CEC2017_F4, @CEC2017_F9})
%   problem_names   - Cell array of problem names for logging
%   D               - Dimension (e.g., 30)
%
% Outputs:
%   best_params     - Best concatenated block parameters found (55 values)
%   trainer_history - Struct with training history
%   seeds           - Array of random seeds used for reproducibility
%   best_fitness    - Best avg fitness value found (mean across all problems)

%% ========================================================================
%% OUTER TRAINER SETTINGS (easy to modify)
%% ========================================================================

% Trainer hyperparameters
TRAINER_POP_SIZE = 50;          % Paper uses 50
TRAINER_MAX_EVALS = 5000;       % Paper uses 5000
NUM_RUNS_PER_CANDIDATE = 2;     % Reduced to 2 (per problem) since we have 3 problems now
                                 % Total: 3 problems × 2 runs = 6 FE evaluations per candidate
SEED_BASE = 12345;              % Reproducibility

% Inner NeuroEA settings (fixed)
INNER_POP_SIZE = 30;
INNER_MAX_FE = 3000;
INNER_D = D;

%% ========================================================================
%% Initialization
%% ========================================================================

fprintf('\n%s\n', repmat('=',1,70));
fprintf('NEUROEA TRAINING: COMBINED MULTI-PROBLEM (Generalization)\n');
fprintf('Problems: %s, %s, %s (D=%d)\n', problem_names{1}, problem_names{2}, problem_names{3}, D);
fprintf('%s\n', repmat('=',1,70));
fprintf('Training approach: Multi-problem (single model for all 3 problems)\n');
fprintf('Outer trainer: GA with pop=%d, max_evals=%d\n', TRAINER_POP_SIZE, TRAINER_MAX_EVALS);
fprintf('Inner NeuroEA: pop=%d, max_FE=%d, runs_per_problem=%d\n', ...
    INNER_POP_SIZE, INNER_MAX_FE, NUM_RUNS_PER_CANDIDATE);
fprintf('Fitness metric: MEAN best objective across all 3 problems\n');
fprintf('Total FE per candidate: 3 problems × %d runs × %d FE = %d\n', ...
    NUM_RUNS_PER_CANDIDATE, INNER_MAX_FE, 3 * NUM_RUNS_PER_CANDIDATE * INNER_MAX_FE);
fprintf('%s\n\n', repmat('=',1,70));

% Create blocks and graph (paper-faithful architecture)
[Blocks, Graph] = train_setup_utils('create_blocks_graph');

% Get parameter bounds
param_lowers = Blocks.lowers();
param_uppers = Blocks.uppers();
num_params = length(param_lowers);

fprintf('Total tunable parameters: %d\n', num_params);
fprintf('Parameter bounds: [%.4f, %.4f] (min,max)\n\n', min(param_lowers), max(param_uppers));

% Initialize random state
rng_stream = RandStream('mt19937ar', 'Seed', SEED_BASE);

% Storage
seed_count = 0;
seeds = zeros(TRAINER_MAX_EVALS * 3 * NUM_RUNS_PER_CANDIDATE, 1);
trainer_history = struct();
trainer_history.best_fitnesses = [];
trainer_history.mean_fitnesses = [];
trainer_history.generation_num = 0;
trainer_history.num_evaluations = 0;

% Initial population
population = repmat(param_lowers, TRAINER_POP_SIZE, 1) + ...
             rand(rng_stream, TRAINER_POP_SIZE, num_params) .* ...
             repmat(param_uppers - param_lowers, TRAINER_POP_SIZE, 1);

population_fitness = zeros(TRAINER_POP_SIZE, 1);

%% ========================================================================
%% Outer GA Loop
%% ========================================================================

eval_idx = 0;

for generation = 1 : ceil(TRAINER_MAX_EVALS / TRAINER_POP_SIZE)
    
    %% Evaluate population
    for i = 1 : size(population, 1)
        if eval_idx >= TRAINER_MAX_EVALS
            break;
        end
        
        eval_idx = eval_idx + 1;
        param_vector = population(i, :);
        
        % Evaluate on all three problems
        problem_fitnesses = zeros(length(problem_classes), NUM_RUNS_PER_CANDIDATE);
        
        for prob_idx = 1 : length(problem_classes)
            problem_class = problem_classes{prob_idx};
            problem_name = problem_names{prob_idx};
            
            for run = 1 : NUM_RUNS_PER_CANDIDATE
                seed_count = seed_count + 1;
                run_seed = SEED_BASE + eval_idx * 10000 + prob_idx * 1000 + run;
                seeds(seed_count) = run_seed;
                
                % Run NeuroEA on this problem with this parameter vector
                problem_fitnesses(prob_idx, run) = evaluate_neuroea_on_problem(...
                    problem_class, INNER_D, INNER_POP_SIZE, INNER_MAX_FE, ...
                    Graph, param_vector, run_seed);
                
                fprintf('Eval %4d / Prob %s / Run %d / Gen %d: fitness=%.6e\n', ...
                    eval_idx, problem_name, run, generation, problem_fitnesses(prob_idx, run));
            end
        end
        
        % Fitness = mean of all problems and all runs
        population_fitness(i) = mean(problem_fitnesses(:));
        
    end % for i
    
    if eval_idx >= TRAINER_MAX_EVALS
        break;
    end
    
    %% Selection and reproduction (simple GA: keep top 50%, reuse to fill population)
    [sorted_fitness, sorted_idx] = sort(population_fitness);
    
    % Keep best
    best_fitness_this_gen = sorted_fitness(1);
    best_population = population(sorted_idx(1 : ceil(TRAINER_POP_SIZE / 2)), :);
    
    % Basic statistics
    mean_fitness_this_gen = mean(population_fitness);
    
    trainer_history.best_fitnesses = [trainer_history.best_fitnesses; best_fitness_this_gen];
    trainer_history.mean_fitnesses = [trainer_history.mean_fitnesses; mean_fitness_this_gen];
    trainer_history.generation_num = generation;
    trainer_history.num_evaluations = eval_idx;
    
    fprintf('\nGeneration %d: best=%.6e, mean=%.6e, evals=%d/%d\n', ...
        generation, best_fitness_this_gen, mean_fitness_this_gen, eval_idx, TRAINER_MAX_EVALS);
    fprintf('%s\n\n', repmat('-',1,70));
    
    % Create new population by mutation of best
    new_population = zeros(TRAINER_POP_SIZE, num_params);
    new_population(1:size(best_population,1), :) = best_population;  % Elitism
    
    % Fill rest with mutated copies
    for i = size(best_population, 1) + 1 : TRAINER_POP_SIZE
        parent_idx = randi(rng_stream, size(best_population, 1));
        parent = best_population(parent_idx, :);
        
        % Gaussian mutation
        mutation_strength = 0.2 * (param_uppers - param_lowers);
        child = parent + mutation_strength .* randn(rng_stream, 1, num_params);
        
        % Clip to bounds
        child = max(param_lowers, min(param_uppers, child));
        new_population(i, :) = child;
    end
    
    population = new_population;
    population_fitness = zeros(TRAINER_POP_SIZE, 1);
    
end % for generation

%% ========================================================================
%% Extract best result
%% ========================================================================

[best_fitness, best_idx] = min(trainer_history.best_fitnesses);
best_params = population(1, :);  % Best found

% Trim seeds array to actual size used
seeds = seeds(1:seed_count);

fprintf('\n%s\n', repmat('=',1,70));
fprintf('TRAINING COMPLETE\n');
fprintf('%s\n', repmat('=',1,70));
fprintf('Best mean fitness (across all 3 problems): %.6e\n', best_fitness);
fprintf('Total evaluations: %d\n', eval_idx);
fprintf('Total generations: %d\n', trainer_history.generation_num);
fprintf('Total seeds used: %d\n', seed_count);
fprintf('%s\n\n', repmat('=',1,70));

end

%% ========================================================================
%% Helper: Evaluate NeuroEA on a single problem with specific parameters
%% ========================================================================

function best_fitness = evaluate_neuroea_on_problem(problem_class, D, pop_size, max_fe, Graph, param_vector, seed)
% Evaluate NeuroEA with given block parameters on the test problem

rng(seed);

% Create problem instance - CEC2017 problems have fixed D, so we use them as-is
problem = feval(problem_class);
problem.maxFE = max_fe;

% Create fresh blocks for this evaluation
Blocks_fresh = [
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

% Set parameters on fresh blocks
Blocks_fresh.ParameterSet(param_vector);

% Create NeuroEA with the configured blocks
algo = NeuroEA('parameter', {Blocks_fresh, Graph});

% Run the algorithm via standard PlatEMO interface
try
    algo.Solve(problem);
    
    % Get best fitness from result
    if ~isempty(algo.result) && size(algo.result, 2) >= 2
        final_pop = algo.result{end, 2};
        if ~isempty(final_pop)
            best_obj = min(final_pop.objs);
            best_fitness = best_obj;
        else
            best_fitness = inf;
        end
    else
        best_fitness = inf;
    end
catch err
    fprintf('Warning: Error in NeuroEA evaluation: %s\n', err.message);
    best_fitness = inf;
end

end
