function [best_operator, trainer_history, seeds, best_fitness, operator_family] = ...
    train_AutoV_cec2017_common(problem_class, problem_name, D, init_operator_file)
% train_AutoV_cec2017_common - Train AutoV operator on a CEC2017 single-objective problem
%
% This function implements a GA outer loop to tune the AutoV operator parameters.
% The operator is represented as k=10 parameter sets, each with 4 parameters.
% Each candidate operator is evaluated by running the inner solver 3 times 
% and taking the median of best objective values.
%
% Inputs:
%   problem_class        - Class handle (e.g., @CEC2017_F1)
%   problem_name         - String name for logging (e.g., 'CEC2017_F1')
%   D                    - Dimension (e.g., 30)
%   init_operator_file   - (Optional) Path to .mat file from stage 1 to initialize from
%                          If not provided, starts from random initialization
%
% Outputs:
%   best_operator        - Best operator matrix found, shape (10, 4)
%   trainer_history      - Struct with training history (fitnesses, evals, gens)
%   seeds                - Array of random seeds used
%   best_fitness         - Best fitness value found
%   operator_family      - String 'h3' (current AutoV operator family)

%% ========================================================================
%% OUTER TRAINER SETTINGS (easy to modify for fair budget control)
%% ========================================================================

% Trainer hyperparameters - OUTER (GA population) loop
TRAINER_POP_SIZE = 20;          % Reduced budget: 20 instead of 50
TRAINER_MAX_EVALS = 500;        % 500 candidate operator evaluations per stage
NUM_RUNS_PER_CANDIDATE = 3;     % Evaluate each operator 3 times, use median

% Inner AutoV settings - INNER (optimization) loop
INNER_POP_SIZE = 30;            % Population for inner solver
INNER_MAX_FE = 3000;            % Max function evaluations for inner solver
INNER_D = D;                    % Dimension matches

% Operator representation
K = 10;                         % Number of parameter sets (fixed)
operator_family = 'h3';         % TSRI operator: h3 = h(x1, x2, l, u)

% AutoV operator bounds (from original AutoV.m)
% Each operator has 4 parameters per set: [w1, w2, w3, w4]
PARAM_LOWER = repmat([0 0 -1 1e-6], K, 1);
PARAM_UPPER = repmat([1 1  1    1], K, 1);

SEED_BASE = 12345;              % Reproducibility

%% ========================================================================
%% Initialization
%% ========================================================================

fprintf('\n%s\n', repmat('=',1,80));
fprintf('AUTOV TRAINING: %s (D=%d)\n', problem_name, D);
fprintf('Operator family: %s with k=%d parameter sets\n', operator_family, K);
fprintf('%s\n', repmat('=',1,80));

fprintf('OUTER TRAINER (GA for operator search):\n');
fprintf('  Population size: %d\n', TRAINER_POP_SIZE);
fprintf('  Max evaluations: %d\n', TRAINER_MAX_EVALS);
fprintf('  Generations:    ~%d\n', ceil(TRAINER_MAX_EVALS / TRAINER_POP_SIZE));
fprintf('\n');

fprintf('INNER SOLVER (AutoV optimization on test problem):\n');
fprintf('  Population size: %d\n', INNER_POP_SIZE);
fprintf('  Max FE:          %d\n', INNER_MAX_FE);
fprintf('  Generations:     ~%d\n', INNER_MAX_FE / INNER_POP_SIZE);
fprintf('  Dimension:       %d\n', INNER_D);
fprintf('\n');

fprintf('CANDIDATE EVALUATION:\n');
fprintf('  Repeat count:    %d runs per candidate operator\n', NUM_RUNS_PER_CANDIDATE);
fprintf('  Fitness metric:  median of best objectives\n');
fprintf('\n');

fprintf('OPERATOR REPRESENTATION:\n');
fprintf('  Search space:    %d dimensions (%d sets × 4 params)\n', K*4, K);
fprintf('  Parameter bounds:\n');
fprintf('    w1 (TSRI r1 coeff)   in [%.4f, %.4f]\n', PARAM_LOWER(1,1), PARAM_UPPER(1,1));
fprintf('    w2 (TSRI r2 sigma)   in [%.4f, %.4f]\n', PARAM_LOWER(1,2), PARAM_UPPER(1,2));
fprintf('    w3 (TSRI r2 mu)      in [%.4f, %.4f]\n', PARAM_LOWER(1,3), PARAM_UPPER(1,3));
fprintf('    w4 (probability)     in [%.4f, %.4f]\n', PARAM_LOWER(1,4), PARAM_UPPER(1,4));
fprintf('%s\n\n', repmat('=',1,80));

% Initialize random state
rng_stream = RandStream('mt19937ar', 'Seed', SEED_BASE);

% Storage for history
seeds = zeros(TRAINER_MAX_EVALS, 1);
trainer_history = struct();
trainer_history.best_fitnesses = [];
trainer_history.mean_fitnesses = [];
trainer_history.generation_num = 0;
trainer_history.num_evaluations = 0;
trainer_history.operator_family = operator_family;
trainer_history.k = K;
trainer_history.inner_pop = INNER_POP_SIZE;
trainer_history.inner_maxfe = INNER_MAX_FE;
trainer_history.inner_D = INNER_D;

%% ========================================================================
%% Initialize population
%% ========================================================================

num_params = K * 4;

if nargin >= 4 && ~isempty(init_operator_file) && isfile(init_operator_file)
    % Stage 2: Load stage 1 results and use as seed
    fprintf('Initializing from stage 1 results: %s\n\n', init_operator_file);
    try
        load(init_operator_file, 'best_operator_matrix');
        init_vec = reshape(best_operator_matrix, 1, []);
        
        % Initialequationize with stage 1 result plus random variations
        population = zeros(TRAINER_POP_SIZE, num_params);
        population(1, :) = init_vec;  % Keep stage 1 best
        
        % Fill rest with mutations of stage 1 best
        for i = 2 : TRAINER_POP_SIZE
            mutation_strength = 0.15 * (reshape(PARAM_UPPER, 1, []) - reshape(PARAM_LOWER, 1, []));
            mutant = init_vec + mutation_strength .* randn(rng_stream, 1, num_params);
            
            % Clip to bounds
            mutant = max(reshape(PARAM_LOWER, 1, []), min(reshape(PARAM_UPPER, 1, []), mutant));
            population(i, :) = mutant;
        end
        fprintf('Initialized population with stage 1 best + mutations\n\n');
    catch ME
        fprintf('Warning: Could not load init file %s. Starting from random.\n', init_operator_file);
        fprintf('  Error: %s\n\n', ME.message);
        
        % Fall back to pure random
        random_vals = rand(rng_stream, TRAINER_POP_SIZE, num_params);
        population = repmat(reshape(PARAM_LOWER, 1, []), TRAINER_POP_SIZE, 1) + ...
                     random_vals .* repmat(reshape(PARAM_UPPER - PARAM_LOWER, 1, []), TRAINER_POP_SIZE, 1);
    end
else
    % Stage 1: Pure random initialization
    random_vals = rand(rng_stream, TRAINER_POP_SIZE, num_params);
    population = repmat(reshape(PARAM_LOWER, 1, []), TRAINER_POP_SIZE, 1) + ...
                 random_vals .* repmat(reshape(PARAM_UPPER - PARAM_LOWER, 1, []), TRAINER_POP_SIZE, 1);
    fprintf('Initialized population with random operators\n\n');
end

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
        operator_vec = population(i, :);
        operator_matrix = reshape(operator_vec, K, 4);  % Reshape to (10, 4)
        
        % Run inner solver with this operator, evaluate 3 times and take median
        run_fitnesses = zeros(NUM_RUNS_PER_CANDIDATE, 1);
        
        for run = 1 : NUM_RUNS_PER_CANDIDATE
            % Seed for reproducibility
            run_seed = SEED_BASE + eval_idx * 1000 + run;
            seeds(eval_idx) = run_seed;
            
            % Evaluate operator on problem
            run_fitnesses(run) = evaluate_autov_operator_on_problem(...
                problem_class, INNER_D, INNER_POP_SIZE, INNER_MAX_FE, ...
                operator_matrix, run_seed);
            
            fprintf('Eval %4d / Run %d / Gen %d: seed=%d, fitness=%.6e\n', ...
                eval_idx, run, generation, run_seed, run_fitnesses(run));
        end
        
        % Fitness is MEDIAN of the 3 runs (per AutoV paper)
        population_fitness(i) = median(run_fitnesses);
        fprintf('        --> MEDIAN fitness: %.6e\n', population_fitness(i));
        
    end % for i
    
    if eval_idx >= TRAINER_MAX_EVALS
        break;
    end
    
    %% Environmental selection: merge, sort, keep better half + elite
    [sorted_fitness, sorted_idx] = sort(population_fitness);
    
    % Keep best 50%
    num_keep = ceil(TRAINER_POP_SIZE / 2);
    best_fitness_this_gen = sorted_fitness(1);
    mean_fitness_this_gen = mean(population_fitness);
    
    trainer_history.best_fitnesses = [trainer_history.best_fitnesses; best_fitness_this_gen];
    trainer_history.mean_fitnesses = [trainer_history.mean_fitnesses; mean_fitness_this_gen];
    trainer_history.generation_num = generation;
    trainer_history.num_evaluations = eval_idx;
    
    fprintf('\n%s\n', repmat('-',1,80));
    fprintf('Generation %d: best=%.6e, mean=%.6e, evals=%d/%d\n', ...
        generation, best_fitness_this_gen, mean_fitness_this_gen, eval_idx, TRAINER_MAX_EVALS);
    fprintf('%s\n\n', repmat('-',1,80));
    
    % Create new population: keep best, add mutations
    best_population = population(sorted_idx(1:num_keep), :);
    new_population = zeros(TRAINER_POP_SIZE, num_params);
    
    % Elitism: keep top half
    new_population(1:num_keep, :) = best_population;
    
    % Fill rest with mutations of best
    for i = num_keep + 1 : TRAINER_POP_SIZE
        parent_idx = randi(rng_stream, num_keep);
        parent = best_population(parent_idx, :);
        
        % Gaussian mutation with adaptive strength
        mutation_strength = 0.20 * (reshape(PARAM_UPPER - PARAM_LOWER, 1, []));
        child = parent + mutation_strength .* randn(rng_stream, 1, num_params);
        
        % Clip to bounds
        lower_flat = reshape(PARAM_LOWER, 1, []);
        upper_flat = reshape(PARAM_UPPER, 1, []);
        child = max(lower_flat, min(upper_flat, child));
        new_population(i, :) = child;
    end
    
    population = new_population;
    population_fitness = zeros(TRAINER_POP_SIZE, 1);
    
end % for generation

%% ========================================================================
%% Extract best result
%% ========================================================================

[best_fitness, ~] = min(trainer_history.best_fitnesses);

% Find the best operator (from final population or history)
best_idx = find(population_fitness == min(population_fitness), 1, 'first');
if isempty(best_idx) || best_idx == 0
    best_idx = 1;
end
best_operator = reshape(population(best_idx, :), K, 4);

fprintf('\n%s\n', repmat('=',1,80));
fprintf('TRAINING COMPLETE\n');
fprintf('%s\n', repmat('=',1,80));
fprintf('Best fitness: %.6e\n', best_fitness);
fprintf('Total candidate evaluations: %d\n', eval_idx);
fprintf('Total inner solver runs: %d (= %d evals × %d runs)\n', ...
    eval_idx * NUM_RUNS_PER_CANDIDATE, eval_idx, NUM_RUNS_PER_CANDIDATE);
fprintf('Total generations: %d\n', trainer_history.generation_num);
fprintf('Best operator matrix (k=%d):\n', K);
disp(best_operator);
fprintf('%s\n\n', repmat('=',1,80));

end

%% ========================================================================
%% Helper: Evaluate AutoV operator on a problem
%% ========================================================================

function best_fitness = evaluate_autov_operator_on_problem(problem_class, D, pop_size, max_fe, operator_matrix, seed)
% Evaluate AutoV with a given operator matrix on the test problem
% 
% This directly implements the AutoV test-mode (Mode==1) loop with explicit
% operator matrix and problem configuration.
%
% Inputs:
%   problem_class   - Class handle (@CEC2017_F1, etc.)
%   D               - Dimension
%   pop_size        - Population size for inner AutoV solver
%   max_fe          - Max function evaluations for inner AutoV solver
%   operator_matrix - Operator parameters, shape (10, 4)
%   seed            - Random seed for reproducibility
%
% Output:
%   best_fitness    - Best objective value found (scalar)

rng(seed);

try
    % Create problem instance
    problem = feval(problem_class);
    problem.D = D;
    problem.maxFE = max_fe;
    
    % Suppress any output/logging during training
    problem.outputFcn = @(~,~)[];
    
    % Operator weights: shape (10, 4) with [w1, w2, w3, w4]
    Weight = operator_matrix;
    
    % Convert probability weights to roulette-wheel probabilities
    % 4th column is raw weight, normalize to cumsum for selection
    Fit = cumsum(Weight(:, 4));
    Fit = Fit ./ max(Fit);
    
    % INITIALIZATION: Create initial population
    population = problem.Initialization();
    [population, fitness] = EnvironmentalSelection(population, pop_size);
    
    % MAIN OPTIMIZATION LOOP
    % Run until max_fe is reached
    while problem.FE < max_fe
        % SELECTION: Binary tournament to create mating pool
        % Create a pool of 2*N individuals for 2-tournament selection
        MatingPool = TournamentSelection(2, 2 * pop_size, fitness);
        
        % VARIATION: Generate offspring using TSRI operator with these weights
        % TSRIOperator handles the probabilistic selection of parameter sets
        % and the stochastic variation based on w1, w2, w3, w4
        offspring = TSRIOperator(problem, Weight, Fit, population(MatingPool));
        
        % ENVIRONMENTAL SELECTION: Merge and select best pop_size individuals
        [population, fitness] = EnvironmentalSelection([population, offspring], pop_size);
        
        % Safeguard to prevent infinite loop if evaluations don't increment
        if problem.FE >= max_fe
            break;
        end
    end
    
    % Return best (minimum) objective value found
    best_fitness = min(fitness);
    
catch ME
    % If evaluation fails for any reason, return a large penalty
    fprintf('    ERROR in AutoV evaluation: %s\n', ME.message);
    best_fitness = 1e10;
end

end
