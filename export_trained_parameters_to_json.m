%% export_trained_parameters_to_json.m
% Export trained NeuroEA parameters to JSON for Python/other languages
%
% This script converts the .mat file to JSON format for easier portability

clear; clc;

fprintf('\n%s\n', repmat('=', 1, 80));
fprintf('EXPORT TRAINED NEUROEA PARAMETERS TO JSON\n');
fprintf('%s\n', repmat('=', 1, 80));

% Configuration
TRAINED_MODEL_FILE = 'trained_NeuroEA_F9_D30_stage2_from_f1.mat';
OUTPUT_JSON_FILE = 'trained_neuroea_params.json';

%% Load trained model
fprintf('\nLoading trained model from: %s\n', TRAINED_MODEL_FILE);

if ~isfile(TRAINED_MODEL_FILE)
    error('Trained model file not found: %s', TRAINED_MODEL_FILE);
end

model_data = load(TRAINED_MODEL_FILE);

%% Extract key parameters
fprintf('\nExtracting parameters...\n');

% Best parameters
best_params_stage2 = model_data.best_params_stage2(:);  % Column vector
best_params_stage1 = model_data.best_params_stage1(:);

% Fitness values
best_fitness_stage1 = model_data.best_fitness_stage1;
best_fitness_stage2 = model_data.best_fitness_stage2;

% Graph (11x11 adjacency matrix)
Graph = model_data.Graph;

% Problem configuration
DIMENSION = model_data.DIMENSION;
INNER_POP = model_data.INNER_POP;
INNER_GEN = model_data.INNER_GEN;
PROBLEM_NAME = model_data.PROBLEM_NAME{1};

fprintf('  Stage 1 fitness: %.6e\n', best_fitness_stage1);
fprintf('  Stage 2 fitness: %.6e\n', best_fitness_stage2);
fprintf('  Total parameters: %d\n', length(best_params_stage2));

%% Create JSON structure
fprintf('\nCreating JSON structure...\n');

json_struct = struct();

% Metadata
json_struct.metadata = struct();
json_struct.metadata.algorithm = 'NeuroEA';
json_struct.metadata.training_approach = 'Transfer Learning (F1 -> F9)';
json_struct.metadata.stage1_problem = 'CEC2017_F1';
json_struct.metadata.stage2_problem = PROBLEM_NAME;
json_struct.metadata.dimension = DIMENSION;
json_struct.metadata.population_size = INNER_POP;
json_struct.metadata.generations = INNER_GEN;
json_struct.metadata.max_fe_per_run = INNER_POP * INNER_GEN;
json_struct.metadata.export_date = string(datetime('now'));

% Architecture description
json_struct.architecture = struct();
json_struct.architecture.num_blocks = 11;
json_struct.architecture.block_names = {
    'P', 'T1', 'T2', 'T3', 'E1', 'E2', 'E3', 'E4', 'C', 'M', 'S'
};
json_struct.architecture.block_descriptions = {
    'Population', ...
    'Tournament_1', 'Tournament_2', 'Tournament_3', ...
    'Exchange_1', 'Exchange_2', 'Exchange_3', 'Exchange_4', ...
    'Crossover', 'Mutation', 'Selection'
};
json_struct.architecture.num_parameters = length(best_params_stage2);

% Trained parameters
json_struct.trained_parameters = struct();
json_struct.trained_parameters.stage1 = best_params_stage1';  % Convert to row vector for JSON
json_struct.trained_parameters.stage2 = best_params_stage2';
json_struct.trained_parameters.num_stage2_params = length(best_params_stage2);

% Fitness values
json_struct.fitness = struct();
json_struct.fitness.stage1 = best_fitness_stage1;
json_struct.fitness.stage2 = best_fitness_stage2;
json_struct.fitness.stage1_problem = 'CEC2017_F1';
json_struct.fitness.stage2_problem = PROBLEM_NAME;

% Graph (connectivity matrix)
json_struct.connectivity = struct();
json_struct.connectivity.graph = Graph;
json_struct.connectivity.graph_description = 'Adjacency matrix (11x11) for block connections';
json_struct.connectivity.block_order = json_struct.architecture.block_names;

% Hyperparameter ranges
json_struct.hyperparameter_ranges = struct();
json_struct.hyperparameter_ranges.epoch = struct('min', 1, 'max', 100000, 'default', 100);
json_struct.hyperparameter_ranges.pop_size = struct('min', 5, 'max', 10000, 'default', 30);
json_struct.hyperparameter_ranges.c1 = struct('min', 0.0, 'max', 1.0, 'description', 'Crossover rate');
json_struct.hyperparameter_ranges.m1 = struct('min', 0.0, 'max', 1.0, 'description', 'Mutation rate');
json_struct.hyperparameter_ranges.tournament_size = struct('min', 2, 'max', 100, 'default', 10);

%% Write to JSON file
fprintf('\nWriting to JSON file: %s\n', OUTPUT_JSON_FILE);

try
    json_str = jsonencode(json_struct);
    fid = fopen(OUTPUT_JSON_FILE, 'w');
    fprintf(fid, json_str);
    fclose(fid);
    
    % Pretty-print (add indentation)
    % Note: jsonencode in newer MATLAB can use prettify, but we do it manually
    json_str_formatted = strrep(json_str, ',"', sprintf(',\n  "'));
    json_str_formatted = strrep(json_str_formatted, '{"', sprintf('{\n  "'));
    json_str_formatted = strrep(json_str_formatted, ':[{', sprintf(': [\n    {'));
    fid = fopen(OUTPUT_JSON_FILE, 'w');
    fprintf(fid, json_str_formatted);
    fprintf(fid, '\n');
    fclose(fid);
    
    fprintf('✓ JSON export successful!\n');
    
catch err
    fprintf('Warning: Could not create pretty JSON, saving raw format\n');
    fid = fopen(OUTPUT_JSON_FILE, 'w');
    fprintf(fid, json_str);
    fclose(fid);
    fprintf('✓ JSON export successful (raw format)\n');
end

%% Summary
fprintf('\n%s\n', repmat('=', 1, 80));
fprintf('EXPORT SUMMARY\n');
fprintf('%s\n', repmat('=', 1, 80));
fprintf('\nOutput file: %s\n', OUTPUT_JSON_FILE);
fprintf('File size: %d bytes\n', dir(OUTPUT_JSON_FILE).bytes);
fprintf('\nJSON structure contains:\n');
fprintf('  - Metadata (algorithm info, dimensions, budget)\n');
fprintf('  - Architecture (11-block NeuroEA structure)\n');
fprintf('  - Trained parameters (Stage 1 and Stage 2)\n');
fprintf('  - Fitness values (Stage 1 and Stage 2)\n');
fprintf('  - Connectivity graph (block connections)\n');
fprintf('  - Hyperparameter ranges (for fine-tuning)\n');

fprintf('\nUsage in Python:\n');
fprintf('  import json\n');
fprintf('  with open(''%s'') as f:\n', OUTPUT_JSON_FILE);
fprintf('    params = json.load(f)\n');
fprintf('  best_params = params[''trained_parameters''][''stage2'']\n');

fprintf('\n%s\n\n', repmat('=', 1, 80));
