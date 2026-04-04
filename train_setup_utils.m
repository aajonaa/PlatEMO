%% train_setup_utils.m
% Utility functions for NeuroEA training with paper-faithful architecture
% scaled for population size 30
%
% Used by: train_NeuroEA_cec2017_f*.m

function varargout = train_setup_utils(action, varargin)
%train_setup_utils - Utility functions for NeuroEA training setup
%
% Usage:
%   [Blocks, Graph] = train_setup_utils('create_blocks_graph')
%   trainer = train_setup_utils('create_ga_trainer', template_pop_size, max_param_evals)

switch action
    case 'create_blocks_graph'
        [varargout{1}, varargout{2}] = create_blocks_graph();
    case 'create_ga_trainer'
        varargout{1} = create_ga_trainer(varargin{1}, varargin{2});
    otherwise
        error('Unknown action: %s', action);
end
end

%% ========================================================================
function [Blocks, Graph] = create_blocks_graph()
% Create paper-faithful NeuroEA architecture, scaled for pop=30
%
% Paper largest architecture pattern (with pop=100):
%   [Population, Tournament, Tournament, Tournament, Exchange, Exchange, Exchange, Exchange, 
%    Crossover, Mutation, Selection]
% Scaled hyperparameters for pop=30:
%   Tournament blocks: Block_Tournament(60,10)    % 3 copies
%   Exchange blocks:   Block_Exchange(3)           % 4 copies
%   Crossover block:   Block_Crossover(2,5)
%   Mutation block:    Block_Mutation(5)
%   Selection block:   Block_Selection(30)
%   Population block:  Block_Population()

fprintf('\n=== Creating NeuroEA Architecture ===\n');
fprintf('Architecture: Paper-faithful largest (11 blocks), scaled pop=30\n');
fprintf('Node order: [P, T1, T2, T3, E1, E2, E3, E4, C, M, S]\n');

% Create blocks in order
Blocks = [
    Block_Population()
    Block_Tournament(60, 10)    % T1: 60 parents, tournament size [1,10]
    Block_Tournament(60, 10)    % T2
    Block_Tournament(60, 10)    % T3
    Block_Exchange(3)           % E1: exchange 3 parents
    Block_Exchange(3)           % E2
    Block_Exchange(3)           % E3
    Block_Exchange(3)           % E4
    Block_Crossover(2, 5)       % C: 2 parents, 5 weight sets
    Block_Mutation(5)           % M: 5 weight sets
    Block_Selection(30)         % S: keep 30 best
];

% Adjacency matrix (11x11)
% Row i, Col j: if Graph(i,j) = w, then node j receives from node i with fraction w
Graph = [
    0    1    1    1    0    0    0    0    0    0    1  ;  % P -> T1,T2,T3,S
    0    0    0    0    0.25 0.25 0.25 0.25 0    0    0  ;  % T1 -> E1,E2,E3,E4
    0    0    0    0    0.25 0.25 0.25 0.25 0    0    0  ;  % T2 -> E1,E2,E3,E4
    0    0    0    0    0.25 0.25 0.25 0.25 0    0    0  ;  % T3 -> E1,E2,E3,E4
    0    0    0    0    0    0    0    0    1    0    0  ;  % E1 -> C
    0    0    0    0    0    0    0    0    1    0    0  ;  % E2 -> C
    0    0    0    0    0    0    0    0    1    0    0  ;  % E3 -> C
    0    0    0    0    0    0    0    0    1    0    0  ;  % E4 -> C
    0    0    0    0    0    0    0    0    0    1    0  ;  % C -> M
    0    0    0    0    0    0    0    0    0    0    1  ;  % M -> S
    1    0    0    0    0    0    0    0    0    0    0  ;  % S -> P
];

fprintf('  Population block: Block_Population() [0 params]\n');
fprintf('  Tournament blocks (3x): Block_Tournament(60,10) [1 param each = 3 total]\n');
fprintf('  Exchange blocks (4x):   Block_Exchange(3) [3 params each = 12 total]\n');
fprintf('  Crossover block:        Block_Crossover(2,5) [30 params]\n');
fprintf('  Mutation block:         Block_Mutation(5) [10 params]\n');
fprintf('  Selection block:        Block_Selection(30) [0 params]\n');
fprintf('  Total tunable parameters: 3 + 12 + 30 + 10 = 55\n\n');
end

%% ========================================================================
function trainer = create_ga_trainer(template_pop_size, max_param_evals)
% Create a simple GA trainer configuration for outer loop
%
% The GA will tune the concatenated block parameters (55 total).
% Fitness is evaluated as: mean best objective over N independent NeuroEA runs

if nargin < 1
    template_pop_size = 50;  % Paper: 50
end
if nargin < 2
    max_param_evals = 5000;  % Paper: 5000
end

trainer = struct();
trainer.pop_size = template_pop_size;
trainer.max_candidate_evals = max_param_evals;
trainer.num_runs_per_candidate = 3;  % Average over 3 independent runs
trainer.name = sprintf('GA_trainer_pop%d_maxFE%d', template_pop_size, max_param_evals);

fprintf('\n=== Training Configuration ===\n');
fprintf('Outer loop: GA tuning block parameters\n');
fprintf('  Population size: %d\n', trainer.pop_size);
fprintf('  Max candidate evaluations: %d\n', trainer.max_candidate_evals);
fprintf('  Runs per candidate: %d (average fitness)\n', trainer.num_runs_per_candidate);
fprintf('Inner loop: NeuroEA optimization\n');
fprintf('  Population size: 30\n');
fprintf('  Max function evaluations: 3000\n');
fprintf('  Dimension: 30\n\n');
end
