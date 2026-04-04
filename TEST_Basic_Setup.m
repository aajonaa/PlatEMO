%% TEST_Basic_Setup.m
% Simple test to verify basic architecture creation works
% This should run quickly and verify the setup

try
    fprintf('Testing basic setup...\n');
    
    % Test 1: Check block classes exist
    fprintf('  ✓ Checking block classes...\n');
    b = Block_Population();
    fprintf('    Block_Population: OK\n');
    
    b = Block_Tournament(60, 10);
    fprintf('    Block_Tournament: OK\n');
    
    b = Block_Exchange(3);
    fprintf('    Block_Exchange: OK\n');
    
    b = Block_Crossover(2, 5);
    fprintf('    Block_Crossover: OK\n');
    
    b = Block_Mutation(5);
    fprintf('    Block_Mutation: OK\n');
    
    b = Block_Selection(30);
    fprintf('    Block_Selection: OK\n');
    
    % Test 2: Create blocks and graph
    fprintf('  ✓ Creating architecture...\n');
    [Blocks, Graph] = train_setup_utils('create_blocks_graph');
    fprintf('    Architecture created: %d blocks\n', length(Blocks));
    fprintf('    Graph: %d x %d\n', size(Graph, 1), size(Graph, 2));
    
    % Test 3: Get parameters
    fprintf('  ✓ Checking parameters...\n');
    params = Blocks.lowers();
    fprintf('    Lower bounds: %d parameters\n', length(params));
    
    params = Blocks.uppers();
    fprintf('    Upper bounds: %d parameters\n', length(params));
    
    % Test 4: Check CEC2017 problems
    fprintf('  ✓ Checking CEC2017 problems...\n');
    p = CEC2017_F1();
    fprintf('    CEC2017_F1: OK\n');
    
    p = CEC2017_F4();
    fprintf('    CEC2017_F4: OK\n');
    
    p = CEC2017_F9();
    fprintf('    CEC2017_F9: OK\n');
    
    fprintf('\n✓ ALL TESTS PASSED\n');
    fprintf('Setup is ready for training.\n');
    
catch err
    fprintf('\n✗ ERROR: %s\n', err.message);
    fprintf('Location: %s, line %d\n', err.stack(1).file, err.stack(1).line);
end
