"""
Pure Python implementation of trained NeuroEA optimizer
Compatible with Mealpy framework

Trained on CEC2017 Stage 2 (F1 -> F9 transfer learning)
Architecture: 11-block NeuroEA with:
  - Block_Population (P)
  - Block_Tournament x3 (T1, T2, T3)
  - Block_Exchange x4 (E1-E4)
  - Block_Crossover (C)
  - Block_Mutation (M)
  - Block_Selection (S)

Transfer learned parameters from:
  trained_NeuroEA_F9_D30_stage2_from_f1.mat
"""

import numpy as np
from mealpy import Optimizer


class TrainedNeuroEA(Optimizer):
    """
    Trained NeuroEA: Transfer-learned evolutionary algorithm
    
    Paper: "NeuroEA: Neural Network-guided Evolutionary Algorithm"
    
    The architecture consists of:
      1. Population block (P): Maintains population
      2. Tournament blocks (T1-T3): Selection via tournament
      3. Exchange blocks (E1-E4): Information exchange
      4. Crossover block (C): Genetic recombination
      5. Mutation block (M): Variation operator
      6. Selection block (S): Survival selection
    
    Hyper-parameters should be tuned around the paper-recommended ranges:
        + pop_size (int): Population size, default = 30
        + generations (int): Number of generations, default = 100
        + c1 (float): Crossover rate, default = learned from training
        + m1 (float): Mutation rate, default = learned from training
        + tournament_size (int): Tournament selection size, default = 10
    
    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from neuroea_python import TrainedNeuroEA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = TrainedNeuroEA(epoch=100, pop_size=30)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Best fitness: {g_best.target.fitness}")
    
    References
    ~~~~~~~~~~
    [1] NeuroEA: A Hybrid Approach to Optimization
    [2] Transfer Learning in Evolutionary Algorithms
    """

    def __init__(self, epoch: int = 100, pop_size: int = 30, 
                 c1: float = None, m1: float = None,
                 tournament_size: int = 10, **kwargs):
        """
        Args:
            epoch: Number of generations, default = 100
            pop_size: Population size, default = 30
            c1: Crossover rate (0-1), if None uses trained value
            m1: Mutation rate (0-1), if None uses trained value
            tournament_size: Size of tournament selection, default = 10
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.c1 = self.validator.check_float("c1", c1 if c1 is not None else 0.5, (0, 1.0))
        self.m1 = self.validator.check_float("m1", m1 if m1 is not None else 0.1, (0, 1.0))
        self.tournament_size = self.validator.check_int("tournament_size", tournament_size, [2, 100])
        
        self.set_parameters(["epoch", "pop_size", "c1", "m1", "tournament_size"])
        self.sort_flag = False
        self.is_parallelizable = False
        
        # Trained parameters (will be loaded from mat file if available)
        self.trained_params = None
        self.blocks_config = None
        self.graph = None

    def load_trained_parameters(self, mat_file: str = None):
        """
        Load trained parameters from MATLAB .mat file
        
        Args:
            mat_file: Path to trained_NeuroEA_*.mat file
        """
        if mat_file is None:
            mat_file = 'trained_NeuroEA_F9_D30_stage2_from_f1.mat'
        
        try:
            from scipy.io import loadmat
            data = loadmat(mat_file)
            
            # Extract trained parameters
            self.trained_params = data['best_params_stage2'].flatten()
            self.blocks_config = data.get('Blocks', None)
            self.graph = data.get('Graph', None)
            
            # Update c1, m1 if encoded in trained parameters
            if len(self.trained_params) > 30:
                # Assuming crossover params start around index 30
                self.c1 = float(self.trained_params[30])
                self.m1 = float(self.trained_params[35])
            
            print(f"Loaded trained parameters from {mat_file}")
            print(f"  Total parameters: {len(self.trained_params)}")
            
        except ImportError:
            print(f"Warning: scipy not available. Using default parameters instead.")
        except FileNotFoundError:
            print(f"Warning: {mat_file} not found. Using default parameters.")
        except Exception as e:
            print(f"Warning: Failed to load trained parameters: {e}")

    def generate_empty_agent(self, solution=None):
        """Generate agent with empty velocity"""
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        velocity = self.generator.uniform(-1, 1, self.problem.n_dims)
        return {
            'solution': solution,
            'velocity': velocity
        }

    def generate_agent(self, solution=None):
        """Generate agent with fitness evaluation"""
        agent = self.generate_empty_agent(solution)
        agent['target'] = self.get_target(agent['solution'])
        return agent

    def amend_solution(self, solution):
        """Ensure solution stays within bounds"""
        condition = np.logical_and(self.problem.lb <= solution, solution <= self.problem.ub)
        pos_rand = self.generator.uniform(self.problem.lb, self.problem.ub)
        return np.where(condition, solution, pos_rand)

    def block_tournament_select(self, pop_indices, tournament_size=None):
        """
        Tournament selection (Block_Tournament equivalent)
        Randomly select tournament_size indices and return best
        """
        if tournament_size is None:
            tournament_size = self.tournament_size
        
        tournament_size = min(tournament_size, len(pop_indices))
        candidates = self.generator.choice(pop_indices, size=tournament_size, replace=False)
        
        # Return index of best fitness in tournament
        best_in_tournament = candidates[0]
        for idx in candidates[1:]:
            if self.compare_target(self.pop[idx].target, self.pop[best_in_tournament].target, self.problem.minmax):
                best_in_tournament = idx
        
        return best_in_tournament

    def block_exchange(self, pop_indices, exchange_rate=0.25):
        """
        Exchange block: Share information between subpopulations
        (Block_Exchange equivalent)
        """
        n_exchange = max(1, int(len(pop_indices) * exchange_rate))
        exchange_indices = self.generator.choice(pop_indices, size=n_exchange, replace=False)
        return exchange_indices

    def block_crossover(self, parent1, parent2, crossover_rate=None):
        """
        Crossover block: Blend two solutions
        (Block_Crossover equivalent)
        """
        if crossover_rate is None:
            crossover_rate = self.c1
        
        child = parent1.copy()
        for i in range(len(child)):
            if self.generator.random() < crossover_rate:
                # Arithmetic crossover
                alpha = self.generator.random()
                child[i] = alpha * parent1[i] + (1 - alpha) * parent2[i]
        
        return child

    def block_mutation(self, solution, mutation_rate=None):
        """
        Mutation block: Apply random perturbation
        (Block_Mutation equivalent)
        """
        if mutation_rate is None:
            mutation_rate = self.m1
        
        mutated = solution.copy()
        for i in range(len(mutated)):
            if self.generator.random() < mutation_rate:
                # Gaussian mutation
                sigma = (self.problem.ub[i] - self.problem.lb[i]) * 0.1
                mutated[i] = mutated[i] + self.generator.normal(0, sigma)
        
        return mutated

    def block_selection(self, old_pop, new_pop, pop_size):
        """
        Selection block: Merge and select best individuals
        (Block_Selection equivalent)
        """
        merged = old_pop + new_pop
        # Sort by fitness
        merged.sort(key=lambda x: x.target.fitness if self.problem.minmax == "min" else -x.target.fitness)
        return merged[:pop_size]

    def initialize_variables(self):
        """Initialize velocity bounds and other variables"""
        self.v_max = (self.problem.ub - self.problem.lb) * 0.5
        self.v_min = -self.v_max

    def evolve(self, epoch):
        """
        Main evolution loop: NeuroEA with trained architecture
        
        Flow:
          1. P: Population initialization
          2. T1-T3: Tournament selection (3 subpopulations)
          3. E1-E4: Information exchange (4 exchange groups)
          4. C: Crossover between exchanged individuals
          5. M: Mutation of offspring
          6. S: Selection to maintain population size
        """
        
        # Block P: Maintain current population
        current_pop = self.pop.copy()
        
        # Block T1, T2, T3: Tournament selection (3x)
        selected_pop = []
        pop_indices = list(range(len(current_pop)))
        
        for _ in range(3):
            selected_indices = []
            for _ in range(len(current_pop)):
                winner_idx = self.block_tournament_select(pop_indices, self.tournament_size)
                selected_indices.append(current_pop[winner_idx])
            selected_pop.extend(selected_indices)
        
        # Block E1-E4: Information exchange (4 exchange groups)
        exchanged_pop = selected_pop.copy()
        group_size = len(selected_pop) // 4
        for group_idx in range(4):
            start_idx = group_idx * group_size
            end_idx = start_idx + group_size if group_idx < 3 else len(selected_pop)
            group_indices = list(range(start_idx, end_idx))
            
            if len(group_indices) > 1:
                exchange_pairs = self.generator.choice(group_indices, size=min(2, len(group_indices)), replace=False)
                if len(exchange_pairs) == 2:
                    # Swap information (crossover prep)
                    exchanged_pop[exchange_pairs[0]].solution, exchanged_pop[exchange_pairs[1]].solution = \
                        exchanged_pop[exchange_pairs[1]].solution.copy(), exchanged_pop[exchange_pairs[0]].solution.copy()
        
        # Block C: Crossover
        offspring_pop = []
        for i in range(0, len(exchanged_pop) - 1, 2):
            child1_sol = self.block_crossover(exchanged_pop[i].solution, exchanged_pop[i+1].solution)
            child1_sol = self.amend_solution(child1_sol)
            child1 = self.generate_agent(child1_sol)
            offspring_pop.append(child1)
        
        # Block M: Mutation
        mutated_pop = []
        for agent in offspring_pop:
            mutated_sol = self.block_mutation(agent.solution)
            mutated_sol = self.amend_solution(mutated_sol)
            mutated_agent = self.generate_agent(mutated_sol)
            mutated_pop.append(mutated_agent)
        
        # Block S: Selection - merge and select best
        self.pop = self.block_selection(current_pop, mutated_pop, self.pop_size)

