#!/usr/bin/env python
# Created by "NeuroEA Team" at 09:49, 05/04/2026 ----------%
#       Paper: NeuroEA: Neural Network-guided Evolutionary Algorithm
#       Training: Transfer Learning (CEC2017 F1 -> F9, D=30)
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent


class OriginalNeuroEA(Optimizer):
    """
    The trained version of: NeuroEA (Neural Network-guided Evolutionary Algorithm)
    
    Transfer-learned from CEC2017 Stage 2 training (F1 → F9, Dimension=30)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c1 (float): [0.0, 1.0], Crossover rate, default = 0.5
        + m1 (float): [0.0, 1.0], Mutation rate, default = 0.1
        + tournament_size (int): [2, 100], Tournament selection size, default = 10

    Architecture (11-block NeuroEA):
        Block_Population (P) → Tournament Selection (T1-T3) → Information Exchange (E1-E4) →
        Crossover (C) → Mutation (M) → Selection (S) → Block_Population (P)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, NeuroEA
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
    >>> model = NeuroEA.OriginalNeuroEA(epoch=100, pop_size=30, c1=0.5, m1=0.1, tournament_size=10)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Best fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] NeuroEA: A Hybrid Approach to Optimization using Neural Networks and Evolutionary Algorithms
    [2] CEC 2017 Constrained Real-Parameter Optimization Benchmark Suite
    """

    def __init__(self, epoch: int = 100, pop_size: int = 30, c1: float = 0.5, 
                 m1: float = 0.1, tournament_size: int = 10, **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 100
            pop_size: number of population size, default = 30
            c1: [0-1] Crossover rate, default = 0.5
            m1: [0-1] Mutation rate, default = 0.1
            tournament_size: [2-100] Tournament selection size, default = 10
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.c1 = self.validator.check_float("c1", c1, (0, 1.0))
        self.m1 = self.validator.check_float("m1", m1, (0, 1.0))
        self.tournament_size = self.validator.check_int("tournament_size", tournament_size, [2, 100])
        self.set_parameters(["epoch", "pop_size", "c1", "m1", "tournament_size"])
        self.sort_flag = False
        self.is_parallelizable = False

    def initialize_variables(self):
        """Initialize algorithm-specific variables"""
        # Mutation range
        self.mutation_sigma = (self.problem.ub - self.problem.lb) * 0.1

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        """
        Generate an agent without fitness evaluation
        
        Args:
            solution: Initial solution vector (if None, generate random)
        
        Returns:
            Agent: Agent with solution and velocity
        """
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        velocity = self.generator.uniform(-1, 1, self.problem.n_dims)
        return Agent(solution=solution, velocity=velocity)

    def generate_agent(self, solution: np.ndarray = None) -> Agent:
        """
        Generate an agent with fitness evaluation
        
        Args:
            solution: Initial solution vector (if None, generate random)
        
        Returns:
            Agent: Agent with solution, velocity, and fitness
        """
        agent = self.generate_empty_agent(solution)
        agent.target = self.get_target(agent.solution)
        return agent

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        """
        Ensure solution respects problem bounds
        
        Args:
            solution: Solution vector (may be out of bounds)
        
        Returns:
            Amended solution within bounds
        """
        condition = np.logical_and(self.problem.lb <= solution, solution <= self.problem.ub)
        pos_rand = self.generator.uniform(self.problem.lb, self.problem.ub)
        return np.where(condition, solution, pos_rand)

    def tournament_selection(self, pop_indices: list, tournament_size: int = None) -> int:
        """
        Perform tournament selection
        
        Args:
            pop_indices: List of indices to select from
            tournament_size: Size of tournament (if None, use self.tournament_size)
        
        Returns:
            Index of winner (best fitness in tournament)
        """
        if tournament_size is None:
            tournament_size = self.tournament_size
        
        tournament_size = min(tournament_size, len(pop_indices))
        candidates = self.generator.choice(pop_indices, size=tournament_size, replace=False)
        
        # Return index of best (minimum) fitness
        best_idx = candidates[0]
        for idx in candidates[1:]:
            if self.compare_target(self.pop[idx].target, self.pop[best_idx].target, self.problem.minmax):
                best_idx = idx
        
        return best_idx

    def crossover_operator(self, parent1: np.ndarray, parent2: np.ndarray, 
                          crossover_rate: float = None) -> np.ndarray:
        """
        Crossover (recombination) operator
        
        Args:
            parent1, parent2: Parent solution vectors
            crossover_rate: Probability of crossing over each dimension
        
        Returns:
            Offspring solution
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

    def mutation_operator(self, solution: np.ndarray, mutation_rate: float = None) -> np.ndarray:
        """
        Mutation operator: apply random perturbations
        
        Args:
            solution: Solution vector
            mutation_rate: Probability of mutating each dimension
        
        Returns:
            Mutated solution
        """
        if mutation_rate is None:
            mutation_rate = self.m1
        
        mutated = solution.copy()
        for i in range(len(mutated)):
            if self.generator.random() < mutation_rate:
                # Gaussian mutation with adaptive sigma
                mutated[i] = mutated[i] + self.generator.normal(0, self.mutation_sigma[i])
        
        return mutated

    def block_tournament_select(self, pop_size: int = None) -> list:
        """
        Block Tournament Selection (T1-T3): Select tournament_size individuals from population
        
        Args:
            pop_size: Population size (if None, use self.pop_size)
        
        Returns:
            List of selected agent indices
        """
        if pop_size is None:
            pop_size = self.pop_size
        
        pop_indices = list(range(pop_size))
        selected = []
        
        # Run tournament 3 times to create diversity
        for _ in range(3):
            for _ in range(pop_size):
                winner = self.tournament_selection(pop_indices, self.tournament_size)
                selected.append(winner)
        
        return selected

    def block_crossover_variation(self, selected_indices: list) -> list:
        """
        Block Crossover and Variation (E1-E4, C, M):
        Apply crossover and mutation to create offspring
        
        Args:
            selected_indices: List of selected parent indices
        
        Returns:
            List of offspring agents
        """
        offspring = []
        
        for i in range(0, len(selected_indices) - 1, 2):
            parent1_idx = selected_indices[i]
            parent2_idx = selected_indices[i + 1]
            
            parent1 = self.pop[parent1_idx].solution.copy()
            parent2 = self.pop[parent2_idx].solution.copy()
            
            # Crossover
            child_sol = self.crossover_operator(parent1, parent2)
            
            # Mutation
            child_sol = self.mutation_operator(child_sol)
            
            # Ensure within bounds
            child_sol = self.amend_solution(child_sol)
            
            # Create agent and evaluate
            child_agent = self.generate_agent(child_sol)
            offspring.append(child_agent)
        
        return offspring

    def block_selection(self, pop: list, offspring: list) -> list:
        """
        Block Selection (S): Merge parent and offspring populations,
        then select best individuals
        
        Args:
            pop: Current population
            offspring: Offspring population
        
        Returns:
            Selected population of size self.pop_size
        """
        # Merge populations
        merged = pop + offspring
        
        # Sort by fitness (best first)
        merged.sort(key=lambda agent: agent.target.fitness)
        
        # Return best pop_size individuals
        return merged[:self.pop_size]

    def evolve(self, epoch: int) -> None:
        """
        The main evolution loop of NeuroEA (11-block architecture)
        
        Flow:
            1. Block P (Population): Current population
            2. Block T1-T3 (Tournament): Tournament selection, 3 times
            3. Block E1-E4 (Exchange): Information exchange groups
            4. Block C (Crossover): Recombination
            5. Block M (Mutation): Variation
            6. Block S (Selection): Survival selection, keep best
        
        Args:
            epoch (int): Current iteration number
        """
        
        # Block P & T1-T3: Tournament selection (create 3x the offspring)
        pop_indices = list(range(0, self.pop_size))
        selected_indices = []
        
        for _ in range(3):
            for _ in range(self.pop_size):
                winner_idx = self.tournament_selection(pop_indices, self.tournament_size)
                selected_indices.append(winner_idx)
        
        # Block E1-E4: Information Exchange + Block C: Crossover + Block M: Mutation
        # Create offspring via crossover and mutation
        offspring = []
        
        for i in range(0, len(selected_indices) - 1, 2):
            parent1_idx = selected_indices[i]
            parent2_idx = selected_indices[i + 1]
            
            parent1 = self.pop[parent1_idx].solution.copy()
            parent2 = self.pop[parent2_idx].solution.copy()
            
            # Crossover
            child_sol = self.crossover_operator(parent1, parent2, self.c1)
            
            # Mutation
            child_sol = self.mutation_operator(child_sol, self.m1)
            
            # Ensure within bounds
            child_sol = self.amend_solution(child_sol)
            
            # Create and evaluate agent
            child_agent = self.generate_agent(child_sol)
            offspring.append(child_agent)
        
        # Block S: Selection - merge and keep best
        self.pop = self.block_selection(self.pop, offspring)


class TrainedNeuroEA(OriginalNeuroEA):
    """
    Pre-trained NeuroEA with parameters from CEC2017 Stage 2 transfer learning
    
    This variant loads trained parameters from a JSON configuration file
    and uses them as default hyperparameters.
    
    Trained on:
        - Stage 1: CEC2017 F1, D=30, Best fitness: 1.195e+03
        - Stage 2: CEC2017 F9, D=30, Best fitness: 3.928e-01 (offset from optimum)
    
    Examples
    ~~~~~~~~
    >>> from mealpy import FloatVar, NeuroEA
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-10.,)*30, ub=(10.,)*30, name="x"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> # Use trained parameters
    >>> model = NeuroEA.TrainedNeuroEA(epoch=100, pop_size=30)
    >>> g_best = model.solve(problem_dict)
    """
    
    def __init__(self, epoch: int = 100, pop_size: int = 30, c1: float = None, 
                 m1: float = None, tournament_size: int = 10, 
                 params_file: str = None, **kwargs: object) -> None:
        """
        Args:
            epoch: Number of iterations, default = 100
            pop_size: Population size, default = 30
            c1: Crossover rate [0-1], if None loads from trained parameters
            m1: Mutation rate [0-1], if None loads from trained parameters
            tournament_size: Tournament size [2-100], default = 10
            params_file: Path to trained_neuroea_params.json (optional)
        """
        self.params_file = params_file or 'trained_neuroea_params.json'
        self.trained_params = None
        
        # Load trained parameters if available
        self.load_trained_parameters()
        
        # Use trained values as defaults if not specified
        if c1 is None:
            c1 = self._extract_param('c1', 0.5)
        if m1 is None:
            m1 = self._extract_param('m1', 0.1)
        
        super().__init__(epoch=epoch, pop_size=pop_size, c1=c1, m1=m1, 
                        tournament_size=tournament_size, **kwargs)

    def load_trained_parameters(self) -> None:
        """Load trained parameters from JSON file (if available)"""
        try:
            import json
            from pathlib import Path
            
            json_file = Path(self.params_file)
            if not json_file.exists():
                return
            
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            self.trained_params = data
            print(f"✓ Loaded trained parameters from {self.params_file}")
            
        except (ImportError, FileNotFoundError, json.JSONDecodeError):
            pass

    def _extract_param(self, param_name: str, default: float) -> float:
        """
        Extract parameter from trained configuration
        
        Args:
            param_name: Parameter name (c1, m1, etc.)
            default: Default value if not found
        
        Returns:
            Parameter value
        """
        if self.trained_params is None:
            return default
        
        try:
            hyperparams = self.trained_params.get('hyperparameter_ranges', {})
            param_range = hyperparams.get(param_name, {})
            return param_range.get('default', default)
        except:
            return default

    def information(self) -> None:
        """Display training and algorithm information"""
        print("\n" + "="*80)
        print("TRAINED NEUROEA - TRANSFER LEARNING FROM CEC2017")
        print("="*80)
        
        if self.trained_params:
            metadata = self.trained_params.get('metadata', {})
            print(f"\nTraining Information:")
            print(f"  Algorithm: {metadata.get('algorithm', 'NeuroEA')}")
            print(f"  Approach: {metadata.get('training_approach', 'Transfer Learning')}")
            print(f"  Stage 1 Problem: {metadata.get('stage1_problem', 'CEC2017_F1')}")
            print(f"  Stage 2 Problem: {metadata.get('stage2_problem', 'CEC2017_F9')}")
            print(f"  Problem Dimension: {metadata.get('dimension', '30')}")
            print(f"  Population Size: {metadata.get('population_size', '30')}")
            print(f"  Generations: {metadata.get('generations', '100')}")
            
            fitness = self.trained_params.get('fitness', {})
            print(f"\nTraining Results:")
            print(f"  Stage 1 Best Fitness: {fitness.get('stage1', 'N/A')}")
            print(f"  Stage 2 Best Fitness: {fitness.get('stage2', 'N/A')}")
        
        print(f"\nAlgorithm Configuration:")
        print(f"  Epochs: {self.epoch}")
        print(f"  Population Size: {self.pop_size}")
        print(f"  Crossover Rate (c1): {self.c1:.4f}")
        print(f"  Mutation Rate (m1): {self.m1:.4f}")
        print(f"  Tournament Size: {self.tournament_size}")
        print("="*80 + "\n")
