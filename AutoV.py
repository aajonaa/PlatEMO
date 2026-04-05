#!/usr/bin/env python
# Created by "AutoV Team" at 09:49, 06/04/2026 ----------%
#       Paper: Principled design of translation, scale, and rotation invariant
#              variation operators for metaheuristics
#       Training: Two-stage training (CEC2017 F1 -> F9, D=30)
#       Updated: 2026-04-06 (1000 candidate operators budget)
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent

# ========================================================================
# HARDCODED TRAINED OPERATOR PARAMETERS FROM STAGE 2 TRANSFER LEARNING
# ========================================================================
# Training Problem: CEC2017 F9 (D=30)
# Transfer Source: CEC2017 F1 (D=30, Stage 1)
# Operator Family: h3 (TSRI - Translation, Scale, Rotation Invariant)
# Parameter Sets: k = 10
# Parameters per set: [w1, w2, w3, w4] = [40 dimensions]
# ========================================================================

# IMPORTANT: These are PLACEHOLDER values. Replace with actual trained values
# from trained_AutoV_CEC2017_F9_D30_stage2_from_f1.mat after training completes.
TRAINED_OPERATOR_STAGE2 = np.array([
    # Placeholder: Replace with actual trained operator matrix (k=10 sets × 4 params)
    # Format: [w1_1, w2_1, w3_1, w4_1, w1_2, w2_2, w3_2, w4_2, ..., w1_10, w2_10, w3_10, w4_10]
    0.5, 0.5, 0.0, 0.1,    # Set 1
    0.5, 0.5, 0.0, 0.1,    # Set 2
    0.5, 0.5, 0.0, 0.1,    # Set 3
    0.5, 0.5, 0.0, 0.1,    # Set 4
    0.5, 0.5, 0.0, 0.1,    # Set 5
    0.5, 0.5, 0.0, 0.1,    # Set 6
    0.5, 0.5, 0.0, 0.1,    # Set 7
    0.5, 0.5, 0.0, 0.1,    # Set 8
    0.5, 0.5, 0.0, 0.1,    # Set 9
    0.5, 0.5, 0.0, 0.1,    # Set 10
])

TRAINED_FITNESS_STAGE2_F9 = 1e10  # Placeholder - update after training
TRAINED_FITNESS_STAGE1_F1 = 1e10  # Placeholder - update after training

# Operator family and configuration
OPERATOR_FAMILY = 'h3'  # TSRI operator
OPERATOR_K = 10  # Number of parameter sets
OPERATOR_PARAMS_PER_SET = 4  # [w1, w2, w3, w4]


class OriginalAutoV(Optimizer):
    """
    The original AutoV algorithm: Automated design of variation operators.
    
    Uses the TSRI (Translation, Scale, Rotation Invariant) operator family:
        o_i = r1 * (u_i - l_i) + r2 * x2_i + (1 - r2) * x1_i
    where:
        r1 ~ N(0, w1^2)
        r2 ~ N(w3, w2^2)
        x1, x2 are parent solutions
        u, l are upper/lower bounds
    
    The operator is parameterized by k=10 sets of [w1, w2, w3, w4],
    where w4 is a probability weight for roulette-wheel selection.

    Hyper-parameters to fine-tune:
        + pop_size (int): [10, 100], population size, default = 30
        + tournament_size (int): [2, 20], tournament selection size, default = 2

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, AutoV
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-100.,) * 30, ub=(100.,) * 30, name="x"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = AutoV.OriginalAutoV(epoch=100, pop_size=30, tournament_size=2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Best fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Y. Tian, X. Zhang, C. He, K. C. Tan, and Y. Jin. Principled design of
        translation, scale, and rotation invariant variation operators for
        metaheuristics. Chinese Journal of Electronics, 2023, 32(1): 111-129.
    [2] CEC2017: Problem definitions and evaluation criteria for the CEC 2017
        competition on constrained real-parameter optimization.
    """

    def __init__(self, epoch: int = 100, pop_size: int = 30, 
                 tournament_size: int = 2, operator_params: np.ndarray = None,
                 **kwargs: object) -> None:
        """
        Args:
            epoch: Maximum number of iterations, default = 100
            pop_size: Population size, default = 30
            tournament_size: [2-20] Tournament selection size, default = 2
            operator_params: Operator parameter matrix (k×4), if None use default
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.tournament_size = self.validator.check_int("tournament_size", tournament_size, [2, 100])
        
        # Operator parameters: 10 sets × 4 params each
        if operator_params is None:
            # Default: equal distribution of probability
            self.operator_params = np.ones((OPERATOR_K, OPERATOR_PARAMS_PER_SET)) * 0.5
            self.operator_params[:, 3] = 1.0 / OPERATOR_K  # Equal probability weights
        else:
            self.operator_params = operator_params.reshape(OPERATOR_K, OPERATOR_PARAMS_PER_SET)
        
        self.set_parameters(["epoch", "pop_size", "tournament_size"])
        self.sort_flag = False
        self.is_parallelizable = False

    def initialize_variables(self):
        """Initialize algorithm-specific variables"""
        # Pre-compute roulette wheel probabilities from w4 values
        self.prob_weights = self.operator_params[:, 3]  # Extract probability column
        self.prob_cumsum = np.cumsum(self.prob_weights)
        self.prob_cumsum = self.prob_cumsum / self.prob_cumsum[-1]  # Normalize

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        """Generate agent without fitness evaluation"""
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        return Agent(solution=solution)

    def generate_agent(self, solution: np.ndarray = None) -> Agent:
        """Generate agent with fitness evaluation"""
        agent = self.generate_empty_agent(solution)
        agent.target = self.get_target(agent.solution)
        return agent

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        """Ensure solution respects problem bounds"""
        condition = np.logical_and(self.problem.lb <= solution, solution <= self.problem.ub)
        pos_rand = self.generator.uniform(self.problem.lb, self.problem.ub)
        return np.where(condition, solution, pos_rand)

    def tournament_selection(self, fitness_list: np.ndarray, size: int = None) -> int:
        """
        Binary tournament selection: select best from random pair
        
        Args:
            fitness_list: Fitness values of population
            size: Tournament size (if None, use self.tournament_size)
        
        Returns:
            Index of selected individual
        """
        if size is None:
            size = self.tournament_size
        
        size = min(size, len(fitness_list))
        candidates = self.generator.choice(len(fitness_list), size=size, replace=False)
        
        # Return index with best (minimum) fitness
        best_idx = candidates[0]
        for idx in candidates[1:]:
            if fitness_list[idx] < fitness_list[best_idx]:
                best_idx = idx
        return best_idx

    def select_operator_set(self) -> int:
        """
        Select an operator parameter set using roulette wheel selection
        based on probability weights (w4)
        
        Returns:
            Index of selected operator set (0 to k-1)
        """
        rand_val = self.generator.random()
        for i, cum_prob in enumerate(self.prob_cumsum):
            if rand_val <= cum_prob:
                return i
        return OPERATOR_K - 1

    def tsri_operator(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        TSRI operator: Translation, Scale, Rotation Invariant variation
        
        o_i = r1 * (u_i - l_i) + r2 * parent2_i + (1 - r2) * parent1_i
        
        where:
            r1 ~ N(0, w1^2)
            r2 ~ N(w3, w2^2)
            w1, w2, w3, w4 from selected operator parameter set
        
        Args:
            parent1: First parent solution
            parent2: Second parent solution
        
        Returns:
            Offspring solution
        """
        # Select operator parameter set via roulette wheel
        op_idx = self.select_operator_set()
        w1, w2, w3, w4 = self.operator_params[op_idx]
        
        # Bounds range
        bounds_range = self.problem.ub - self.problem.lb
        
        # Sample random variables
        r1 = self.generator.normal(0, w1, self.problem.n_dims)  # N(0, w1^2)
        r2 = self.generator.normal(w3, w2, self.problem.n_dims)  # N(w3, w2^2)
        
        # TSRI formula
        offspring = r1 * bounds_range + r2 * parent2 + (1 - r2) * parent1
        
        return offspring

    def evolve(self, epoch: int) -> None:
        """
        Main evolution loop using TSRI operator and binary tournament selection
        
        Flow:
            1. Binary tournament selection: create mating pool
            2. Generate offspring using TSRI operator
            3. Environmental selection: merge parents and offspring, keep best
        
        Args:
            epoch: Current iteration number
        """
        # Binary tournament selection: create 2*pop_size individuals for mating
        fit_array = np.array([agent.target.fitness for agent in self.pop])
        mating_pool = []
        
        for _ in range(2 * self.pop_size):
            winner = self.tournament_selection(fit_array, self.tournament_size)
            mating_pool.append(winner)
        
        # Generate offspring using TSRI operator
        offspring = []
        for i in range(0, len(mating_pool) - 1, 2):
            parent1_idx = mating_pool[i]
            parent2_idx = mating_pool[i + 1]
            
            parent1 = self.pop[parent1_idx].solution.copy()
            parent2 = self.pop[parent2_idx].solution.copy()
            
            # Apply TSRI operator
            child_sol = self.tsri_operator(parent1, parent2)
            
            # Ensure within bounds
            child_sol = self.amend_solution(child_sol)
            
            # Evaluate offspring
            child_agent = self.generate_agent(child_sol)
            offspring.append(child_agent)
        
        # Environmental selection: merge and keep best pop_size
        merged = self.pop + offspring
        merged.sort(key=lambda agent: agent.target.fitness)
        self.pop = merged[:self.pop_size]


class TrainedAutoV(OriginalAutoV):
    """
    Pre-trained AutoV with operator parameters from CEC2017 Stage 2 transfer learning
    
    Transfer Learning Setup:
      - Stage 1: Trained on CEC2017 F1 (Shifted Sphere, D=30)
      - Stage 2: Trained on CEC2017 F9 (Shifted Composite, D=30)
      
    Operator Configuration:
      - Family: h3 (TSRI - Translation, Scale, Rotation Invariant)
      - Parameter sets: k = 10
      - Parameters per set: [w1, w2, w3, w4] = [r1_coeff, r2_sigma, r2_mu, prob]
      - Total dimensions: 40
      
    Training Specification:
      - Inner solver: pop=30, max_FE=3000, D=30
      - Outer budget: 500 operators per stage, 1000 total
      - Fitness aggregation: MEDIAN of 3 runs per operator
      
    Examples
    ~~~~~~~~
    >>> from mealpy import FloatVar
    >>> from AutoV import TrainedAutoV
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-100.,)*30, ub=(100.,)*30, name="x"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> # Use pre-trained operator from stage 2
    >>> model = TrainedAutoV(epoch=100, pop_size=30)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Best fitness: {g_best.target.fitness}")
    
    Loading from .mat file:
    ~~~~~~~~~~~~~~~~~~~~~~~~
    >>> from AutoV import TrainedAutoV
    >>> import scipy.io as sio
    >>>
    >>> # Load trained operator from MATLAB .mat file
    >>> mat_data = sio.loadmat('trained_AutoV_CEC2017_F9_D30_stage2_from_f1.mat')
    >>> operator_matrix = mat_data['best_operator_matrix']
    >>> 
    >>> model = TrainedAutoV(epoch=100, pop_size=30, operator_params=operator_matrix)
    >>> g_best = model.solve(problem_dict)
    """
    
    def __init__(self, epoch: int = 100, pop_size: int = 30, 
                 tournament_size: int = 2, operator_params: np.ndarray = None,
                 **kwargs: object) -> None:
        """
        Args:
            epoch: Number of iterations, default = 100
            pop_size: Population size, default = 30
            tournament_size: [2-20] Tournament size, default = 2
            operator_params: Trained operator parameter matrix (10×4)
                           If None, loads hardcoded stage 2 training result
        """
        # Use hardcoded stage 2 training if not provided
        if operator_params is None:
            operator_params = TRAINED_OPERATOR_STAGE2.copy()
        
        super().__init__(epoch=epoch, pop_size=pop_size, tournament_size=tournament_size,
                        operator_params=operator_params, **kwargs)
        
        # Store training metadata
        self.trained_config = {
            'source': 'hardcoded_stage2_transfer_learning',
            'stage1_problem': 'CEC2017_F1',
            'stage2_problem': 'CEC2017_F9',
            'dimension': 30,
            'operator_family': OPERATOR_FAMILY,
            'operator_k': OPERATOR_K,
            'inner_pop_size': 30,
            'inner_max_fe': 3000,
            'outer_budget_per_stage': 500,
            'total_outer_budget': 1000,
            'best_fitness_stage1': TRAINED_FITNESS_STAGE1_F1,
            'best_fitness_stage2': TRAINED_FITNESS_STAGE2_F9,
        }

    def get_trained_parameters(self) -> np.ndarray:
        """
        Get the trained operator parameter matrix
        
        Returns:
            Operator parameters as (k, 4) matrix
        """
        return self.operator_params.copy()

    def get_operator_details(self) -> dict:
        """
        Get detailed operator configuration
        
        Returns:
            Dictionary with operator details
        """
        return {
            'family': OPERATOR_FAMILY,
            'parameter_sets': OPERATOR_K,
            'params_per_set': OPERATOR_PARAMS_PER_SET,
            'parameter_names': ['w1 (r1_coeff)', 'w2 (r2_sigma)', 'w3 (r2_mu)', 'w4 (probability)'],
            'parameter_matrix': self.operator_params.copy(),
            'bounds': {
                'w1': [0.0, 1.0],
                'w2': [0.0, 1.0],
                'w3': [-1.0, 1.0],
                'w4': [1e-6, 1.0]
            }
        }

    def information(self) -> None:
        """Display trained AutoV configuration and performance information"""
        print("\n" + "="*80)
        print("TRAINED AUTOV - STAGE 2 TRANSFER LEARNING")
        print("="*80)
        
        print(f"\nTraining Configuration:")
        print(f"  Algorithm: AutoV (TSRI operator design)")
        print(f"  Training Approach: Two-Stage Transfer Learning")
        print(f"  Stage 1 Problem: CEC2017 F1 (Shifted Sphere, D=30)")
        print(f"  Stage 2 Problem: CEC2017 F9 (Shifted Composite, D=30)")
        print(f"  Training Criteria: 1000 candidate operators total")
        print(f"  File Source: trained_AutoV_CEC2017_F9_D30_stage2_from_f1.mat")
        
        print(f"\nOperator Configuration:")
        print(f"  Family: {OPERATOR_FAMILY} (TSRI - Translation, Scale, Rotation Invariant)")
        print(f"  Parameter Sets: k = {OPERATOR_K}")
        print(f"  Params per Set: {OPERATOR_PARAMS_PER_SET}")
        print(f"  Parameter Names: w1 (r1_coeff), w2 (r2_sigma), w3 (r2_mu), w4 (probability)")
        print(f"  Search Space: {OPERATOR_K * OPERATOR_PARAMS_PER_SET} dimensions")
        
        print(f"\nTSRI Operator Equation:")
        print(f"  o_i = r1 * (u_i - l_i) + r2 * x2_i + (1 - r2) * x1_i")
        print(f"  where r1 ~ N(0, w1^2), r2 ~ N(w3, w2^2)")
        
        print(f"\nTraining Budget:")
        print(f"  Inner solver (per evaluation): D=30, pop=30, maxFE=3000")
        print(f"  Outer optimizer: {500} operators per stage")
        print(f"  Total operators tuned: 1000 (500 × 2 stages)")
        print(f"  Fitness aggregation: MEDIAN of 3 runs per operator")
        print(f"  Number of runs per operator: 3")
        print(f"  Total function evaluations: ~9 million")
        
        print(f"\nTrained Parameters:")
        print(f"  Stage 1 fitness (F1): {TRAINED_FITNESS_STAGE1_F1:.6e}")
        print(f"  Stage 2 fitness (F9): {TRAINED_FITNESS_STAGE2_F9:.6e}")
        if TRAINED_FITNESS_STAGE1_F1 < 1e10:
            transfer_improvement = (1 - TRAINED_FITNESS_STAGE2_F9/TRAINED_FITNESS_STAGE1_F1) * 100
            print(f"  Transfer improvement: {transfer_improvement:.2f}%")
        
        print(f"\nCurrent Algorithm Configuration:")
        print(f"  Epochs: {self.epoch}")
        print(f"  Population Size: {self.pop_size}")
        print(f"  Tournament Size: {self.tournament_size}")
        
        print(f"\nOperator Parameter Matrix ({OPERATOR_K} sets × {OPERATOR_PARAMS_PER_SET} params):")
        print(f"  {'Set':<6}{'w1':<12}{'w2':<12}{'w3':<12}{'w4 (prob)':<12}")
        print(f"  {'-'*54}")
        for i, params in enumerate(self.operator_params):
            print(f"  {i+1:<6}{params[0]:<12.6f}{params[1]:<12.6f}{params[2]:<12.6f}{params[3]:<12.6f}")
        print(f"  {'-'*54}")
        print(f"  Sum of w4 (prob weights): {self.operator_params[:, 3].sum():.6f}")
        
        print("="*80 + "\n")


# Utility function to load trained operator from MATLAB .mat file
def load_trained_operator_from_mat(mat_filepath: str) -> np.ndarray:
    """
    Load trained AutoV operator from a MATLAB .mat file
    
    Args:
        mat_filepath: Path to the .mat file (e.g., 'trained_AutoV_CEC2017_F9_D30_stage2_from_f1.mat')
    
    Returns:
        Operator parameter matrix (k×4), or None if loading fails
    
    Example:
        >>> operator_matrix = load_trained_operator_from_mat('trained_AutoV_CEC2017_F9_D30_stage2_from_f1.mat')
        >>> model = TrainedAutoV(operator_params=operator_matrix)
    """
    try:
        import scipy.io as sio
        
        mat_data = sio.loadmat(mat_filepath)
        
        if 'best_operator_matrix' in mat_data:
            operator_matrix = mat_data['best_operator_matrix']
            print(f"Loaded operator from: {mat_filepath}")
            print(f"Operator shape: {operator_matrix.shape}")
            return operator_matrix
        else:
            print(f"Error: 'best_operator_matrix' not found in {mat_filepath}")
            print(f"Available keys: {list(mat_data.keys())}")
            return None
    except ImportError:
        print("Error: scipy.io not available. Install scipy to load .mat files:")
        print("  pip install scipy")
        return None
    except FileNotFoundError:
        print(f"Error: File not found: {mat_filepath}")
        return None
    except Exception as e:
        print(f"Error loading {mat_filepath}: {e}")
        return None


# Utility function to load training history and metadata
def load_training_info_from_mat(mat_filepath: str) -> dict:
    """
    Load training information from a MATLAB .mat file
    
    Args:
        mat_filepath: Path to the .mat file
    
    Returns:
        Dictionary with training metadata
    
    Example:
        >>> info = load_training_info_from_mat('trained_AutoV_CEC2017_F9_D30_stage2_from_f1.mat')
        >>> print(f"Best fitness: {info['best_fitness']:.6e}")
    """
    try:
        import scipy.io as sio
        
        mat_data = sio.loadmat(mat_filepath)
        
        info = {}
        
        # Basic info
        if 'best_fitness' in mat_data:
            info['best_fitness'] = float(mat_data['best_fitness'])
        if 'PROBLEM_NAME' in mat_data:
            info['problem_name'] = str(mat_data['PROBLEM_NAME'].flat[0])
        if 'DIMENSION' in mat_data:
            info['dimension'] = int(mat_data['DIMENSION'].flat[0])
        if 'operator_family' in mat_data:
            info['operator_family'] = str(mat_data['operator_family'].flat[0])
        
        # Training history (if available)
        if 'trainer_history' in mat_data:
            try:
                hist = mat_data['trainer_history']
                if 'best_fitnesses' in hist.dtype.names:
                    info['best_fitnesses'] = hist['best_fitnesses'].flat
                if 'num_evaluations' in hist.dtype.names:
                    info['num_evaluations'] = int(hist['num_evaluations'].flat[0])
            except:
                pass
        
        return info
    except ImportError:
        print("Error: scipy.io not available")
        return {}
    except Exception as e:
        print(f"Error loading training info from {mat_filepath}: {e}")
        return {}


if __name__ == "__main__":
    """
    Example usage of TrainedAutoV
    """
    print("AutoV - Automated Design of Variation Operators")
    print("=" * 80)
    print("\nExample 1: Using hardcoded trained operator (stage 2)")
    print("-" * 80)
    
    from mealpy import FloatVar
    
    # Define a simple test problem
    def sphere_function(solution):
        return np.sum(solution**2)
    
    problem_dict = {
        "bounds": FloatVar(n_vars=30, lb=(-100.,)*30, ub=(100.,)*30, name="x"),
        "obj_func": sphere_function,
        "minmax": "min",
    }
    
    # Create and run trained AutoV
    print("\nCreating TrainedAutoV model...")
    model = TrainedAutoV(epoch=10, pop_size=30, tournament_size=2)
    
    print("Displaying trained information...")
    model.information()
    
    print("Running optimization...")
    g_best = model.solve(problem_dict)
    print(f"\nBest solution found: {g_best.solution[:5]}... (first 5 dims)")
    print(f"Best fitness: {g_best.target.fitness:.6e}")
    
    print("\n" + "=" * 80)
    print("Example 2: Loading operator from .mat file")
    print("-" * 80)
    
    mat_file = 'trained_AutoV_CEC2017_F9_D30_stage2_from_f1.mat'
    operator = load_trained_operator_from_mat(mat_file)
    
    if operator is not None:
        print(f"Successfully loaded operator from {mat_file}")
        print(f"Creating model with loaded operator...")
        
        model2 = TrainedAutoV(epoch=10, pop_size=30, operator_params=operator)
        g_best2 = model2.solve(problem_dict)
        print(f"Best fitness with loaded operator: {g_best2.target.fitness:.6e}")
    else:
        print(f"Could not load operator from {mat_file}")
        print("Make sure to run the MATLAB training first:")
        print("  train_AutoV_cec2017_stage2_f9_D30_from_f1.m")
    
    print("\n" + "=" * 80)
