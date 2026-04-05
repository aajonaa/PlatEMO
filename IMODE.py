#!/usr/bin/env python
# Created by "IMODE Team" at 09:50, 06/04/2026 ----------%
#       Paper: Improved Multi-Operator Differential Evolution
#       Algorithm: IMODE with adaptive CR, F, and operator selection
#       Framework: mealpy (Evolutionary Algorithms Library)
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent
from scipy import stats

# ========================================================================
# IMODE - Improved Multi-Operator Differential Evolution
# ========================================================================


class OriginalIMODE(Optimizer):
    """
    The original IMODE algorithm: Improved Multi-Operator Differential Evolution
    
    IMODE features:
      - Three mutation operators with adaptive selection
      - Adaptive crossover rate (CR) and scaling factor (F)
      - Archive-based diversity maintenance
      - Adaptive population size reduction
      - Two crossover modes (uniform and segmented)
    
    Hyper-parameters to fine-tune:
        + minN (int): [2, 20], Minimum population size, default = 4
        + aRate (float): [1.0, 5.0], Archive size ratio to population, default = 2.6
        + cr_mean (float): [0.0, 1.0], Initial mean CR, default = 0.2
        + f_mean (float): [0.0, 1.0], Initial mean F, default = 0.2

    Mutation Operators:
        1. DE/current-to-pbest/2: x + F*(p_best - x) + F*(r1 - r2)
        2. DE/current-to-pbest-archive/2: x + F*(p_best - x) + F*(r1 - r3)
        3. DE/rand-pbest/2: F*(r1 + p_best - r3)
    
    Crossover Modes:
        - Uniform crossover (40% probability)
        - Segmented crossover (60% probability)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, IMODE
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
    >>> model = OriginalIMODE(epoch=100, pop_size=50, minN=4, aRate=2.6)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Best fitness: {g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] K. M. Sallam, S. M. Elsayed, R. K. Chakrabortty, and M. J. Ryan. Improved
        multi-operator differential evolution algorithm for solving unconstrained
        problems. Proceedings of the IEEE Congress on Evolutionary Computation, 2020.
    """

    def __init__(self, epoch: int = 100, pop_size: int = 50, minN: int = 4, 
                 aRate: float = 2.6, cr_mean: float = 0.2, f_mean: float = 0.2,
                 **kwargs: object) -> None:
        """
        Args:
            epoch: Maximum number of iterations, default = 100
            pop_size: Population size, default = 50
            minN: [2-20] Minimum population size, default = 4
            aRate: [1.0-5.0] Archive size ratio to population, default = 2.6
            cr_mean: [0.0-1.0] Initial mean CR, default = 0.2
            f_mean: [0.0-1.0] Initial mean F, default = 0.2
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.minN = self.validator.check_int("minN", minN, [2, 20])
        self.aRate = self.validator.check_float("aRate", aRate, [1.0, 5.0])
        self.cr_mean = self.validator.check_float("cr_mean", cr_mean, (0, 1.0))
        self.f_mean = self.validator.check_float("f_mean", f_mean, (0, 1.0))
        
        self.set_parameters(["epoch", "pop_size", "minN", "aRate", "cr_mean", "f_mean"])
        self.sort_flag = False
        self.is_parallelizable = False

    def initialize_variables(self):
        """Initialize algorithm-specific variables"""
        # Memory for CR and F adaptation
        self.memory_size = 20 * self.problem.n_dims
        self.MCR = np.ones(self.memory_size) * self.cr_mean
        self.MF = np.ones(self.memory_size) * self.f_mean
        self.k = 0  # Current memory index
        
        # Operator selection probabilities
        self.MOP = np.ones(3) / 3  # Equal probability for all 3 operators
        
        # Archive for diversity
        self.archive = []

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

    def update_population_size(self, epoch: int) -> int:
        """
        Reduce population size linearly over generations based on epoch
        N(t) = max(minN, ceil(pop_size - (pop_size - minN) * epoch / epoch_max))
        
        Args:
            epoch: Current epoch number (0 to epoch-1)
        
        Returns:
            Updated population size
        """
        # Linear reduction from pop_size to minN over the training period
        progress = epoch / max(1, self.epoch - 1) if self.epoch > 1 else 1.0
        N = max(self.minN,
                int(np.ceil(self.pop_size - (self.pop_size - self.minN) * progress)))
        return N

    def select_parent_set(self, population: list, fraction: float) -> np.ndarray:
        """
        Select parent set from population
        p_best set has size = max(1, ceil(fraction * N))
        """
        N = len(population)
        pbest_size = max(1, int(np.ceil(fraction * N)))
        
        # Get fitness values
        fitness = np.array([agent.target.fitness for agent in population])
        
        # Select best individuals
        best_indices = np.argsort(fitness)[:pbest_size]
        return np.array([population[idx].solution for idx in best_indices])

    def generate_cr_values(self, N: int) -> np.ndarray:
        """
        Generate CR values from normal distribution
        CR ~ N(MCR[i], 0.1)
        """
        # Sample from memory with indices
        indices = self.generator.choice(self.memory_size, N)
        cr_values = np.clip(
            self.generator.normal(self.MCR[indices], np.sqrt(0.1)),
            0, 1
        )
        return cr_values

    def generate_f_values(self, N: int) -> np.ndarray:
        """
        Generate F values from Cauchy distribution
        F ~ Cauchy(MF[i], 0.1), clipped to (0, 1]
        """
        indices = self.generator.choice(self.memory_size, N)
        f_values = []
        
        for i in range(N):
            while True:
                # Cauchy distribution with location=MF[i], scale=sqrt(0.1)
                f = stats.cauchy.rvs(loc=self.MF[indices[i]], 
                                      scale=np.sqrt(0.1))
                f = np.clip(f, 0.001, 1.0)  # Clip to valid range
                if f > 0:
                    break
            f_values.append(f)
        
        return np.array(f_values)

    def select_operator(self, N: int) -> list:
        """
        Select operator for each individual based on probabilities
        Returns list of indices for each operator
        """
        # Cumulative sum for roulette wheel selection
        cumsum_prob = np.cumsum(self.MOP)
        
        # Select operator for each individual
        operators = []
        for _ in range(N):
            rand_val = self.generator.random()
            op_idx = np.searchsorted(cumsum_prob, rand_val, side='right')
            op_idx = min(op_idx, len(self.MOP) - 1)
            operators.append(op_idx)
        
        # Group indices by operator
        op_groups = [[] for _ in range(3)]
        for idx, op in enumerate(operators):
            op_groups[op].append(idx)
        
        return op_groups

    def crossover(self, pop_dec: np.ndarray, off_dec: np.ndarray, 
                  cr_values: np.ndarray) -> np.ndarray:
        """
        Apply crossover operation
        With 40% uniform crossover, 60% segmented crossover
        """
        N = len(pop_dec)
        
        if self.generator.random() < 0.4:
            # Uniform crossover
            site = self.generator.random((N, self.problem.n_dims)) > cr_values[:, None]
            result = np.where(site, pop_dec, off_dec)
        else:
            # Segmented crossover
            result = off_dec.copy()
            for i in range(N):
                # Random first position
                p1 = self.generator.integers(0, self.problem.n_dims)
                # Find length
                p2 = np.searchsorted(
                    np.concatenate([[0], np.where(self.generator.random(self.problem.n_dims) > cr_values[i])[0]]),
                    p1 + 1
                ) - p1
                
                if p2 < self.problem.n_dims:
                    # Copy from original to result
                    indices = np.arange(p2)
                    result[i, (p1 + indices) % self.problem.n_dims] = pop_dec[i, (p1 + indices) % self.problem.n_dims]
        
        return result

    def de_current_pbest_2(self, x: np.ndarray, pbest: np.ndarray, 
                           r1: np.ndarray, r2: np.ndarray, 
                           f: float) -> np.ndarray:
        """
        DE/current-to-pbest/2 mutation
        v = x + F*(pbest - x) + F*(r1 - r2)
        
        Args:
            x: Current individual
            pbest: Best individual from pbest set
            r1, r2: Random individuals
            f: Scaling factor
        """
        return x + f * (pbest - x) + f * (r1 - r2)

    def de_current_pbest_archive_2(self, x: np.ndarray, pbest: np.ndarray, 
                                    r1: np.ndarray, r3: np.ndarray, 
                                    f: float) -> np.ndarray:
        """
        DE/current-to-pbest-archive/2 mutation
        v = x + F*(pbest - x) + F*(r1 - r3)
        
        Args:
            x: Current individual
            pbest: Best individual from pbest set
            r1, r3: Random individuals (from population or archive)
            f: Scaling factor
        """
        return x + f * (pbest - x) + f * (r1 - r3)

    def de_rand_pbest_2(self, pbest: np.ndarray, r1: np.ndarray, 
                       r3: np.ndarray, f: float) -> np.ndarray:
        """
        DE/rand-pbest/2 mutation
        v = F*(r1 + pbest - r3)
        
        Args:
            pbest: Best individual from pbest set
            r1, r3: Random individuals
            f: Scaling factor
        """
        return f * (r1 + pbest - r3)

    def update_memory(self, success_indices: np.ndarray, 
                     cr_values: np.ndarray, f_values: np.ndarray,
                     success_rates: np.ndarray) -> None:
        """
        Update CR and F memory based on successful solutions
        
        Uses safe division to avoid numerical warnings when values are near zero.
        """
        if len(success_indices) > 0:
            # Weighted average
            weights = success_rates[success_indices]
            weights = weights / np.sum(weights)
            
            # Update MCR with safe division
            cr_success = cr_values[success_indices]
            numerator_cr = np.sum(weights * cr_success**2)
            denominator_cr = np.sum(weights * cr_success)
            
            if denominator_cr > 1e-10:  # Safe threshold to avoid division by zero
                self.MCR[self.k] = numerator_cr / denominator_cr
            else:
                self.MCR[self.k] = 0.5  # Reset to default if near zero
            
            # Update MF with safe division
            f_success = f_values[success_indices]
            numerator_f = np.sum(weights * f_success**2)
            denominator_f = np.sum(weights * f_success)
            
            if denominator_f > 1e-10:  # Safe threshold to avoid division by zero
                self.MF[self.k] = numerator_f / denominator_f
            else:
                self.MF[self.k] = 0.5  # Reset to default if near zero
            
            # Clip to valid range
            self.MCR[self.k] = np.clip(self.MCR[self.k], 0, 1)
            self.MF[self.k] = np.clip(self.MF[self.k], 0.001, 1)
        else:
            # No success, reset to default
            self.MCR[self.k] = 0.5
            self.MF[self.k] = 0.5
        
        # Move to next memory position
        self.k = (self.k + 1) % self.memory_size

    def update_operator_prob(self, success_indices: np.ndarray,
                            op_groups: list, success_rates: np.ndarray) -> None:
        """
        Update operator selection probabilities based on success
        """
        if len(success_indices) == 0:
            # No successful solutions, reset to equal probability
            self.MOP = np.ones(3) / 3
        else:
            # Calculate success rate for each operator
            op_success_rates = []
            for op_idx in range(3):
                op_indices = np.array(op_groups[op_idx])
                if len(op_indices) > 0:
                    success_in_op = np.isin(op_indices, success_indices)
                    mean_rate = np.mean(success_rates[op_indices[success_in_op]]) if np.any(success_in_op) else 0
                else:
                    mean_rate = 0
                op_success_rates.append(mean_rate)
            
            # Update probabilities
            op_success_rates = np.array(op_success_rates)
            if np.sum(op_success_rates) > 0:
                self.MOP = op_success_rates / np.sum(op_success_rates)
            else:
                self.MOP = np.ones(3) / 3
            
            # Clip to [0.1, 0.9]
            self.MOP = np.clip(self.MOP, 0.1, 0.9)

    def evolve(self, epoch: int) -> None:
        """
        Main evolution loop of IMODE (one iteration)
        
        Flow:
            1. Reduce population size (if needed)
            2. Maintain archive
            3. Generate CR and F values
            4. Select operators
            5. Generate offspring via mutation and crossover
            6. Select and replace individuals
            7. Update memory and operator probabilities
        
        Args:
            epoch: Current generation number
        """
        # Update population size based on progress
        N = self.update_population_size(epoch)
        
        if N < len(self.pop):
            # Reduce population size - keep best individuals
            fitness = np.array([agent.target.fitness for agent in self.pop])
            best_indices = np.argsort(fitness)[:N]
            self.pop = [self.pop[i] for i in best_indices]
        
        # Maintain archive size
        if len(self.archive) > 0:
            max_archive_size = max(1, int(np.ceil(self.aRate * len(self.pop))))
            if len(self.archive) > max_archive_size:
                archive_indices = self.generator.choice(len(self.archive), max_archive_size, replace=False)
                self.archive = [self.archive[i] for i in archive_indices]
        
        # Get population decisions
        pop_dec = np.array([agent.solution for agent in self.pop])
        
        # Select pbest set (top 25%)
        pbest_dec = self.select_parent_set(self.pop, 0.25)
        pbest_indices = self.generator.choice(len(pbest_dec), N)
        
        # Generate CR, F values and select operators
        cr_values = self.generate_cr_values(N)
        f_values = self.generate_f_values(N)
        op_groups = self.select_operator(N)
        
        # Select random individuals
        r1_indices = self.generator.choice(len(self.pop), N)
        r2_indices = self.generator.choice(len(self.pop), N)
        r3_indices = self.generator.choice(len(self.pop), N)
        
        # For archive access
        combined_pop = self.pop + self.archive
        r2_archive_indices = self.generator.choice(len(combined_pop), N)
        
        # Generate offspring
        offspring_dec = pop_dec.copy()
        
        # Operator 1: DE/current-to-pbest/2
        for idx in op_groups[0]:
            offspring_dec[idx] = self.de_current_pbest_2(
                pop_dec[idx],
                pbest_dec[pbest_indices[idx]],
                self.pop[r1_indices[idx]].solution,
                combined_pop[r2_archive_indices[idx]].solution,
                f_values[idx]
            )
        
        # Operator 2: DE/current-to-pbest-archive/2
        for idx in op_groups[1]:
            offspring_dec[idx] = self.de_current_pbest_archive_2(
                pop_dec[idx],
                pbest_dec[pbest_indices[idx]],
                self.pop[r1_indices[idx]].solution,
                self.pop[r3_indices[idx]].solution,
                f_values[idx]
            )
        
        # Operator 3: DE/rand-pbest/2
        for idx in op_groups[2]:
            offspring_dec[idx] = self.de_rand_pbest_2(
                pbest_dec[pbest_indices[idx]],
                self.pop[r1_indices[idx]].solution,
                self.pop[r3_indices[idx]].solution,
                f_values[idx]
            )
        
        # Apply crossover
        offspring_dec = self.crossover(pop_dec, offspring_dec, cr_values)
        
        # Ensure within bounds
        offspring_dec = np.array([self.amend_solution(sol) for sol in offspring_dec])
        
        # Evaluate offspring
        offspring = [self.generate_agent(sol) for sol in offspring_dec]
        
        # Compare and update
        old_fitness = np.array([agent.target.fitness for agent in self.pop])
        new_fitness = np.array([agent.target.fitness for agent in offspring])
        
        # Success criterion
        success = new_fitness < old_fitness
        success_indices = np.where(success)[0]
        
        # Calculate success rates
        delta = old_fitness - new_fitness
        success_rates = np.clip(delta / (np.abs(old_fitness) + 1e-10), 0, 1)
        
        # Update population and archive
        if np.any(success):
            # Add successful old solutions to archive
            self.archive.extend([self.pop[i] for i in success_indices])
            
            # Replace with successful offspring
            for idx in success_indices:
                self.pop[idx] = offspring[idx]
        
        # Update memory
        self.update_memory(success_indices, cr_values, f_values, success_rates)
        
        # Update operator probabilities
        self.update_operator_prob(success_indices, op_groups, success_rates)


class TrainedIMODE(OriginalIMODE):
    """
    IMODE algorithm with potentially trained hyperparameters
    
    Currently uses default parameters. Can be extended to load trained
    hyperparameters in the future.
    
    Training Configuration (hypothetical):
      - Problem: CEC2017 benchmark suite
      - Dimension: D=30
      - Population size: 50
      - Generations: 100
      - Fitness metric: MEDIAN of 3 runs
    
    Examples
    ~~~~~~~~
    >>> from mealpy import FloatVar
    >>> from IMODE import TrainedIMODE
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-100.,)*30, ub=(100.,)*30, name="x"),
    >>>     "obj_func": lambda x: np.sum(x**2),
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = TrainedIMODE(epoch=100, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    """
    
    def __init__(self, epoch: int = 100, pop_size: int = 50, minN: int = 4,
                 aRate: float = 2.6, cr_mean: float = 0.2, f_mean: float = 0.2,
                 **kwargs: object) -> None:
        """
        Args:
            epoch: Number of iterations, default = 100
            pop_size: Population size, default = 50
            minN: Minimum population size, default = 4
            aRate: Archive size ratio, default = 2.6
            cr_mean: Initial mean CR, default = 0.2
            f_mean: Initial mean F, default = 0.2
        """
        super().__init__(epoch=epoch, pop_size=pop_size, minN=minN,
                        aRate=aRate, cr_mean=cr_mean, f_mean=f_mean, **kwargs)
        
        # Training metadata (for future use)
        self.trained_config = {
            'algorithm': 'IMODE',
            'framework': 'mealpy',
            'status': 'default_parameters'
        }

    def information(self) -> None:
        """Display IMODE configuration information"""
        print("\n" + "="*80)
        print("IMODE - Improved Multi-Operator Differential Evolution")
        print("="*80)
        
        print(f"\nAlgorithm Configuration:")
        print(f"  Epochs: {self.epoch}")
        print(f"  Population Size: {self.pop_size}")
        print(f"  Minimum Population Size: {self.minN}")
        print(f"  Archive Size Ratio: {self.aRate}")
        print(f"  Initial Mean CR: {self.cr_mean}")
        print(f"  Initial Mean F: {self.f_mean}")
        
        print(f"\nMutation Operators (3 total):")
        print(f"  1. DE/current-to-pbest/2")
        print(f"     v = x + F*(p_best - x) + F*(r1 - r2)")
        print(f"  2. DE/current-to-pbest-archive/2")
        print(f"     v = x + F*(p_best - x) + F*(r1 - r3)")
        print(f"  3. DE/rand-pbest/2")
        print(f"     v = F*(r1 + p_best - r3)")
        
        print(f"\nCrossover Modes:")
        print(f"  1. Uniform crossover (40% probability)")
        print(f"  2. Segmented crossover (60% probability)")
        
        print(f"\nAdaptation Mechanisms:")
        print(f"  - CR adaptation: Normal distribution N(MCR[i], 0.1)")
        print(f"  - F adaptation: Cauchy distribution C(MF[i], 0.1)")
        print(f"  - Operator adaptation: Success-based probability adjustment")
        print(f"  - Population size: Linear reduction from N to minN")
        
        print(f"\nMemory Configuration:")
        print(f"  Memory size: {20 * 30} (20 × D, where D=dimension)")
        print(f"  Archive size: ceil({self.aRate} × N)")
        
        print("="*80 + "\n")


if __name__ == "__main__":
    """
    Example usage of IMODE
    """
    print("IMODE - Improved Multi-Operator Differential Evolution")
    print("=" * 80)
    
    from mealpy import FloatVar
    
    # Define a simple test problem
    def sphere_function(solution):
        return np.sum(solution**2)
    
    problem_dict = {
        "bounds": FloatVar(n_vars=30, lb=(-100.,)*30, ub=(100.,)*30, name="x"),
        "obj_func": sphere_function,
        "minmax": "min",
    }
    
    # Create and run IMODE
    print("\nCreating OriginalIMODE model...")
    model = OriginalIMODE(epoch=50, pop_size=50, minN=4, aRate=2.6)
    
    print("Running optimization...")
    g_best = model.solve(problem_dict)
    print(f"\nBest solution found: {g_best.solution[:5]}... (first 5 dims)")
    print(f"Best fitness: {g_best.target.fitness:.6e}")
    
    # Display information
    print("\n" + "="*80)
    print("TrainedIMODE with information display:")
    model2 = TrainedIMODE(epoch=50, pop_size=50)
    model2.information()
    
    print("Running TrainedIMODE...")
    g_best2 = model2.solve(problem_dict)
    print(f"Best fitness: {g_best2.target.fitness:.6e}")
    
    print("\n" + "="*80)
