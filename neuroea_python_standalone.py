"""
Pure Python version of TrainedNeuroEA that loads parameters from JSON
No external dependencies required (except standard library)

Provides the same interface as the scipy version but uses JSON for parameter storage.
"""

import json
import numpy as np
from pathlib import Path


class TrainedNeuroEA:
    """
    Pure Python implementation of trained NeuroEA optimizer.
    
    Loads trained parameters from JSON configuration file.
    Compatible with Mealpy framework when wrapped properly.
    
    Parameters:
        - epoch (int): Number of iterations
        - pop_size (int): Population size
        - c1 (float): Crossover rate [0-1]
        - m1 (float): Mutation rate [0-1]
        - tournament_size (int): Tournament selection size
        - params_file (str): Path to trained_neuroea_params.json
    """
    
    def __init__(self, epoch=100, pop_size=30, c1=None, m1=None, 
                 tournament_size=10, params_file='trained_neuroea_params.json'):
        """Initialize NeuroEA optimizer with trained parameters"""
        
        self.epoch = max(1, min(epoch, 100000))
        self.pop_size = max(5, min(pop_size, 10000))
        self.tournament_size = max(2, min(tournament_size, 100))
        
        # Load trained parameters if available
        self.params_file = params_file
        self.trained_params = None
        self.graph = None
        self.metadata = {}
        
        self.load_trained_parameters()
        
        # Set rates
        if c1 is not None:
            self.c1 = max(0.0, min(c1, 1.0))
        else:
            self.c1 = 0.5  # Default crossover rate
            
        if m1 is not None:
            self.m1 = max(0.0, min(m1, 1.0))
        else:
            self.m1 = 0.1  # Default mutation rate

    def load_trained_parameters(self):
        """Load trained parameters from JSON file"""
        
        try:
            json_file = Path(self.params_file)
            
            if not json_file.exists():
                print(f"Warning: {self.params_file} not found. Using default parameters.")
                return
            
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract parameters
            if 'trained_parameters' in data:
                self.trained_params = np.array(data['trained_parameters'].get('stage2', []))
            
            if 'connectivity' in data:
                self.graph = np.array(data['connectivity'].get('graph', []))
            
            if 'metadata' in data:
                self.metadata = data['metadata']
            
            print(f"✓ Loaded trained parameters from {self.params_file}")
            if self.trained_params is not None:
                print(f"  Parameters: {len(self.trained_params)} dimensions")
            if self.metadata:
                print(f"  Algorithm: {self.metadata.get('algorithm', 'NeuroEA')}")
                print(f"  Stage 2 Problem: {self.metadata.get('stage2_problem', 'CEC2017_F9')}")
                print(f"  Dimension: {self.metadata.get('dimension', '30')}")
        
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {self.params_file}")
        except Exception as e:
            print(f"Warning: Error loading parameters: {e}")

    def set_seed(self, seed):
        """Set random seed for reproducibility"""
        np.random.seed(seed)

    def tournament_select(self, population, fitness_values, tournament_size=None):
        """
        Tournament selection: randomly select tournament_size individuals
        and return the index of the best one.
        """
        if tournament_size is None:
            tournament_size = self.tournament_size
        
        tournament_size = min(tournament_size, len(population))
        pop_indices = np.arange(len(population))
        candidates = np.random.choice(pop_indices, size=tournament_size, replace=False)
        
        # Return index of best (minimum) fitness
        best_idx = candidates[0]
        for idx in candidates[1:]:
            if fitness_values[idx] < fitness_values[best_idx]:
                best_idx = idx
        
        return best_idx

    def crossover(self, parent1, parent2, crossover_rate=None):
        """
        Crossover operator: blend two solutions
        
        Args:
            parent1, parent2: Parent solution vectors
            crossover_rate: Probability of crossing over each dimension
        
        Returns:
            child: Offspring solution
        """
        if crossover_rate is None:
            crossover_rate = self.c1
        
        child = parent1.copy()
        for i in range(len(child)):
            if np.random.random() < crossover_rate:
                # Arithmetic crossover
                alpha = np.random.random()
                child[i] = alpha * parent1[i] + (1 - alpha) * parent2[i]
        
        return child

    def mutate(self, solution, mutation_rate=None, bounds=None):
        """
        Mutation operator: apply random perturbations
        
        Args:
            solution: Solution vector
            mutation_rate: Probability of mutating each dimension
            bounds: (lb, ub) tuples for bounding
        
        Returns:
            mutated: Mutated solution
        """
        if mutation_rate is None:
            mutation_rate = self.m1
        
        mutated = solution.copy()
        
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                # Gaussian mutation
                if bounds is not None:
                    lb, ub = bounds[i]
                    sigma = (ub - lb) * 0.1
                    mutated[i] = mutated[i] + np.random.normal(0, sigma)
                    # Clip to bounds
                    mutated[i] = np.clip(mutated[i], lb, ub)
                else:
                    mutated[i] = mutated[i] + np.random.normal(0, 0.1)
        
        return mutated

    def evolve_generation(self, population, fitness_values, bounds=None):
        """
        Single generation of the NeuroEA algorithm
        
        Flow:
          1. Tournament selection
          2. Crossover
          3. Mutation
          4. Survivor selection
        
        Args:
            population: List of solution vectors
            fitness_values: Fitness value for each solution
            bounds: Search space bounds
        
        Returns:
            new_population: Updated population
            new_fitness: Updated fitness values
        """
        new_population = []
        new_fitness = []
        
        # Generate offspring via tournament selection, crossover, mutation
        for _ in range(len(population)):
            # Tournament selection (2 parents)
            parent1_idx = self.tournament_select(population, fitness_values)
            parent2_idx = self.tournament_select(population, fitness_values)
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover
            child = self.crossover(parent1, parent2)
            
            # Mutation
            if bounds is not None:
                child = self.mutate(child, bounds=bounds)
            else:
                child = self.mutate(child)
            
            new_population.append(child)
        
        return new_population

    def get_info(self):
        """Return optimizer configuration summary"""
        info = {
            'name': 'TrainedNeuroEA',
            'epoch': self.epoch,
            'pop_size': self.pop_size,
            'c1': self.c1,
            'm1': self.m1,
            'tournament_size': self.tournament_size,
            'parameters_loaded': self.trained_params is not None,
            'num_trained_parameters': len(self.trained_params) if self.trained_params is not None else 0,
        }
        return info

    def __str__(self):
        """String representation"""
        info = self.get_info()
        return f"""
TrainedNeuroEA Configuration:
  Epochs: {info['epoch']}
  Population size: {info['pop_size']}
  Crossover rate: {info['c1']:.4f}
  Mutation rate: {info['m1']:.4f}
  Tournament size: {info['tournament_size']}
  Trained parameters loaded: {info['parameters_loaded']}
  Number of trained parameters: {info['num_trained_parameters']}
"""


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*80)
    print("TRAINED NEUROEA - PURE PYTHON (NO DEPENDENCIES)")
    print("="*80)
    
    # Create optimizer
    optimizer = TrainedNeuroEA(epoch=100, pop_size=30)
    print(optimizer)
    
    # Show configuration
    info = optimizer.get_info()
    print("\nOptimizer Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Example: Simple optimization on sphere function
    print("\n" + "="*80)
    print("EXAMPLE: Sphere Function Optimization")
    print("="*80)
    
    def sphere(x):
        """Simple sphere benchmark"""
        return np.sum(x**2)
    
    # Initialize random population
    D = 5  # Dimension
    bounds = [(-5.0, 5.0) for _ in range(D)]
    population = [np.random.uniform(-5.0, 5.0, D) for _ in range(optimizer.pop_size)]
    fitness = np.array([sphere(x) for x in population])
    
    print(f"\nInitial population:")
    print(f"  Dimension: {D}")
    print(f"  Population size: {optimizer.pop_size}")
    print(f"  Best fitness: {fitness.min():.6f}")
    print(f"  Mean fitness: {fitness.mean():.6f}")
    
    # Run a few generations
    print(f"\nRunning 10 generations...")
    for gen in range(10):
        new_pop = optimizer.evolve_generation(population, fitness, bounds)
        new_fitness = np.array([sphere(x) for x in new_pop])
        
        # Simple (μ+λ) strategy: keep best individuals
        combined_pop = population + new_pop
        combined_fitness = np.concatenate([fitness, new_fitness])
        
        best_indices = np.argsort(combined_fitness)[:optimizer.pop_size]
        population = [combined_pop[i] for i in best_indices]
        fitness = combined_fitness[best_indices]
        
        if (gen + 1) % 3 == 0:
            print(f"  Gen {gen+1:2d}: best={fitness.min():.6f}, mean={fitness.mean():.6f}")
    
    print(f"\nFinal best fitness: {fitness.min():.6f}")
    print(f"Final best solution: {population[np.argmin(fitness)]}")
    
    print("\n" + "="*80)
    print("Ready to integrate with Mealpy framework!")
    print("="*80)
