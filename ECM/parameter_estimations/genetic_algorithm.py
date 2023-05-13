import time

import numpy as np


def timer(solver_func):
    """
    Timer function is intended to be a decorator function that takes in any solver function and calculates the solver
    solution time. It then displays the solution time.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        sol = solver_func(*args, **kwargs)
        print(f"Solver execution time: {time.time() - start_time}s")
        return sol
    return wrapper


class GA:
    def __init__(self, n_chromosomes, bounds, obj_func, n_pool, n_elite, n_generations, mutating_factor=0.8):
        """
        Initializes the population by drawing numbers from a uniform distribution, within the bounds.
        :params n_chromosomes: (float) the number of parameter sets per generation
        :params bounds: (numpy array) contains the bounds for each of the parameters. Each row in this numpy array
                corresponds to the bounds for the parameter.
        :params obj_func: (func) a function object that defines the objective function. Note: the objective function has
                         to handle numpy array with 1 dimension. Furthermore, the GA class tries to minimize the
                         objective function.
        :params n_pool: (int) number of pooling population
        :params n_elite: (int) number of elite population.
        :param n_generations: (int) number of generations.
        :params mutating_factor: (float) the ratio of population (excluding the elite chromosomes) to mutate.
        """

        if isinstance(n_chromosomes, int):
            self.n_chromosomes = n_chromosomes
        else:
            raise TypeError("n_chromsomes need to be an integer")

        # Ensure the bounds are a 2D numpy array, and has two columns.
        if isinstance(bounds, np.ndarray):
            if bounds.ndim == 2:
                if bounds.shape[1] == 2:
                    self.bounds = bounds
                else:
                    raise ValueError("Bounds need to have 2 columns. The first and second columns specifies the "
                                     "lower_bound and upper bounds, respectively.")
            else:
                raise ValueError("Numpy array needs to be 2D Numpy array.")
        else:
            raise TypeError("Bounds need to be Numpy array.")

        self.n_genes = self.bounds.shape[0]

        if callable(obj_func):
            self.obj_func = obj_func
        else:
            raise TypeError("obj_func needs to be a func.")

        if isinstance(n_pool, int):
            if n_pool <= self.n_chromosomes:
                self.n_pool = n_pool
            else:
                raise ValueError(f"n_pool: {n_pool} exceeds the n_chromosomes: {self.n_chromosomes}")
        else:
            raise TypeError("n_pool needs to be a integer.")

        if isinstance(n_elite, int):
            if n_elite <= self.n_pool:
                self.n_elite = n_elite
            else:
                raise ValueError(f"n_elite: {n_elite} exceeds the n_pool: {n_pool}.")
        else:
            raise TypeError("n_elite needs to be a integer.")

        if isinstance(n_generations, int):
            self.n_generations = n_generations
        else:
            raise TypeError("n_generations needs to be a integer.")

        if isinstance(mutating_factor, float):
            self.mutating_factor = mutating_factor
        else:
            TypeError("mutating factor needs to be a float.")

    def initialize_population(self):
        """
        Initializes the population by drawing numbers from a uniform distribution, within the bounds.
        :returns: (Numpy array) a numpy array of populations. Each row contains the chromosomes (i.e. parameter set)
                and each column contains the genes (i.e., parameter values)
        """
        pop = np.zeros((self.n_chromosomes, self.n_genes)) # each row will be a parameter set
        for chromo_index in range(self.n_chromosomes):
            for gene_index in range(self.n_genes):
                a = np.random.uniform(self.bounds[gene_index][0], self.bounds[gene_index][1])
                pop[chromo_index, gene_index] = a
        return pop

    def calc_fitness(self, population):
        """
        Calculates the fitness of the parameters. In each iteration, the chromosomes are passed into the objective
        function and the fitness value is calculated.
        :param population: (Numpy array) population where the row indicates the chromosome.
        :return fitness_array: (Numpy array): a array containing the fitness from each chromosome.
        """
        fitness_array = np.array([])
        for chromosome in population:
            fitness = self.obj_func(chromosome)
            fitness_array = np.append(fitness_array, fitness)
        return fitness_array

    def sorting(self, population, fitness_array):
        """
        Sorts the population array according to the values in the fitness array.
        """
        sorted_fitness_array_index = fitness_array.argsort()
        sorted_fitness_array = fitness_array[sorted_fitness_array_index]
        sorted_population = population[sorted_fitness_array_index]
        return sorted_fitness_array_index, sorted_fitness_array, sorted_population

    def mating_population(self, population, fitness_array):
        sorted_fitness_array_index, sorted_fitness_array, sorted_population = self.sorting(population=population,
                                                                                           fitness_array=fitness_array)
        return sorted_population[:self.n_pool]

    def create_new_population(self, mating_population):
        # create a numpy array for elite population
        elite_population = mating_population[:self.n_elite]
        # initiate a numpy array for children population
        children_population = np.zeros([self.n_chromosomes-self.n_elite, self.n_genes])
        for new_chromosome in range(self.n_chromosomes - self.n_elite):
            random_index = np.random.choice(self.n_pool, 2, replace=False)
            alpha = np.random.uniform(0, 1)
            for gene in range(self.n_genes):
                # cross-over operation
                child = alpha * mating_population[random_index[0], gene] + (1 - alpha) * \
                        mating_population[random_index[1], gene]
                children_population[new_chromosome, gene] = child
        # Append elite and children populations
        return np.concatenate((elite_population, children_population), axis=0)

    def mutate(self, population):
        # calc. the population to mutate.
        n_population_to_mutate = int((self.n_chromosomes - self.n_elite) * (1 - self.mutating_factor))
        # create an array of population to mutate.
        random_index = np.random.choice(self.n_chromosomes, n_population_to_mutate, replace=False)
        for chromsome_i in random_index:
            for gene_i in range(self.n_genes):
                population[chromsome_i, gene_i] = population[chromsome_i, gene_i] \
                                                  + np.std(population[:,gene_i]) * np.random.uniform(0, 1)
        return population

    @timer
    def solve(self):
        """
        Performs the genetic algorithm.
        :returns: (Numpy array) an array of parameters after optimizing through genetic algorithm
        """
        # First, initialize the population
        population = self.initialize_population()
        for generation_i in range(self.n_generations):
            # Then, calculate the fitness
            fitness_array = self.calc_fitness(population=population)
            # Create the mating population
            population = self.mating_population(population=population, fitness_array=fitness_array)
            # Create new population based on elite populations, crossover, and mutations
            population = self.create_new_population(mating_population=population)
            population = self.mutate(population=population) # This population is used for the next iteration
        return population[0]



def objective_func(x):
    return x.sum()
ga_example = GA(n_chromosomes=10, bounds=np.array([[0,1],[0,1],[0,100]]), obj_func=objective_func, n_pool=7, n_elite=3, n_generations=3)
# initial_population = ga_example.initialize_population()
# # print(initial_population)
# fitness_array = ga_example.calc_fitness(initial_population)
# # print(fitness_array)
# mating_pool = ga_example.mating_population(population=initial_population, fitness_array=fitness_array)
# # print(mating_pool)
# new_population = ga_example.create_new_population(mating_population=mating_pool)
# # print(new_population)
# mutated_pop = ga_example.mutate(population=new_population)
result = ga_example.solve()
print(result)