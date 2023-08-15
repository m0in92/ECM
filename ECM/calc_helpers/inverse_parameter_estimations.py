import time
import warnings

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


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

        self.n_mutatation = self.n_chromosomes - self.n_pool
        if self.n_mutatation == 0:
            raise ValueError("There are no mutation children. Please adjust the n_pool and n_elite.")

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

    def mutate(self, population) -> npt.ArrayLike:
        # calc. the population to mutate.
        chromosome_i_to_choose_from = np.arange(start=self.n_chromosomes - self.n_mutatation, stop=self.n_chromosomes)
        for chromsome_i in chromosome_i_to_choose_from:
            for gene_i in range(self.n_genes):
                population[chromsome_i, gene_i] = population[chromsome_i, gene_i] \
                                                  + np.std(population[:,gene_i]) * np.random.uniform(0, 1)
        return population

    @timer
    def solve(self) -> tuple:
        """
        Performs the genetic algorithm.
        :returns: a tuple of Numpy arrays of optimized parameters and objective function's value after each generation.
        """
        obj_func_value_array = np.array([]) # stores the objective function values after each generation.
        # First, initialize the population
        population = self.initialize_population()
        for generation_i in range(self.n_generations+1):
            # information for the user.
            print("Generation: ", generation_i)
            # Then, calculate the fitness
            fitness_array = self.calc_fitness(population=population)
            # Create the mating population
            population = self.mating_population(population=population, fitness_array=fitness_array)
            # Create new population based on elite populations, crossover, and mutations
            population = self.create_new_population(mating_population=population)
            population = self.mutate(population=population) # This population is used for the next iteration.
            # Update arrays
            obj_func_value = self.obj_func(population[0])
            obj_func_value_array = np.append(obj_func_value_array, obj_func_value)
            # Display information to the user.
            print(f"Optimized parameter list for generation {generation_i}: ", population[0])
            print(f"objective function output: {obj_func_value}")
        return population[0], obj_func_value_array

    def plot(self, obj_func_value_array:npt.ArrayLike) -> None:
        """
        Plots the objective function values at each generation.
        :params obj_func_value_array: (Numpy array) array containg the objective function values at each generation.
        """
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(obj_func_value_array)
        ax.set_xlabel("Generations")
        ax.set_ylabel("MSE")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def MSE(array_sim, array_actual) -> npt.ArrayLike:
        """
        Mean squared error.
        :param array_pred:
        :param array_actual:
        :return:
        """
        if len(array_sim) == len(array_actual):
            return np.mean(np.square(array_sim - array_actual))
        else:
            warnings.warn("Lengths of the vectors are not equal.")
            return np.mean(np.square(array_sim - array_actual[:len(array_sim)]))
