import unittest

import numpy as np

from ECM.parameter_estimations.genetic_algorithm import GA


class TestGA(unittest.TestCase):
    def test_constructor_with_correct_inputs(self):
        n_genes = 3
        n_chromosomes = 10
        bounds=np.array([[0, 1], [0, 1], [0, 100]])
        n_pool = 7
        n_elite = 3
        n_generations = 3
        mutating_factor = 0.5
        def objective_func(param1, param2, param3):
            return param1 + param2 + param3

        ga = GA(n_chromosomes=n_chromosomes, bounds=np.array([[0,1],[0,1],[0,100]]), obj_func=objective_func,
                n_pool=n_pool, n_elite=n_elite, n_generations=n_generations, mutating_factor=mutating_factor)
        self.assertEqual(n_genes, ga.n_genes)
        self.assertEqual(n_chromosomes, ga.n_chromosomes)
        self.assertTrue(np.array_equal(bounds, ga.bounds))
        self.assertEqual(n_pool, ga.n_pool)
        self.assertEqual(n_elite, ga.n_elite)
        self.assertEqual(n_generations, ga.n_generations)
        self.assertEqual(mutating_factor, ga.mutating_factor)