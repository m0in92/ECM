import unittest
from ECM.utils.time_related import timer

class TestTimer(unittest.TestCase):
    def test_timer_functionality(self):
        """
        This test checks if the output from the timer function is a function.
        """
        def solver_example(a,b):
            return a + b
        a = 1
        b = 1
        sample_variable = timer(solver_example)
        self.assertTrue(callable(sample_variable))
