import unittest
from ECM.parameter_set_manager import ParameterSets


class TestParameterSets(unittest.TestCase):
    def test_constructor(self):
        param_set = ParameterSets('test')
        self.assertEqual(0.225, param_set.R0)
        self.assertEqual(0.001, param_set.R1)
        self.assertEqual(0.03, param_set.C1)
        self.assertEqual(1.1, param_set.cap)
        self.assertEqual(0, param_set.E_R0)
        self.assertEqual(0, param_set.E_R1)

        with self.assertRaises(expected_exception=Exception) as e:
            ParameterSets(name="no_such_parameter_set")
