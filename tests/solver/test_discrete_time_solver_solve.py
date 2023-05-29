import unittest

import numpy as np

import ECM


class TestDiscreteTimeSolver(unittest.TestCase):
    def test_constructor_with_correct_parameters(self):
        def OCV_func(SOC):
            return 2.5 + 0.5 * SOC
        def eta_func(i):
            return 1
        R0 = 0.01
        R1 = 0.1
        C1 = 100
        capacity = 1
        SOC_0 = 0.5
        E_R0 = 1000
        E_R1 = 5000
        ecm = ECM.Thevenin1RC(R0=R0, R1=R1, C1=C1, OCV_func=OCV_func, eta_func=eta_func, capacity=capacity, SOC_0=SOC_0,
                          E_R0=E_R0, E_R1=E_R1, T_amb=298.15)

        t_app = np.array([1,2,3,4,5])
        i_app = np.array([1, 1, 1, 1, 1])
        solver = ECM.DTSolver(ECM_obj=ecm, isothermal=True, t_app=t_app, i_app=i_app)
        sol = solver.solve()

        z_sol = np.array([0.5, 0.49972222, 0.49944444, 0.49916667, 0.49888889])
        v_sol = np.array([2.74, 2.73034485, 2.7215953, 2.71366516, 2.70647645])
        self.assertTrue(np.all(np.isclose(z_sol, sol.z_sim)))
        self.assertTrue(np.all(np.isclose(v_sol, sol.v_sim)))

