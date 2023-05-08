import unittest

import numpy as np

from src.model.ecm import Thevenin1RC
from src.solver.discrete_time_solver import DT_solver


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
        ecm = Thevenin1RC(R0=R0, R1=R1, C1=C1, OCV_func=OCV_func, eta_func=eta_func, capacity=capacity, SOC_0=SOC_0,
                          E_R0=E_R0, E_R1=E_R1, T_amb=298.15)

        t_app = np.array([1,2,3,4,5])
        i_app = np.array([0.1,0.1,0.1,0.1,0.1])
        solver = DT_solver(ECM_obj=ecm, isothermal=True, t_app=t_app, i_app=i_app)
        z_pred, v_pred = solver.solve()

        z_sol = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
        v_sol = np.array([2.749, 2.69804837, 2.64718731, 2.59640818, 2.5457032 ])
        self.assertTrue(np.all(np.isclose(z_sol, z_pred)))
        self.assertTrue(np.all(np.isclose(v_sol, v_pred)))

