import unittest

import numpy as np

import ECM


class TestDiscreteTimeSolver(unittest.TestCase):
    def test_constructor_with_correct_parameters(self):
        def OCV_func(SOC):
            return 2.5 + 0.5 * SOC
        def eta_func(i):
            return 1.123
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
        i_app = np.array([1,1,1,1,1])
        solver = ECM.DTSolver(ECM_obj=ecm, isothermal=True, t_app=t_app, i_app=i_app)

        self.assertEqual(ecm.R0, solver.ECM_obj.R0) # just to check the ECM object is passed correctly
        self.assertEqual(True, solver.isothermal)
        self.assertTrue(np.array_equal(t_app, solver.t_app))
        self.assertTrue(np.array_equal(i_app, solver.i_app))

    def test_check_constructor_with_unequal_t_and_I(self):
        def OCV_func(SOC):
            return 2.5 + 0.5 * SOC
        def eta_func(i):
            return 1.123
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
        i_app = np.array([1,1,1,1])
        with self.assertRaises(TypeError):
            ECM.DTSolver(ECM_obj=ecm, isothermal=True, t_app=t_app, i_app=i_app)

    def test_check_constructor_with_t_app_as_float(self):
        def OCV_func(SOC):
            return 2.5 + 0.5 * SOC
        def eta_func(i):
            return 1.123
        R0 = 0.01
        R1 = 0.1
        C1 = 100
        capacity = 1
        SOC_0 = 0.5
        E_R0 = 1000
        E_R1 = 5000
        ecm = ECM.Thevenin1RC(R0=R0, R1=R1, C1=C1, OCV_func=OCV_func, eta_func=eta_func, capacity=capacity, SOC_0=SOC_0,
                          E_R0=E_R0, E_R1=E_R1, T_amb=298.15)

        t_app = 1.0
        i_app = np.array([1,1,1,1])
        with self.assertRaises(TypeError):
            ECM.DTSolver(ECM_obj=ecm, isothermal=True, t_app=t_app, i_app=i_app)

    def test_check_constructor_with_matrix_t(self):
        def OCV_func(SOC):
            return 2.5 + 0.5 * SOC

        def eta_func(i):
            return 1.123

        R0 = 0.01
        R1 = 0.1
        C1 = 100
        capacity = 1
        SOC_0 = 0.5
        E_R0 = 1000
        E_R1 = 5000
        ecm = ECM.Thevenin1RC(R0=R0, R1=R1, C1=C1, OCV_func=OCV_func, eta_func=eta_func, capacity=capacity, SOC_0=SOC_0,
                          E_R0=E_R0, E_R1=E_R1, T_amb=298.15)

        t_app = np.array([[1],[3],[4]])
        i_app = np.array([[1],[3],[4]])
        with self.assertRaises(TypeError):
            ECM.DTSolver(ECM_obj=ecm, isothermal=True, t_app=t_app, i_app=i_app)
