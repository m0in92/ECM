import unittest

import numpy as np

from src.model.ecm import Thevenin1RC


class TestThevenin1RC(unittest.TestCase):
    def test_constructor_correct_inputs(self):
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
        ecm = Thevenin1RC(R0=R0, R1=R1, C1=C1, OCV_func=OCV_func, eta_func=eta_func, capacity=capacity, SOC_0=SOC_0,
                          E_R0=E_R0, E_R1=E_R1, T_amb=298.15)
        self.assertEqual(R0, ecm.R0_ref)
        self.assertEqual(R0, ecm.R0)
        self.assertEqual(R1, ecm.R1_ref)
        self.assertEqual(R1, ecm.R1)
        self.assertEqual(C1, ecm.C1)
        # test OCV at differnet SOC
        self.assertEqual(OCV_func(0.25), ecm.OCV_func(0.25))
        self.assertEqual(OCV_func(0.5), ecm.OCV_func(0.5))
        self.assertEqual(OCV_func(0.75), ecm.OCV_func(0.75))
        # continue testing
        self.assertEqual(1.123, ecm.eta_func(1))
        self.assertEqual(SOC_0, ecm.SOC)
        self.assertEqual(E_R0, ecm.E_R0)
        self.assertEqual(E_R1, ecm.E_R1)

    def test_constructor_not_OCV_funcs(self):
        def eta_func(i):
            return 1.123
        R0 = 0.01
        R1 = 0.1
        C1 = 100
        capacity = 1
        SOC_0 = 0.5
        E_R0 = 1000
        E_R1 = 5000
        OCV_func = 1.0
        with self.assertRaises(TypeError):
            Thevenin1RC(R0=R0, R1=R1, C1=C1, OCV_func=OCV_func, eta_func=eta_func, capacity=capacity, SOC_0=SOC_0,
                        E_R0=E_R0, E_R1=E_R1)

    def test_constructor_not_eta_funcs(self):
        def OCV_func(SOC):
            return 2.5 + 0.5 * SOC
        eta_func = 1.345
        R0 = 0.01
        R1 = 0.1
        C1 = 100
        capacity = 1
        SOC_0 = 0.5
        E_R0 = 1000
        E_R1 = 5000
        OCV_func = 1.0
        with self.assertRaises(TypeError):
            Thevenin1RC(R0=R0, R1=R1, C1=C1, OCV_func=OCV_func, eta_func=eta_func, capacity=capacity, SOC_0=SOC_0,
                        E_R0=E_R0, E_R1=E_R1)

    def test_R0(self):
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
        T_amb = 298.15
        ecm = Thevenin1RC(R0=R0, R1=R1, C1=C1, OCV_func=OCV_func, eta_func=eta_func, capacity=capacity, SOC_0=SOC_0,
                          E_R0=E_R0, E_R1=E_R1, T_amb=T_amb)
        # at reference temperature
        ecm.T = T_amb
        self.assertEqual(R0, ecm.R0)
        # at higher temperature. The internal resistance should increase
        ecm.T = 313.15
        self.assertEqual(0.01019511779991307, ecm.R0)
        # at lower temperature. The internal resistance should decrease
        ecm.T = 273.15
        self.assertEqual(0.009637505879223662, ecm.R0)

    def test_R1(self):
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
        T_amb = 298.15
        ecm = Thevenin1RC(R0=R0, R1=R1, C1=C1, OCV_func=OCV_func, eta_func=eta_func, capacity=capacity, SOC_0=SOC_0,
                          E_R0=E_R0, E_R1=E_R1, T_amb=T_amb)
        # at reference temperature
        ecm.T = T_amb
        self.assertEqual(R1, ecm.R1)
        # at higher temperature. The internal resistance should increase
        ecm.T = 313.15
        self.assertEqual(0.11014410062791449, ecm.R1)
        # at lower temperature. The internal resistance should decrease
        ecm.T = 273.15
        self.assertEqual(0.08314253845431369, ecm.R1)

    def test_delta_t(self):
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
        T_amb = 298.15
        ecm = Thevenin1RC(R0=R0, R1=R1, C1=C1, OCV_func=OCV_func, eta_func=eta_func, capacity=capacity, SOC_0=SOC_0,
                          E_R0=E_R0, E_R1=E_R1, T_amb=T_amb)

        t_current = 1
        t_next = 3
        self.assertEqual(2, ecm.delta_t(t_next=t_next, t_current=t_current))

    def test_iR_next(self):
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
        T_amb = 298.15
        ecm = Thevenin1RC(R0=R0, R1=R1, C1=C1, OCV_func=OCV_func, eta_func=eta_func, capacity=capacity, SOC_0=SOC_0,
                          E_R0=E_R0, E_R1=E_R1, T_amb=T_amb)

        t_current = 1
        t_next = 3
        i_app = 1
        self.assertEqual(np.exp(-2/(0.1*100))*0+(1-np.exp(-2/(0.1*100)))*1, ecm.i_R1_next(i_app=i_app, t_next=t_next,
                                                                                          t_current=t_current))

    def test_v(self):
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
        T_amb = 298.15
        ecm = Thevenin1RC(R0=R0, R1=R1, C1=C1, OCV_func=OCV_func, eta_func=eta_func, capacity=capacity, SOC_0=SOC_0,
                          E_R0=E_R0, E_R1=E_R1, T_amb=T_amb)
        i_app = 1
        self.assertEqual(2.5+0.5*0.5-0.01*1, ecm.v(i_app=i_app))



