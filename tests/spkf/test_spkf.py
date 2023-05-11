import unittest

import numpy as np

from ECM.kf.spkf import SPKF


def f_func(x_k, u_k, w_k):
    return np.sqrt(5+x_k) + w_k

def h_func(x_k, u_k, v_k):
    return x_k**3 + v_k


class TestSPKFConstructor(unittest.TestCase):
    def test_initialization_with_correct_array_inputs(self):
        x_hat_array = np.array([2, 2])
        aug_state_vector_actual = np.array([[2],[2],[0],[0]])
        aug_cov_matrix_actual = np.diag([2, 2, 1, 2])
        SigmaX = np.array([[2,0],[0,2]])

        spkf_obj = SPKF(xhat=np.array(x_hat_array), Ny=1, SigmaX=SigmaX, SigmaW=1, SigmaV=2, f_func=f_func,
                        h_func=h_func)
        self.assertTrue(np.array_equal(aug_state_vector_actual, spkf_obj.aug_state_vector))
        self.assertTrue(np.array_equal(aug_cov_matrix_actual ,spkf_obj.aug_cov_matrix))
        self.assertEqual(9, spkf_obj.p + 1) # No. of generated sigma points

    def test_initialization_with_correct_matrix_inputs(self):
        x_hat_array = np.array([[2]])
        aug_state_vector_actual = np.array([[2], [0], [0]])
        aug_cov_matrix_actual = np.diag([2, 1, 2])

        spkf_obj = SPKF(xhat=np.array(x_hat_array), Ny=1, SigmaX=np.array([[2]]), SigmaW=1, SigmaV=2,
                        f_func=f_func,
                        h_func=h_func)
        self.assertTrue(np.array_equal(aug_state_vector_actual, spkf_obj.aug_state_vector))
        self.assertTrue(np.array_equal(aug_cov_matrix_actual, spkf_obj.aug_cov_matrix))
        self.assertEqual(7, spkf_obj.p + 1)  # No. of generated sigma points

    def test_initialization_with_correct_int_inputs(self):
        x_hat_int = 2
        aug_state_vector_actual = np.array([[2],[0],[0]])
        aug_cov_matrix_actual = np.diag([2, 1, 2])

        spkf_obj = SPKF(xhat= x_hat_int, Ny=1, SigmaX=2, SigmaW=1, SigmaV=2, f_func=f_func, h_func=h_func)
        self.assertTrue(np.array_equal(aug_state_vector_actual, spkf_obj.aug_state_vector))
        self.assertTrue(np.array_equal(aug_cov_matrix_actual ,spkf_obj.aug_cov_matrix))
        self.assertEqual(7, spkf_obj.p + 1) # No. of generated sigma points


class TestCDKFAttributes(unittest.TestCase):
    def test_tuning_parameters(self):
        x_hat_int = 2

        spkf_obj = SPKF(xhat=x_hat_int, Ny=1, SigmaX=2, SigmaW=1, SigmaV=2, f_func=f_func, h_func=h_func)
        self.assertTrue(np.sqrt(3), spkf_obj.gamma)
        self.assertEqual(np.sqrt(3), spkf_obj.h)

    def test_alpha_m0(self):
        x_hat_int = 2

        spkf_obj = SPKF(xhat=x_hat_int, Ny=1, SigmaX=2, SigmaW=1, SigmaV=2, f_func=f_func, h_func=h_func)
        self.assertAlmostEqual(0, spkf_obj.alpha_m0)

    def test_aplha_m(self):
        x_hat_int = 2

        spkf_obj = SPKF(xhat=x_hat_int, Ny=1, SigmaX=2, SigmaW=1, SigmaV=2, f_func=f_func, h_func=h_func)
        self.assertAlmostEqual(1 / 6, spkf_obj.alpha_m)

    def test_aplha_m_vec(self):
        x_hat_int = 2

        spkf_obj = SPKF(xhat=x_hat_int, Ny=1, SigmaX=2, SigmaW=1, SigmaV=2, f_func=f_func, h_func=h_func)
        self.assertTrue(np.all(np.isclose(np.array([[0],[1/6],[1/6],[1/6],[1/6],[1/6],[1/6]]), spkf_obj.alpha_m_vec)))

    def test_alpha_c0(self):
        x_hat_int = 2

        spkf_obj = SPKF(xhat=x_hat_int, Ny=1, SigmaX=2, SigmaW=1, SigmaV=2, f_func=f_func, h_func=h_func)
        self.assertAlmostEqual(0, spkf_obj.alpha_m0)

    def test_aplha_c(self):
        x_hat_int = 2

        spkf_obj = SPKF(xhat=x_hat_int, Ny=1, SigmaX=2, SigmaW=1, SigmaV=2, f_func=f_func, h_func=h_func)
        self.assertAlmostEqual(1 / 6, spkf_obj.alpha_m)

    def test_alpha_c_vec(self):
        x_hat_int = 2
        actual_alpha_c_vec = np.array([[0], [1 / 6], [1 / 6], [1 / 6], [1 / 6], [1 / 6], [1 / 6]])

        spkf_obj = SPKF(xhat= x_hat_int, Ny=1, SigmaX=2, SigmaW=1, SigmaV=2, f_func=f_func, h_func=h_func)
        self.assertTrue(np.all(np.isclose(actual_alpha_c_vec, spkf_obj.alpha_c_vec)))


class TestSigmaPoints(unittest.TestCase):
    def test_sqrt_aug_cov_matrix(self):
        x_hat_int = 2
        Ny = 1
        SigmaX, SigmaW, SigmaV = 1, 1, 2
        sqrt_aug_cov_matrix_actual = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.41421356]])

        spkf_obj = SPKF(xhat=x_hat_int, Ny=Ny, SigmaX=SigmaX, SigmaW=SigmaW, SigmaV=SigmaV, f_func=f_func, h_func=h_func)
        self.assertTrue(np.all(np.isclose(sqrt_aug_cov_matrix_actual, spkf_obj.sqrt_aug_cov_matrix)))

    def test_SigmaPoint_matrix(self):
        x_hat_int = 2
        Ny = 1
        SigmaX, SigmaW, SigmaV = 1, 1, 2
        sigma_point_matrix_actual = np.array([[2,3.73205081,2,2,0.26794919,2,2],
                                              [0,0,1.73205081,0,0,-1.73205081,0],
                                              [0,0,0,2.44948974,0,0,-2.44948974]])

        spkf_obj = SPKF(xhat=x_hat_int, Ny=Ny, SigmaX=SigmaX, SigmaW=SigmaW, SigmaV=SigmaV, f_func=f_func, h_func=h_func)
        self.assertTrue(np.all(np.isclose(sigma_point_matrix_actual, spkf_obj.sigma_point_matrix)))







