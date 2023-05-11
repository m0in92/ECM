import unittest

import numpy as np

from ECM.kf.spkf import SPKF


def f_func(x_k, u_k, w_k):
    return np.sqrt(5+x_k) + w_k

def h_func(x_k, u_k, v_k):
    return x_k**3 + v_k


class test_SPKF(unittest.TestCase):
    def test_state_time_estimate(self):
        x_hat_int = 2
        Ny = 1
        SigmaX, SigmaW, SigmaV = 1, 1, 2
        x_estimate_actual = 2.638868491883301

        spkf_obj = SPKF(xhat=x_hat_int, Ny=Ny, SigmaX=SigmaX, SigmaW=SigmaW, SigmaV=SigmaV, f_func=f_func, h_func=h_func)
        self.assertAlmostEqual(x_estimate_actual, spkf_obj.state_estimate_time_update(0)[1][0,0])

    def test_error_covariance_time_update(self):
        x_hat_int = 2
        Ny = 1
        SigmaX, SigmaW, SigmaV = 1, 1, 2
        x_cov_actual = 1.0363730825455537

        spkf_obj = SPKF(xhat=x_hat_int, Ny=Ny, SigmaX=SigmaX, SigmaW=SigmaW, SigmaV=SigmaV, f_func=f_func,
                        h_func=h_func)
        Xx, xhat = spkf_obj.state_estimate_time_update(0)
        self.assertAlmostEqual(x_cov_actual, spkf_obj.covariance_prediction(Xx=Xx, xhat=xhat)[1][0, 0])

    def test_output_prediction(self):
        x_hat_int = 2
        Ny = 1
        SigmaX, SigmaW, SigmaV = 1, 1, 2
        u = 0
        Y_actual = np.array([18.52025918, 25.80324827, 83.90124036, 20.96974892, 12.09100405, 0.7628016, 16.07076943])
        output_pred_actual = 26.5998021

        spkf_obj = SPKF(xhat=x_hat_int, Ny=Ny, SigmaX=SigmaX, SigmaW=SigmaW, SigmaV=SigmaV, f_func=f_func,
                        h_func=h_func)
        Xx, xhat = spkf_obj.state_estimate_time_update(u=u)
        SigmaX = spkf_obj.covariance_prediction(Xx=Xx, xhat=xhat)
        Y, y_hat = spkf_obj.output_estimate(Xx=Xx, u=u)
        self.assertTrue(np.all(np.isclose(Y_actual, Y)))
        self.assertAlmostEqual(output_pred_actual, y_hat[0,0])