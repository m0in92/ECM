import unittest

import numpy as np

from ECM.kf.spkf import SPKF


def f_func(x_k, u_k, w_k):
    return np.sqrt(5+x_k) + w_k

def h_func(x_k, u_k, v_k):
    return x_k**3 + v_k


class TestMeasurementUpdateswith1DInputs(unittest.TestCase):
    def test_estimator_gain_matrix(self):
        x_hat_int = 2
        Ny = 1
        SigmaX, SigmaW, SigmaV = 1, 1, 2
        u = 0

        spkf_obj = SPKF(xhat=x_hat_int, Ny=Ny, SigmaX=SigmaX, SigmaW=SigmaW, SigmaV=SigmaV, f_func=f_func,
                        h_func=h_func)
        # Step 1a:
        Xx, xhat = spkf_obj.state_estimate_time_update(u=u)
        # Step 1b:
        Xs, SigmaX = spkf_obj.covariance_prediction(Xx=Xx, xhat=xhat)
        # Step 1c:
        Y, y_hat = spkf_obj.output_estimate(Xx=Xx, u=u)
        # Step 2a:
        SigmaY, Lx = spkf_obj.estimator_gain_matrix(Y=Y, yhat=y_hat, Xs=Xs)
        self.assertAlmostEqual(0.03457607, Lx[0,0])

    def test_state_measurement_update(self):
        x_hat_int = 2
        Ny = 1
        SigmaX, SigmaW, SigmaV = 1, 1, 2
        u = 0
        xhat_update_actual = 1.8966845497917038

        spkf_obj = SPKF(xhat=x_hat_int, Ny=Ny, SigmaX=SigmaX, SigmaW=SigmaW, SigmaV=SigmaV, f_func=f_func,
                        h_func=h_func)
        # Step 1a:
        Xx, xhat = spkf_obj.state_estimate_time_update(u=u)
        # Step 1b:
        Xs, SigmaX = spkf_obj.covariance_prediction(Xx=Xx, xhat=xhat)
        # Step 1c:
        Y, y_hat = spkf_obj.output_estimate(Xx=Xx, u=u)
        # Step 2a:
        SigmaY, Lx_g = spkf_obj.estimator_gain_matrix(Y=Y, yhat=y_hat, Xs=Xs)
        # Step 2b
        xhat_update = spkf_obj.state_update(L=Lx_g, xhat=xhat, ytrue=5.134553960326726, yhat=y_hat)
        self.assertAlmostEqual(xhat_update_actual, xhat_update[0,0])

    def test_cov_measurement_update(self):
        x_hat_int = 2
        Ny = 1
        SigmaX, SigmaW, SigmaV = 1, 1, 2
        u = 0
        cov_update_actual = 0.17865761849939343

        spkf_obj = SPKF(xhat=x_hat_int, Ny=Ny, SigmaX=SigmaX, SigmaW=SigmaW, SigmaV=SigmaV, f_func=f_func,
                        h_func=h_func)
        # Step 1a:
        Xx, xhat = spkf_obj.state_estimate_time_update(u=u)
        # Step 1b:
        Xs, SigmaX = spkf_obj.covariance_prediction(Xx=Xx, xhat=xhat)
        # Step 1c:
        Y, y_hat = spkf_obj.output_estimate(Xx=Xx, u=u)
        # Step 2a:
        SigmaY, Lx_g = spkf_obj.estimator_gain_matrix(Y=Y, yhat=y_hat, Xs=Xs)
        # Step 2b
        xhat_update = spkf_obj.state_update(L=Lx_g, xhat=xhat, ytrue=5.134553960326726, yhat=y_hat)
        # Step 2c
        SigmaX = spkf_obj.cov_measurement_update(Lx_g, SigmaX=SigmaX, SigmaY=SigmaY)
        self.assertAlmostEqual(cov_update_actual, SigmaX[0,0])


class TestEstimatesUsing2DInputs(unittest.TestCase):
    pass


