import numpy as np

import ECM.solver.base
from ECM.kf.spkf import SPKF


class DT_solver(ECM.solver.base.BaseSolver):
    """
    This is the class that solves the ECM model equations for time and applied current arrays. It outputs the terminal
    SOC and voltage at the time steps specified by the input time array.

    The discrete time model for first-order Thevenin model is given by:

    z[k+1] = z[k] - delta_t*eta[k]*i_app[k]/capacity
    i_R1[k+1] = exp(-delta_t/(R1*C1))*i_R1[k] + (1-exp(-delta_t/(R1*C1))) * i_app[k]
    v[k] = OCV(z[k]) - R1*i_R1[k] - R0*i_app[k]

    Where k represents the time-point and delta_t represents the time-step between z[k+1] and z[k].
    """

    def __init__(self, ECM_obj, isothermal, t_app, i_app):
        super().__init__(ECM_obj=ECM_obj, isothermal=isothermal, t_app=t_app, i_app=i_app)

    def solve(self, verbose=False):
        # list for storing SOC and voltage values
        z_array, v_array = np.zeros(len(self.t_app)), np.zeros(len(self.t_app))  # initial conditions
        z_array[0] = self.ECM_obj.SOC  # add initial SOC to the array
        for k in range(len(self.t_app)-1):
            # calc. the SOC of the next time step
            t_next = self.t_app[k + 1]
            t_current = self.t_app[k]
            z_array[k + 1] = self.ECM_obj.SOC_next(i_app=self.i_app[k], t_next=t_next, t_current=t_current)
            i_R1_next = self.ECM_obj.i_R1_next(i_app=self.i_app[k], t_next=t_next, t_current=t_current)
            v_array[k] = self.ECM_obj.v(i_app=self.i_app[k])
            self.ECM_obj.SOC = z_array[k + 1]  # update ECM_obj's SOC attribute for the next iteration.
            self.ECM_obj.i_R1 = i_R1_next  # update ECM's obj i_R1 attribute for the next iteration.
            if verbose:
                print('k: ',k ,', t [s]: ', t_current, ' ,I [A]: ', self.i_app[k], ' , SOC: ', z_array[k],', V [V]: ', v_array[k])
        v_array[-1] = self.ECM_obj.v(i_app=self.i_app[-1])
        return z_array, v_array


class DT_solver_spkf(ECM.solver.base.BaseSolver):
    def __init__(self, ECM_obj, isothermal, t_app, i_app, SigmaX, SigmaW, SigmaV, V_actual):
        super().__init__(ECM_obj=ECM_obj, isothermal=isothermal, t_app=t_app, i_app=i_app)

        # Initialize SPKF object
        xhat_init = np.array([[self.ECM_obj.SOC],[0]])
        self.spkf_object = SPKF(xhat=xhat_init, Ny=1, SigmaX = SigmaX, SigmaW=SigmaW, SigmaV=SigmaV, f_func=self.f_func,
                                h_func=self.h_func)

        # Sensor readings
        if isinstance(V_actual, np.ndarray):
            if len(V_actual) == len(self.t_app):
                self.V_actual = V_actual
            else:
                raise ValueError("length of V_actual needs to be equal to t_app.")
        else:
            raise TypeError("V_actual needs to be a Numpy array.")

        # Other variables
        self.delta_t = 0 # this class attribute is introduced since delta_t is required in state equation.

    def f_func(self, x_k, u_k, w_k):
        """
        State Equation.
        :param x_k:
        :param u_k:
        :param w_k:
        :param delta_t:
        :return:
        """
        R1 = self.ECM_obj.R1
        C1 = self.ECM_obj.C1
        m1 = np.array([[1, 0], [0, np.exp(-self.delta_t / (R1*C1))]])
        m2 = np.array([[-self.delta_t / (3600 * self.ECM_obj.capacity)], [1 - np.exp(-self.delta_t / (R1 * C1))]])
        return m1 @ x_k + m2 * (u_k + w_k)

    def h_func(self, x_k, u_k, v_k):
        """
        Output Equation.
        :param x_k:
        :param u_k:
        :param v_k:
        :return:
        """
        testing = self.ECM_obj.OCV_func(x_k[0, :]) - self.ECM_obj.R1 * x_k[1, :] - self.ECM_obj.R0 * u_k + v_k
        return self.ECM_obj.OCV_func(x_k[0, :]) - self.ECM_obj.R1 * x_k[1, :] - self.ECM_obj.R0 * u_k + v_k

    def solve(self):
        # Introduce arrays for SOC (represented by z) and V (represented by v) storage.
        z_array, v_array = np.array([]), np.array([])
        for k_ in range(len(self.t_app)-1):
            t_current = self.t_app[k_]
            t_next = self.t_app[k_+1]
            self.delta_t = self.ECM_obj.delta_t(t_next, t_current)
            u = self.i_app[k_]

            self.spkf_object.solve(u=u, ytrue=self.V_actual[k_])

            z_array = np.append(z_array, self.spkf_object.xhat[0,0])
            v_array = np.append(v_array, self.spkf_object.yhat[0])

        return z_array, v_array

