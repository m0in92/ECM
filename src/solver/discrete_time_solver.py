import numpy as np

from src.model.ecm import Thevenin1RC


class DT_solver:
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
        if isinstance(ECM_obj, Thevenin1RC):
            self.ECM_obj = ECM_obj
        else:
            raise TypeError("ECM_obj must be an instance of Thevenin1RC class.")
        if isinstance(isothermal, bool):
            self.isothermal = isothermal
        else:
            raise TypeError("isothermal must be a boolean (True or False).")
        # Ensure the t_app and i_app are numpy arrays of equal length. Furthermore, they should be vectors.
        if isinstance(t_app, np.ndarray):
            if t_app.ndim == 1:
                self.t_app = t_app
            else:
                raise TypeError("t_app must be a vector.")
        else:
            raise TypeError("t_app must be a numpy array.")
        if isinstance(i_app, np.ndarray):
            if len(i_app) == len(t_app):
                self.i_app = i_app
            else:
                raise TypeError("i_app and t_app must be of the same length.")
        else:
            raise TypeError("i_app needs to be numpy object and of the same length as t_app.")

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
