import numpy as np

import ECM.solver.base


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
