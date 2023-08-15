import numpy as np

from ECM.calc_helpers.constants import PhysicsConstants
from ECM.parameter_set_manager import ParameterSets


class Thevenin1RC:
    """
    This class creates a first order Thevenin model object for a lithium-ion battery cell. It contains relevant model
    parameters as class attributes and methods to calculate SOC and terminal voltage.

    Thevenin first order model is a phenomenological model that can be used to simulate the terminal voltage across a
    lithium-ion battery cell. It has representative electrical components that represent the open-circuit voltage,
    internal resistance, and diffusion voltages. The set of differential and algebraic equations are:

    dz/dt = -eta(t) * i_app(t) / capacity
    di_R1/dt = -i_R1/(R1*C1) + i_app(t)/(R1*C1)
    v(t) = OCV(z(t)) - R1*i_R1(t) - R0*i_app(t)

    Where the second equation is a non-homogenous linear first-order differential equation. Furthermore, the variables
    are:
    z: state of charge (SOC)
    R0: resistance of the resistor that represents the battery cell's internal resistance
    R1: resistance of the resistor in the RC pair.
    C1: capacitance of the capacitor in the RC pair.
    i_R1: current through R1
    i_app: applied current
    eta: Colombic efficiency

    Note that the RC pair represents the diffusion voltage in the battery cell.


    After time discretization, the set of algebraic equations are:

    z[k+1] = z[k] - delta_t*eta[k]*i_app[k]/capacity
    i_R1[k+1] = exp(-delta_t/(R1*C1))*i_R1[k] + (1-exp(-delta_t/(R1*C1))) * i_app[k]
    v[k] = OCV(z[k]) - R1*i_R1[k] - R0*i_app[k]

    Where k represents the time-point and delta_t represents the time-step between z[k+1] and z[k].

    Code Notes:
    1. It is assumed for now that eta is a function of applied current only.
    2. Discharge currrent is positve and charge current is negative by convention.
    """
    def __init__(self, param_set_name: str, SOC_init: float, T_amb=298.15):
        # def __init__(self, R0, R1, C1, OCV_func, eta_func, capacity, SOC_0, E_R0=None, E_R1=None, T_amb=298.15)
        """
        Constructor class for Thevenin1RC class.
        """
        param_set = ParameterSets(name= param_set_name)
        self.R0_ref = param_set.R0
        self.R1_ref = param_set.R1
        self.C1 = param_set.C1
        self.OCV_func = param_set.func_OCV
        # # Ensure OCV_func and eta_func are function objects.
        # if callable(OCV_func):
        #     self.OCV_func = OCV_func
        # else:
        #     raise TypeError("OCV must be a function object.")
        self.eta_func = param_set.func_eta
        # if callable(eta_func):
        #     self.eta_func = eta_func
        # else:
        #     raise TypeError("eta must be a function object.")
        self.capacity = param_set.cap
        self.SOC = SOC_init # Initial condition for the SOC
        self.E_R0 = param_set.E_R0
        self.E_R1 = param_set.E_R1
        self.i_R1 = 0.0  # Initial condition for i_R1
        self.T_ref = T_amb  # Reference temperature
        self.T = T_amb # Initial condition for the temperature

    @property
    def R0(self):
        """
        calculates the internal resistance of the battery cell at a given temperature.
        :param T: (float) temperature of the battery cell.
        :return: (float) internal resistance of the battery cell.
        """
        if self.E_R0 is not None:
            return self.R0_ref * np.exp(self.E_R0 / PhysicsConstants.R * (1/self.T_ref - 1/self.T))
        else:
            return self.R0_ref

    @property
    def R1(self):
        """
        calculates the internal resistance of the battery cell at a given temperature.
        :param T: (float) temperature of the battery cell.
        :return: (float) internal resistance of the battery cell.
        """
        if self.E_R1 is not None:
            return self.R1_ref * np.exp(self.E_R1 / PhysicsConstants.R * (1/self.T_ref - 1/self.T))
        else:
            return self.R1_ref

    @staticmethod
    def delta_t(t_next, t_current):
        """
        This method calculates the time-step between two time-points.
        :param t_next: (float) time at time step, k+1
        :param t_current: (float) time at current time step, k
        :return: (float) time period between two time steps, k+1 and k
        """
        return t_next - t_current

    def SOC_next(self, i_app, t_next, t_current):
        """
        This methods calculates the SOC at the next time-step
        :param i_app: (float) applied current at the current time step, k
        :param t_next: (float) time at time step, k+1
        :param t_current: (float) time at current step, k
        :return: (float) SOC at time step, k+1
        """
        delta_t = self.delta_t(t_next, t_current)
        return self.SOC - delta_t * self.eta_func(i_app) * i_app / (3600 * self.capacity)

    def i_R1_next(self, i_app, t_next, t_current):
        """Measures the current through R1 (i_R1) at the next time step.
        :param i_app: (float) applied current at the current time step, k
        :param t_next: (float) time at time step, k+1
        :param t_current: (float) time at current step, k
        """
        delta_t = self.delta_t(t_next, t_current)
        return np.exp(-delta_t/(self.R1*self.C1)) * self.i_R1 + (1-np.exp(-delta_t/(self.R1*self.C1))) * i_app

    def v(self, i_app):
        """
        This method calculates the cell terminal voltage.
        :param i_app: (float) applied current at current time step, k
        :return: (float) terminal voltage at the current time step, k
        """
        return self.OCV_func(self.SOC) - self.R1 * self.i_R1 - self.R0 * i_app


