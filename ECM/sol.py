import numpy as np
import matplotlib.pyplot as plt


class Solution:
    """
    Creates a solution object that stores the simulation results. The object methods provides the plotting
    functionalities.
    """
    def __init__(self, t_sim, i_sim, z_sim, v_sim, t_actual=None, i_actual=None, z_actual=None, v_actual=None):
        """
        Solution Constructor
        """
        if isinstance(t_sim, np.ndarray):
            self.t_array = t_sim

        if isinstance(i_sim, np.ndarray):
            self.i_array = i_sim

        if isinstance(z_sim, np.ndarray):
            self.z_array = z_sim

        if isinstance(v_sim, np.ndarray):
            self.v_array = v_sim

        self.t_actual = t_actual
        if isinstance(t_actual, np.ndarray):
            self.t_actual = t_actual

        self.i_actual = i_actual
        if isinstance(i_actual, np.ndarray):
            self.i_actual = i_actual

        self.z_actual = z_actual
        if isinstance(z_actual, np.ndarray):
            self.z_actual = z_actual

        self.v_actual = v_actual
        if isinstance(v_actual, np.ndarray):
            if isinstance(t_actual, np.ndarray):
                if len(v_actual) == len(t_actual):
                    self.v_actual = v_actual
                else:
                    raise ValueError(f"Length of v_actual, {len(v_actual)}, does not match the length of t_actual "
                                     f"{len(t_actual)}")
            else:
                raise TypeError("Since t_actual is Numpy array, so please input v_actual as a Numpy array as well.")


    def plot(self):
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)

        if self.v_actual is not None:
            ax1.plot(self.t_actual, self.v_actual, label="exp.")
        ax1.plot(self.t_array, self.v_array, label="pred.")
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('V [V]')
        ax1.legend()

        ax2.plot(self.t_array, self.z_array, label="SOC pred.")
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('SOC')

        ax3.plot(self.t_array, self.i_array, label="I_exp")
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('I [A]')

        plt.tight_layout()
        plt.show()