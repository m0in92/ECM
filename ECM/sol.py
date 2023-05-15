import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


class Solution:
    """
    Creates a solution object that stores the simulation results. The object methods provides the plotting
    functionalities.
    """
    def __init__(self, t_sim: npt.ArrayLike, i_sim: npt.ArrayLike, z_sim: npt.ArrayLike, v_sim: npt.ArrayLike,
                 t_actual: None|npt.ArrayLike= None, i_actual: None|npt.ArrayLike= None,
                 z_actual: None|npt.ArrayLike= None, v_actual: None|npt.ArrayLike =None) -> None:
        """
        Solution Constructor.
        :param t_sim: (Numpy array) simulation time array.
        :param i_sim: (Numpy array) simulation current array
        :param z_sim: (Numpy array) simulation SOC array
        :param v_sim: (Numpy array) simulation terminal voltage array.
        """
        if isinstance(t_sim, np.ndarray):
            self.t_array = t_sim

        if isinstance(i_sim, np.ndarray):
            if len(i_sim) == len(self.t_array):
                self.i_array = i_sim
            else:
                raise ValueError("Lengths of t_sim and i_sim do not match.")

        if isinstance(z_sim, np.ndarray):
            if len(z_sim) == len(self.t_array):
                self.z_array = z_sim
            else:
                raise ValueError("Lengths of t_sim and z_sim do not match.")

        if isinstance(v_sim, np.ndarray):
            if len(v_sim) == len(self.t_array):
                self.v_array = v_sim
            else:
                raise ValueError("Lengths of t_sim and z_sim do not match.")

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