import warnings

import numpy as np
import numpy.typing as npt
import scipy.interpolate

from ECM.model.ecm import Thevenin1RC


class BaseSolver:
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

    @staticmethod
    def match_sim_actual_array(t_sim_array: npt.ArrayLike,
                               t_actual_array: npt.ArrayLike,
                               sim_array: npt.ArrayLike,
                               method: str = 'equal') -> npt.ArrayLike:
        """
        This method identifies the gaps in the actual/experimental data and remove them so that
        the lengths of the prediction and actual arrays match.
        :param t_sim_array: (Numpy Array)
        :param t_actual_array: (Numpy Array)
        :param sim_array: (Numpy Array)
        :param method: (string) can be 'equal' or 'interpolation'
        :return: (Numpy Array) The array of simulated values corresponding to the experimental values.
        """
        result_array = np.array([])  # array to be outputted.
        if method == 'equal':
            for k, time in enumerate(t_sim_array):
                if time in t_actual_array:
                    # if condition is true, add the simulated value at this index to result array
                    result_array = np.append(result_array, sim_array[k])
            if result_array.shape[0] == 0:
                warnings.warn("No match between the t_sim_array and t_actual_array.")
        elif method == 'interpolation':
            sim_array_interp = scipy.interpolate.interp1d(t_sim_array, sim_array)
            result_array = sim_array_interp(t_actual_array)
        return result_array