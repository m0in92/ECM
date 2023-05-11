import numpy as np
import warnings


def match_sim_actual_array(t_sim_array, t_actual_array, sim_array):
    """
    This method identifies the gaps in the actual/experimental data and remove them so that
    the lengths of the prediction and actual arrays match.
    :param t_sim_array:
    :param t_actual_array:
    :param sim_array:
    :return:
    """
    result_array = np.array([]) # array to be outputted.
    for k, time in enumerate(t_sim_array):
        if time in t_actual_array:
            # if condition is true, add the simulated value at this index to result array
            result_array = np.append(result_array, sim_array[k])
    return result_array


def MSE(array_sim, array_actual):
    """
    Mean squared error.
    :param array_pred:
    :param array_actual:
    :return:
    """
    if len(array_sim) == len(array_actual):
        return np.mean(np.square(array_sim - array_actual))
    else:
        warnings.warn("Lengths of the vectors are not equal.")
        return np.mean(np.square(array_sim - array_actual[:len(array_sim)]))