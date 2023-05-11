import numpy as np

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