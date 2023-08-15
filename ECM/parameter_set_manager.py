import os
import importlib

import pandas as pd

from ECM.config import definations


class ParameterSets:
    PARAMETER_SET_DIR = definations.PARAMETER_SET_DIR

    def __init__(self, name: str):
        self.check_parameter_set(name)
        self.name = name

        FOLDER_DIR = os.path.join(self.PARAMETER_SET_DIR , self.name)
        df = self.parse_csv(os.path.join(FOLDER_DIR, "params.csv"))
        self.R0 = df["R0 [ohms]"]
        self.R1 = df["R1 [ohms]"]
        self.C1 = df["C1 [F]"]
        self.cap = df["capacity [Ahr]"]
        self.E_R0 = df["E_R0 [J/mol]"]
        self.E_R1 = df["E_R1 [J/mol]"]

        func_module = importlib.import_module(f'parameter_sets.{self.name}.funcs')  # imports the python module
        self.func_OCV = func_module.func_OCV
        self.func_SOC = func_module.func_SOC
        self.func_eta = func_module.func_eta

    @classmethod
    def list_parameters_sets(cls):
        """
        Returns the list of available parameter sets.
        :return: (list) list of available parameters sets.
        """
        return os.listdir(cls.PARAMETER_SET_DIR)

    @classmethod
    def check_parameter_set(cls, name):
        """
        Checks if the inputted parameter name is in the parameter set. If not available, it raises an exception.
        """
        if name not in cls.list_parameters_sets():
            raise ValueError(f'{name} no in parameter sets.')

    @classmethod
    def parse_csv(cls, file_path):
        """
        reads the csv file and returns a Pandas DataFrame.
        :param file_path: the absolute or relative file drectory of the csv file.
        :return: the dataframe with the column containing numerical values only.
        """
        return pd.read_csv(file_path, index_col=0)["Value"]


