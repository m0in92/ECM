"""
Copyright 2023 by Moin Ahmed. All Rights Reserved.

This example uses the dataset for the A123-LiFP cell found in CALCE public dataset (https://calce.umd.edu/battery-data).
Battery cell parameters:
    Capacity: 1.1 Ahr
    Ambient temperature: 20 degreesC
"""

import numpy as np
import pandas as pd # pandas package is imported for reading datasets.
import matplotlib.pyplot as plt

from ocv_soc_func import OCV_func, SOC_func
import ECM
from ECM.parameter_estimations.genetic_algorithm import GA


def eta_func(i):
    return 1 if i<=0 else 0.9995


# Read experimental data
t_lim_index = 5000
df = pd.read_csv('../../data/CALCE_A123/A1-A123-Dynamics.csv')
t = df['Test_Time(s)'].to_numpy()
I = df['Current(A)'].to_numpy()
V = df['Voltage(V)'].to_numpy()

# Define the objective function.
def obj_func(param_list):
    """
    Objective function for the genetic algorithm.
    :params param_list: (Numpy array) Numpy array of ECM parameters where the first element represents R0, second element
            represents R1, and the third elements represents C1
    """
    # Create an Thevenin1RC object
    ecm = ECM.Thevenin1RC(R0=param_list[0], R1=param_list[1], C1=param_list[2], OCV_func=OCV_func, eta_func=eta_func,
                          capacity=1.1,
                          SOC_0=0.38775,
                          E_R0=None, E_R1=None, T_amb=293.15)
    # Create a solver object and then call the solve method.
    ## remember the current convention: positive for discharge and negative for charge.
    solver = ECM.DTSolver(ECM_obj=ecm, isothermal=True, t_app=t[:t_lim_index], i_app=-I[:t_lim_index],
                          V_actual=V[:t_lim_index])
    try:
        sol = solver.solve()
        v_actual, z_pred, v_pred = sol.v_actual, sol.z_array, sol.v_array
        return GA.MSE(array_sim=v_pred, array_actual=v_actual)
    except:
        return 1000 # if simulation fails due to abnormal parameters, MSE defaults to this value


 # initiate the GA object and perform optimization through genetic_algorithm.
bounds = np.array([[0.01, 0.5],[0.001, 0.1],[0.01,1]])
ga = GA(n_chromosomes=10, bounds=bounds, obj_func=obj_func, n_pool=8, n_elite=3, n_generations=10,
         mutating_factor=0.8)
optizimized_param_list, mse_values = ga.solve()
ga.plot(obj_func_value_array=mse_values)

