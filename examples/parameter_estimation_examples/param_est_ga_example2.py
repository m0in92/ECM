"""
Copyright 2023 by Moin Ahmed. All Rights Reserved.

This example uses genetic algorithm for parameter estimation. The solver used is HybridSolver.

This example uses the dataset for the A123-LiFP cell found in CALCE public dataset (https://calce.umd.edu/battery-data).
Furthermore, Sigma-point Kalman filter is used at time-periods where experimental data is available.

Battery cell parameters:
    Capacity: 1.1 Ahr
    Ambient temperature: 20 degreesC
"""
import numpy as np
import pandas as pd

from ocv_soc_func import OCV_func, SOC_func
import ECM
from ECM.parameter_estimations.genetic_algorithm import GA


def eta_func(i):
    return 1 if i<=0 else 0.9995


t_lim_index = 1000

# Read experimental data
df = pd.read_csv('../../data/CALCE_A123/A1-A123-Dynamics.csv')
t = df['Test_Time(s)'].to_numpy()
I = df['Current(A)'].to_numpy()
V = df['Voltage(V)'].to_numpy()

# SPKF parameters
SigmaX = np.array([[1e-8, 0], [0, 1e-8]])
SigmaW, SigmaV = 1e-8, 1e-8


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
    solver = ECM.HybridDiscreteSolver(ECM_obj=ecm, isothermal=True, t_app=t[:t_lim_index], i_app=-I[:t_lim_index],
                                      SigmaX=SigmaX, SigmaW=SigmaW, SigmaV=SigmaV, V_actual=V[:t_lim_index],
                                      t_start=0, t_end=t[t_lim_index], t_interval=1)
    sol = solver.solve()
    t_actual, v_actual, t_sim, z_pred, v_pred = sol.t_actual, sol.v_actual, sol.t_array, sol.z_array, sol.v_array
    v_pred = ECM.HybridDiscreteSolver.match_sim_actual_array(t_sim_array=t_sim, t_actual_array=t_actual,
                                                             sim_array=v_pred, method='interpolation')
    return GA.MSE(array_sim=v_pred, array_actual=v_actual)


# initiate the GA object and perform optimization through genetic_algorithm.
bounds = np.array([[0.01, 0.5],[0.001, 0.1],[0.01,1]])
ga = GA(n_chromosomes=10, bounds=bounds, obj_func=obj_func, n_pool=7, n_elite=3, n_generations=1,
         mutating_factor=0.8)
optizimized_param_list, mse_values = ga.solve()
ga.plot(obj_func_value_array=mse_values)

# Create ECM object and plot
ecm = ECM.Thevenin1RC(R0=optizimized_param_list[0], R1=optizimized_param_list[1], C1=optizimized_param_list[2],
                      OCV_func=OCV_func, eta_func=eta_func,
                      capacity=1.1,
                      SOC_0=0.38775,
                      E_R0=None, E_R1=None, T_amb=293.15)
solver = ECM.HybridDiscreteSolver(ECM_obj=ecm, isothermal=True, t_app=t[:t_lim_index], i_app=-I[:t_lim_index],
                                  SigmaX=SigmaX, SigmaW=SigmaW, SigmaV=SigmaV, V_actual=V[:t_lim_index],
                                  t_start=0, t_end=t[t_lim_index], t_interval=1)
sol = solver.solve()
sol.plot()

