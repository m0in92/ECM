"""
Copyright 2023 by Moin Ahmed. All Rights Reserved.

This example uses the dataset for the A123-LiFP cell found in CALCE public dataset (https://calce.umd.edu/battery-data).
Battery cell parameters:
    Capacity: 1.1 Ahr
    Ambient temperature: 20 degreesC
"""

import pandas as pd # pandas package is imported for reading datasets.

import ECM


# Read experimental data
t_lim_index = 26030
df = pd.read_csv('../data/CALCE_A123/A1-A123-Dynamics.csv')
t = df['Test_Time(s)'].to_numpy()
I = df['Current(A)'].to_numpy()
V = df['Voltage(V)'].to_numpy()

# Create an Thevenin1RC object
ecm = ECM.Thevenin1RC(param_set_name='Calce_A123', SOC_init=0.38775, T_amb=293.15)
# Create a solver object and then call the solve method.
## remember the current convention: positive for discharge and negative for charge.
solver = ECM.DTSolver(ECM_obj=ecm, isothermal=True, t_app=t[:t_lim_index], i_app=-I[:t_lim_index], V_actual=V[:t_lim_index])

sol = solver.solve(verbose=False)

# Plots
sol.plot()