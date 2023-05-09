"""
Copyright 2023 by Moin Ahmed. All Rights Reserved.

This example uses the dataset for the A123-LiFP cell found in CALCE public dataset (https://calce.umd.edu/battery-data).
Battery cell parameters:
    Capacity: 1.1 Ahr
    Ambient temperature: 20 degreesC
"""

import pandas as pd
import matplotlib.pyplot as plt

from ocv_soc_func import OCV_func, SOC_func
from src.model.ecm import Thevenin1RC
from src.solver.discrete_time_solver import DT_solver


def eta_func(i):
    if i <= 0:
        return 1
    else:
        return 0.9995


t_lim_index = 26030

df = pd.read_csv('../data/CALCE_A123/A1-A123-Dynamics.csv')
t = df['Test_Time(s)'].to_numpy()
I = df['Current(A)'].to_numpy()
V = df['Voltage(V)'].to_numpy()

# Rough parameter estimations just by eye-balling
ecm = Thevenin1RC(R0=0.225, R1=0.001, C1=0.03, OCV_func=OCV_func, eta_func=eta_func, capacity=1.1, SOC_0=0.38775,
                  E_R0=None, E_R1=None, T_amb=293.15)

# remember the current convention: positive for discharge and negative for charge
solver = DT_solver(ECM_obj=ecm, isothermal=True, t_app=t[:t_lim_index], i_app=-I[:t_lim_index])
z_pred, v_pred = solver.solve(verbose=True)

# Plots
fig = plt.figure()

ax1 = fig.add_subplot(3,1,1)
ax1.plot(t, V, label="exp.")
ax1.plot(t[:t_lim_index], v_pred, label="pred.")
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('V [V]')

ax2 = fig.add_subplot(3,1,2)
ax2.plot(t[:t_lim_index], z_pred, label="SOC pred.")
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('SOC')

ax3 = fig.add_subplot(3,1,3)
ax3.plot(t, I, label="I_exp")
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('I [A]')

plt.legend()
plt.tight_layout()
plt.show()