"""
Copyright 2023 by Moin Ahmed. All Rights Reserved.

This example uses the dataset for the A123-LiFP cell found in CALCE public dataset (https://calce.umd.edu/battery-data).
Furthermore, Sigma-point Kalman filter is used at time-periods where experimental data is available.

Battery cell parameters:
    Capacity: 1.1 Ahr
    Ambient temperature: 20 degreesC
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ocv_soc_func import OCV_func, SOC_func
import ECM


def eta_func(i):
    return 1 if i<=0 else 0.9995


t_lim_index = 26000

# Read experimental data
df = pd.read_csv('../data/CALCE_A123/A1-A123-Dynamics.csv')
t = df['Test_Time(s)'].to_numpy()
I = df['Current(A)'].to_numpy()
V = df['Voltage(V)'].to_numpy()

# Rough parameter estimations just by eye-balling
ecm = ECM.Thevenin1RC(R0=0.225, R1=0.001, C1=0.03, OCV_func=OCV_func, eta_func=eta_func, capacity=1.1, SOC_0=0.417,
                      E_R0=None, E_R1=None, T_amb=293.15)

# remember the current convention: positive for discharge and negative for charge
SigmaX = np.array([[1e-8, 0],[0, 1e-8]])
SigmaW, SigmaV = 1e-8, 1e-8
solver = ECM.HybridDiscreteSolver(ECM_obj=ecm, isothermal=True, t_app=t[:t_lim_index], i_app=-I[:t_lim_index],
                                  SigmaX=SigmaX, SigmaW=SigmaW, SigmaV=SigmaV, V_actual=V[:t_lim_index],
                                  t_start=0, t_end=40000, t_interval=1)
z_pred, v_pred = solver.solve()

# Plots
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)

ax1.plot(t, V, label="exp.")
ax1.plot(solver.t_array[1:], v_pred, label="pred.")
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('V [V]')
ax1.legend(loc='right')

ax2.plot(solver.t_array[1:], z_pred, label="SOC pred.")
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('SOC')

ax3.plot(t, I, label="I_exp")
ax3.plot(solver.t_array, -solver.I_array, label="I_sim")
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('I [A]')

plt.tight_layout()
plt.show()