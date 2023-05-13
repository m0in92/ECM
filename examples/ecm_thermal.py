import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ECM
from data.Synthetic.funcs import *


# Read experimental data
df = pd.read_csv('../data/Synthetic/test.csv')
t, I, V, T = df['Time [s]'], df['I [A]'], df['V [V]'], df['Temp [K]']

# Create an Thevenin1RC object
ecm = ECM.Thevenin1RC(R0=0.225, R1=0.001, C1=0.03, OCV_func=OCV_func, eta_func=eta_func, capacity=1.1, SOC_0=0.38775,
                      E_R0=None, E_R1=None, T_amb=293.15)
# Create a solver object and then call the solve method.
## remember the current convention: positive for discharge and negative for charge.
solver = ECM.DTSolver(ECM_obj=ecm, isothermal=True, t_app=t[:t_lim_index], i_app=-I[:t_lim_index], V_actual=V[:t_lim_index])



# Plots
# plt.plot(t, I)
# plt.plot(t, T)
plt.plot(t, V)
plt.show()