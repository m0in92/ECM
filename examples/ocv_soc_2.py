import numpy as np

from ocv_soc_func import *
import matplotlib.pyplot as plt


soc = np.linspace(0.05,0.95)
ocv = OCV_func(soc)

plt.plot(soc_discharge, V_discharge, label="discharge")
plt.plot(soc_charge, V_charge, label="charge")
plt.plot(soc, ocv, label='avg.')
plt.xlabel('SOC')
plt.ylabel('OCV [V]')
plt.legend()
plt.show()