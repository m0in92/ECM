import numpy as np
import matplotlib.pyplot as plt

from parameter_sets.test.funcs import *


array_soc = np.linspace(0.05, 0.95)
array_ocv = func_OCV(array_soc)

plt.plot(soc_discharge, V_discharge, label="Discharge")
plt.plot(soc_charge, V_charge, label="Charge")
plt.plot(array_soc, array_ocv, label="OCV")
plt.legend()
plt.show()
