
import numpy as np
from scipy.stats import norm

xgrid = np.linspace(.01,.99,1000)
qq = norm.ppf(xgrid)

import matplotlib.pyplot as plt
plt.plot(xgrid, qq)
plt.show()
