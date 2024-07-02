import pandas as pd

# import numpy as np
import numpy as np

# simple array
data = np.array([-1, -1, -1, 1, 1, 1,1])
data2 = np.array(['u', 'u', 'u', 'u', 'd', 'd','d','d'])

ser = pd.Series(data2)
print(ser)

print(ser.describe())