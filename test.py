import numpy as np
m = np.array([[0,1],[2,0]])
m[m==0] = 3
print(m)