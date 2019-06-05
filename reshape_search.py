import numpy as np


a = np.arange(16.).reshape(4, 4)
b = np.reshape(a, (2, 2, 2, 2))
c = np.transpose(b, [0, 2, 1, 3])
d = np.reshape(c, (4, 4))

print(a)
print(d)
