"""Softmax."""
# bigger values closer to 1, smaller values closer to 0; if scores*10, then 10 and 2 will closer to 0 than 1 and 0.2
scores = [3.0, 1.0, 0.2]

import numpy as np

def softmax(x):
    """Compute softmax values for x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
    #numpy.ones_like(x) return a ndarray with the same shape of x, all values are 1
    
plt.plot(x, softmax(scores).T, linewidth=2)
    #ndarray.T means transpose, Same as self.transpose(), except that self is returned if self.ndim < 2.
plt.show()
