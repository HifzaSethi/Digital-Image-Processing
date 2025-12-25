import numpy as np
import matplotlib.pyplot as plt

# Given points
L = 256
r1, s1 = 70, 0
r2, s2 = 140, 255

# Create piecewise transformation function
r = np.arange(0, 256)
s = np.piecewise(
    r,
    [r < r1, (r >= r1) & (r < r2), r >= r2],
    [
        lambda r: (s1 / r1) * r,  # region 1
        lambda r: ((s2 - s1) / (r2 - r1)) * (r - r1) + s1,  # region 2
        lambda r: ((L - 1 - s2) / (L - 1 - r2)) * (r - r2) + s2  # region 3
    ]
)

plt.figure(figsize=(6,5))
plt.plot(r, s, color='blue')
plt.title('Contrast Stretching Transfer Function')
plt.xlabel('Input Intensity (r)')
plt.ylabel('Output Intensity (s)')
plt.grid(True)
plt.show()
