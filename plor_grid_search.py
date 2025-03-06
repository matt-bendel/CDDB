import matplotlib.pyplot as plt
import numpy as np

# Baseline Results...
# FID: 84.54
# PSNR: 24.22
# LPIPS: 0.2541

psnrs = [24.25, 24.24, 24.19, 23.78, 21.33]
fids = [84.45, 84.31, 84.53, 85.65, 97.13]
lpipss = [0.2527, 0.2530, 0.2537, 0.2624, 0.3462]

x_axis = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

fig, ax1 = plt.subplots()

color = 'tab:green'
ax1.set_xlabel('$\lambda$')
ax1.set_ylabel('FID', color=color)
ax1.semilogx(x_axis, fids, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('PSNR', color=color)  # we already handled the x-label with ax1
ax2.semilogx(x_axis, psnrs, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('grid_search_results.png')
plt.close(fig)
