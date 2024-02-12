#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# Benchmark results with 10M samples, 100k reps, on a Tesla V100
# Values are measured GFLOP/s
runs = ("Dims = 3", "Dims = 6", "Dims = 9")
flops = {
    'Eigen (naive)': [674.165, 492.485, 59.0964],
    # 'Manual (naive)': [672.826, 492.395, 78.0256],
    'FETA2': [924.304, 1233.33, 1253.7],
    'Manual': [915.05, 1246.92, 1284.53],
}

nRuns = len(runs)
nBars = len(flops)
x = np.arange(nRuns)  # the label locations
width = 0.3  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in flops.items():
    offset = width * multiplier
    measurement = [round(i) for i in measurement]
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Average performance [GFLOP/s]')
ax.set_title('Vector dot product (10M samples, double precision)')
ax.set_xticks(x + width * (nBars-1.)/2, runs)
ax.legend(loc='upper left', ncols=4)
ax.set_ylim(0, 1450)

plt.show()
