#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys
import json
import argparse

# A simple script to visualise benchmark results.
# The input is read from STDIN and is expected to be JSON as
# is output by `parseBench.py`


def makePlot(suiteName, nSamples, prec, runs, flops):
    nRuns = len(runs)
    nBars = len(flops)
    x = np.arange(nRuns)  # the label locations
    width = 1./(nBars+1)  # 0.3  # the width of the bars
    multiplier = 0
    ymax = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurements in flops.items():
        offset = width * multiplier
        measurements = np.array([round(i) for i in measurements])
        ymax = max(ymax, np.max(measurements))
        rects = ax.bar(x + offset, measurements, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Average performance [GFLOP/s]')
    ax.set_title(f'{suiteName} ({nSamples/1e6}M samples, {prec} precision)')
    ax.set_xticks(x + width * (nBars-1.)/2, runs)
    ax.legend(loc='upper left', ncols=2)
    ax.set_ylim(0, ymax*1.2)

    plt.show()


def checkEq(expected, actual):
    if expected == None:
        return actual
    else:
        if actual == expected:
            return actual
        else:
            raise ValueError(f"Inconsistent metadata within a bench suite! {
                             actual} != {expected}")


def plotBenchSuite(suiteName, suiteData, exclude):
    nReps, nSamples, prec = None, None, None
    runs = []
    flops = {}
    for runName in suiteData.keys():
        runs.append(runName)
        for seriesName in suiteData[runName].keys():
            if seriesName in exclude:
                continue

            seriesData = suiteData[runName][seriesName]

            nReps = checkEq(nReps, seriesData['nReps'])
            nSamples = checkEq(nSamples, seriesData['nSamples'])
            prec = checkEq(prec, seriesData['prec'])

            if seriesName not in flops.keys():
                flops[seriesName] = []
            flops[seriesName].append(seriesData['gflops'])

    print(f'Plotting benchmark suite {suiteName}')
    makePlot(suiteName, nSamples, prec, runs, flops)


def flatten(array2d):
    if array2d is None:
        return []

    out = []
    for item in array2d:
        out.extend(item)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='plotBench',
                                     description='Plot benchmark results.      \
            Values are read from STDIN as a JSON string in the format returned \
                                      by `parseBench.py`.')

    parser.add_argument('-e', '--exclude', action='append', nargs='+',
                        help="Exclude one or more data series from the plot.")
    args = parser.parse_args()

    exclude = flatten(args.exclude)

    allInput = sys.stdin.read()
    benchData = json.loads(allInput)
    for benchSuite in benchData.keys():
        plotBenchSuite(benchSuite, benchData[benchSuite], exclude)
