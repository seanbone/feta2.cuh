#!/usr/bin/env python3

import sys
import re
import json

# A simple script to parse the output of the benchmark runs.
# Reads the output of a benchmark run from stdin, and prints a JSON
#  representation of the relevant information.
# Is it silly to parse the raw GTest output? Probably, but I'm doing it anyway.

# Example usage:
# ./build/tests/benchmarks/feta2_benchmarks | ./scripts/parseBench.py

reFlags = re.MULTILINE | re.DOTALL


def parseSuite(suiteContent):
    """ Parse the output of one benchmark suite, returning a dictionary """
    implsRegex = r'\[\s+RUN\s+\] (?P<tag>(?P<suite>\w+)\/(?P<run>\d+)\.(?P<impl>\w+))'\
        r'(?:.+)Precision: (?P<prec>\w+)(?:.+)'\
        r'nSamples: (?P<nSamples>\S+)(?:.+)'\
        r'nReps: (?P<nReps>\S+)(?:.+)'\
        r'GFLOP/s: (?P<gflops>\d+\.\d+)(?:.+)'\
        r'\[\s+OK\s+\] (?P=tag)'

    implsData = {}
    for implMatch in re.finditer(implsRegex, suiteContent, reFlags):
        impl = implMatch.group("impl")
        if impl not in implsData.keys():
            implsData[impl] = {}

        implsData[impl]['gflops'] = float(implMatch.group('gflops'))
        implsData[impl]['nReps'] = float(implMatch.group('nReps'))
        implsData[impl]['nSamples'] = float(implMatch.group('nSamples'))
        implsData[impl]['prec'] = implMatch.group('prec')

    return implsData


def parseBenchRun(allInput):
    """ Parses the output of an entire benchmark run,
        returning a dictionary with the important data.
    """
    suiteRegex = r'(?P<tag>\[(-)+\] (?:\d+) tests from '\
        r'(?P<suite>\w+)\/(?P<run>\d+)), where TypeParam = '\
        r'feta2_bench::VecDims<(?P<nDims>\d+)>'\
        r'(?P<content>.+)'\
        r'(?P=tag)'

    benchData = {}

    for suiteMatch in re.finditer(suiteRegex, allInput, reFlags):
        suite = suiteMatch.group("suite")
        if suite not in benchData.keys():
            benchData[suite] = {}

        nDims = suiteMatch.group("nDims")
        benchData[suite][f"VecDims={nDims}"] = parseSuite(
            suiteMatch.group("content"))

    return benchData


if __name__ == "__main__":
    allInput = sys.stdin.read()
    benchData = parseBenchRun(allInput)
    print(json.dumps(benchData, indent=4))
