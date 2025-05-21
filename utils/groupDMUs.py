import numpy as np
from collections import defaultdict


def groupDMUs(y):
    n, s = y.shape
    dmu_outputs = {}
    profile_to_dmus = defaultdict(set)
    profile_to_outputs = defaultdict(set)
    correspondingR = {}

    for dmu in range(n):
        produced_outputs = {output for output in range(s) if y[dmu, output] != 0}
        dmu_outputs[dmu] = produced_outputs
        profile_to_dmus[frozenset(produced_outputs)].add(dmu)

    N = list(profile_to_dmus.values())  

    for output in range(s):
        producing_dmus = {dmu for dmu in range(n) if y[dmu, output] != 0}
        profile_to_outputs[frozenset(producing_dmus)].add(output)

    R = list(profile_to_outputs.values())  

    for i, output_set in enumerate(R):
        for output in output_set:
            correspondingR[output] = i

    L = {}
    dmu_to_subgroup = {}

    for i, dmu_group in enumerate(N):
        L[i] = set()  
        for dmu in dmu_group:

            L[i].update(correspondingR[output] for output in dmu_outputs[dmu])
            dmu_to_subgroup[dmu] = i 
            
    M = [set() for _ in range(len(R))]
    for dmu, outputs in dmu_outputs.items():
        for output in outputs:
            k = correspondingR[output]
            M[k].add(dmu)

    return N, R, dmu_outputs, correspondingR, L, dmu_to_subgroup, M
