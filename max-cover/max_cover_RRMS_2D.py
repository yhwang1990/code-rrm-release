import functools
import math
from queue import PriorityQueue

import numpy as np


@functools.total_ordering
class QItem:
    def __init__(self, item_id: int, gain: float, iteration: int):
        self.item_id = item_id
        self.gain = gain
        self.iteration = iteration

    def __lt__(self, obj):
        return self.gain > obj.gain

    def __eq__(self, obj):
        return self.gain == obj.gain

    def __ne__(self, obj):
        return self.gain != obj.gain

    def __gt__(self, obj):
        return self.gain < obj.gain


def GetWeightsForTwoDimension(k):
    gap = math.pi / (2.0 * (k + 1))
    theta = gap
    weights = []
    for i in range(k):
        weights.append([math.cos(theta), math.sin(theta)])
        theta += gap
    return weights


def Greedy(r: int, weights: np.ndarray, norm: np.ndarray, data: list[set[int]], attr: list[int], groups: list[set[int]]):
    sol = set()
    dim = len(groups)
    covs = [set() for _ in range(dim)]
    q = PriorityQueue()
    max_id = -1
    max_gain = 0
    for node_id in range(len(data)):
        gain = 0
        for j in range(dim):
            gain += weights[j] * len(data[node_id].intersection(groups[j]))
        if gain > 0:
            q.put(QItem(node_id, gain, 0))
        if gain > max_gain:
            max_id = node_id
            max_gain = gain
    sol.add(max_id)
    for u in data[max_id]:
        covs[attr[u]].add(u)

    for it in range(1, r):
        max_id = -1
        while not q.empty():
            q_item = q.get()
            if q_item.item_id not in sol and q_item.iteration < it:
                gains = [0] * dim
                for u in data[q_item.item_id]:
                    if u not in covs[attr[u]]:
                        gains[attr[u]] += 1
                gain = 0
                for j in range(dim):
                    gain += weights[j] * gains[j]
                if gain > 0:
                    q.put(QItem(q_item.item_id, gain, it))
            elif q_item.item_id not in sol:
                max_id = q_item.item_id
                break
        if max_id >= 0:
            sol.add(max_id)
            for u in data[max_id]:
                covs[attr[u]].add(u)
        else:
            break
    func_vals = np.array([(len(covs[j]) / norm[j]) for j in range(dim)])
    func_val = np.dot(weights, func_vals)
    return sol, func_val, func_vals


def CalcFuncVals(S: set[int], weights: np.ndarray, norm: np.ndarray, data, attr, groups):
    dim = len(groups)
    covs = [set() for _ in range(dim)]
    for v in S:
        for u in data[v]:
            covs[attr[u]].add(u)
    func_vals = np.array([(len(covs[j]) / norm[j]) for j in range(dim)])
    func_val = np.dot(weights, func_vals)
    return func_val


def BasisVectors(dim: int):
    basis_vectors = list()
    for i in range(dim):
        basis_vectors.append(np.zeros(dim))
        basis_vectors[i][i] = 1.0
    return basis_vectors


def DeltaNetValidation(dim, size):
    delta_net = list()
    np.random.seed(17)
    for i in range(size):
        weights = np.absolute(np.random.normal(0, 1, size=dim))
        weights = weights / np.linalg.norm(weights)
        delta_net.append(weights)
    return delta_net


def MaxRegretRatio(sol: list[set[int]], data, attr, groups, normVector, dim, r):
    if dim == 2:
        validation = DeltaNetValidation(dim, 100)
    else:
        validation = DeltaNetValidation(dim, 1000)
    mrr = 0
    for v in validation:
        _, func_val, _ = Greedy(r, v, normVector, data, attr, groups)
        max_val_sol = 0
        for s in sol:
            tempValue = CalcFuncVals(s, v, normVector, data, attr, groups)
            if max_val_sol < tempValue:
                max_val_sol = tempValue
        tempRatio = max(1.0 - 1.0 * max_val_sol / func_val, 0)
        if mrr < tempRatio:
            mrr = tempRatio
    return mrr


def RRMS(dim, k, r, data, attr, groups):
    weights = GetWeightsForTwoDimension(k)
    sol = []
    for weight in weights:
        S, _, _ = Greedy(r, np.array(weight), np.ones(dim), data, attr, groups)
        sol.append(S)
    return sol


def RRMS_Star(k, r, data, attr, groups):
    dim = 2
    if k < dim:
        return 1.0, list()
    sol = list()
    sol.extend(RRMS(dim, k, r, data, attr, groups))
    d = 0
    basis_vectors = BasisVectors(dim)
    normVector = np.zeros(dim)
    for weights in basis_vectors:
        _, func_val, _ = Greedy(r, weights, np.ones(dim), data, attr, groups)
        normVector[d] = float(func_val)
        d += 1
    mrr = MaxRegretRatio(sol, data, attr, groups, normVector, dim, r)
    return mrr, sol


def RRMS_Origin(k, r, data, attr, groups):
    dim = 2
    if k < dim:
        return 1.0, list()
    sol = list()
    basis_vectors = BasisVectors(dim)
    normVector = np.zeros(dim)
    d = 0
    for weights in basis_vectors:
        S, func_val, _ = Greedy(r, weights, np.ones(dim), data, attr, groups)
        sol.append(S)
        normVector[d] = float(func_val)
        d += 1
    sol.extend(RRMS(dim, k - dim, r, data, attr, groups))
    mrr = MaxRegretRatio(sol, data, attr, groups, normVector, dim, r)
    return mrr, sol


def SingleObj(r, data, attr, groups):
    dim = 2
    sol = list()
    basis_vectors = BasisVectors(dim)
    normVector = np.zeros(dim)
    d = 0
    for weights in basis_vectors:
        S, func_val, _ = Greedy(r, weights, np.ones(dim), data, attr, groups)
        sol.append(S)
        normVector[d] = float(func_val)
        d += 1
    mrr = MaxRegretRatio(sol, data, attr, groups, normVector, dim, r)
    return mrr, sol
