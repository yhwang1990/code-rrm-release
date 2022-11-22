import functools
import math
from queue import PriorityQueue

import numpy as np
from scipy.spatial import ConvexHull


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


def Greedy(r: int, weights: np.ndarray, norm: np.ndarray, sim: np.ndarray, cluster: list[set[int]], sums: np.ndarray):
    sol = set()

    dim = len(cluster)
    n = len(sim)

    q = PriorityQueue()
    max_id = -1
    max_gain = 0
    for i in range(n):
        gain = 0
        for d in range(dim):
            gain += weights[d] * sums[i][d]
            gain -= weights[d] * sim[i][i]
        if gain > 0:
            q.put(QItem(i, gain, 0))
        if gain > max_gain:
            max_id = i
            max_gain = gain
    sol.add(max_id)

    for it in range(1, r):
        max_id = -1
        while not q.empty():
            q_item = q.get()
            if q_item.item_id not in sol and q_item.iteration < it:
                gains = [0] * dim
                for d in range(dim):
                    gains[d] += sums[q_item.item_id][d]
                div = sim[q_item.item_id][q_item.item_id]
                for j in sol:
                    div += 2 * sim[q_item.item_id][j]
                for d in range(dim):
                    gains[d] -= div
                gain = 0
                for d in range(dim):
                    gain += weights[d] * gains[d]
                if gain > 0:
                    q.put(QItem(q_item.item_id, gain, it))
            elif q_item.item_id not in sol:
                max_id = q_item.item_id
                break
        if max_id >= 0:
            sol.add(max_id)
        else:
            break
    sol_sum = [0] * dim
    for i in sol:
        for d in range(dim):
            sol_sum[d] += sums[i][d]
    div = 0
    for i in sol:
        for j in sol:
            div += sim[i][j]
    func_vals = np.array([((sol_sum[d] - div) / norm[d]) for d in range(dim)])
    func_val = np.dot(weights, func_vals)
    return sol, func_val, func_vals


def CalcFuncVals(S: set[int], weights: np.ndarray, norm: np.ndarray, sim: np.ndarray, cluster: list[set[int]], sums: np.ndarray):
    dim = len(cluster)
    sol_sum = [0] * dim
    for i in S:
        for d in range(dim):
            sol_sum[d] += sums[i][d]
    div = 0
    for i in S:
        for j in S:
            div += sim[i][j]
    func_vals = np.array([((sol_sum[d] - div) / norm[d]) for d in range(dim)])
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


def MaxRegretRatio(sol: list[set[int]], sim: np.ndarray, cluster: list[set[int]], sums: np.ndarray, normVector, dim, r):
    if dim == 2:
        validation = DeltaNetValidation(dim, 100)
    else:
        validation = DeltaNetValidation(dim, 1000)
    mrr = 0
    for v in validation:
        _, func_val, _ = Greedy(r, v, normVector, sim, cluster, sums)
        max_val_sol = 0
        for s in sol:
            tempValue = CalcFuncVals(s, v, normVector, sim, cluster, sums)
            if max_val_sol < tempValue:
                max_val_sol = tempValue
        tempRatio = max(1.0 - 1.0 * max_val_sol / func_val, 0)
        if mrr < tempRatio:
            mrr = tempRatio
    return mrr


def PS(pointList):
    newPoints = []
    for point in pointList:
        newPoints.append([point[0], point[1], point[2], 0.0])
        newPoints.append([point[0], point[1], 0.0, point[3]])
        newPoints.append([point[0], 0.0, point[2], point[3]])
        newPoints.append([0.0, point[1], point[2], point[3]])
        newPoints.append([point[0], 0.0, 0.0, 0.0])
        newPoints.append([0.0, point[1], 0.0, 0.0])
        newPoints.append([0.0, 0.0, point[2], 0.0])
        newPoints.append([0.0, 0.0, 0.0, point[3]])
    newPoints.append([0, 0, 0, 0])
    newPointsList = pointList + newPoints
    points = np.mat(newPointsList)
    hull = ConvexHull(points)
    b = 10000000 * np.ones((4, 1))
    nonNegativeNormVectors = []
    for simplex in hull.simplices:
        tempMatrix = points[simplex, :]
        if math.fabs(np.linalg.det(tempMatrix)) > 0.000001:
            vector = np.linalg.solve(tempMatrix, b)
            if np.sum(vector <= 0) == 4:
                vector = 0.0 - vector
            if np.sum(vector < 0) == 0:
                normValue = np.linalg.norm(vector)
                fvector = vector / normValue
                isSame = False
                for item in nonNegativeNormVectors:
                    if np.sum(abs(item - fvector)) < 0.000001:
                        isSame = True
                        break
                if abs(1.0 - np.sum(abs(fvector))) < 0.000001:
                    isSame = True
                if not isSame:
                    nonNegativeNormVectors.append(fvector)
    return nonNegativeNormVectors


def Polytope(dim, k, r, base, base_func_vals, sim: np.ndarray, cluster: list[set[int]], sums: np.ndarray):
    sol = list()
    sol.extend(base)
    sol_space_points = list()
    sol_space_points.extend(base_func_vals)
    while len(sol) < k:
        nonNegativeNormVectors = PS(sol_space_points)
        for normVec in nonNegativeNormVectors:
            weight = np.zeros(dim)
            for d in range(dim):
                weight[d] = normVec[d][0]
            S, func_val, func_vals = Greedy(r, weight, np.ones(dim), sim, cluster, sums)
            sol.append(S)
            sol_space_points.append(list(func_vals))
            if len(sol) == k:
                return sol
    return sol


def Do(k, r, sim: np.ndarray, cluster: list[set[int]], sums: np.ndarray):
    dim = 4
    if k < dim:
        return 1.0, list()
    base = list()
    base_func_vals = list()
    basis_vectors = BasisVectors(dim)
    normVector = np.zeros(dim)
    d = 0
    for weights in basis_vectors:
        S, func_val, func_vals = Greedy(r, weights, np.ones(dim), sim, cluster, sums)
        base.append(S)
        base_func_vals.append(list(func_vals))
        normVector[d] = float(func_val)
        d += 1
    sol = list()
    sol.extend(Polytope(dim, k, r, base, base_func_vals, sim, cluster, sums))
    mrr = MaxRegretRatio(sol, sim, cluster, sums, normVector, dim, r)
    return mrr, sol
