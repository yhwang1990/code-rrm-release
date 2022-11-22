import functools
import math
import numpy as np
from queue import PriorityQueue


class Solution:
    def __init__(self, weights: np.ndarray, sol: set[int], func_val: float, func_vec: np.ndarray):
        self.weights = weights
        self.sol = sol
        self.func_val = func_val
        self.func_vec = func_vec


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


def greedy(r: int, weights: np.ndarray, norm: np.ndarray, data: list[set[int]], attr: list[int], groups: list[set[int]]):
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


def basisVectors(dim: int):
    basis_vectors = list()
    for i in range(dim):
        basis_vectors.append(np.zeros(dim))
        basis_vectors[i][i] = 1.0
    return basis_vectors


def deltaNet2D(dim, basis_vectors):
    delta_net = list()
    delta_net.append(basis_vectors[0])
    size = 50 - dim
    theta = math.pi / (2.0 * (size + 1))
    step = theta
    for i in range(size):
        delta_net.append(np.array([math.cos(theta), math.sin(theta)]))
        theta += step
    delta_net.append(basis_vectors[1])
    return delta_net


def deltaNetmD(seed, dim, basis_vectors):
    delta_net = list()
    delta_net.extend(basis_vectors)
    sizes = {3: 500, 4: 1000, 5: 2000, 6: 4000, 7: 8000}
    np.random.seed(seed)
    for i in range(sizes[dim] - dim):
        weights = np.absolute(np.random.normal(0, 1, size=dim))
        weights = weights / np.linalg.norm(weights)
        delta_net.append(weights)
    return delta_net


def epsKernel(seed: int, k: int, dim: int, allSolutions: list[Solution]):
    S1 = list()
    scale = (1 + math.sqrt(dim)) / (1 - 1 / math.exp(1))
    points = list()
    if dim == 2:
        if k == 1:
            points.append(np.array([math.cos(math.pi / 4.0), math.sin(math.pi / 4.0)]) * scale)
        elif k == 2:
            points.append(np.array([math.cos(0), math.sin(0)]) * scale)
            points.append(np.array([math.cos(math.pi / 2.0), math.sin(math.pi / 2.0)]) * scale)
        else:
            points.append(np.array([math.cos(0), math.sin(0)]) * scale)
            theta = math.pi / (2.0 * (k - 1))
            step = math.pi / (2.0 * (k - 1))
            for i in range(1, k - 1):
                p = np.array([math.cos(theta), math.sin(theta)]) * scale
                points.append(p)
                theta += step
            points.append(np.array([math.cos(math.pi / 2.0), math.sin(math.pi / 2.0)]) * scale)
    else:
        np.random.seed(seed)
        for i in range(k):
            p = np.absolute(np.random.normal(0, 1, size=dim))
            p = p / np.linalg.norm(p) * scale
            points.append(p)
    for p in points:
        min_idx = -1
        min_dist = float('inf')
        for i in range(len(allSolutions)):
            dist = np.linalg.norm(p - allSolutions[i].func_vec)
            if dist < min_dist:
                min_idx = i
                min_dist = dist
        S1.append(allSolutions[min_idx])
    return S1


def greedyHS2D(covs, n):
    idxs = []
    end = 0
    while end < n - 1:
        max_id = -1
        max_end = -1
        for i in range(len(covs)):
            if covs[i][0] <= end + 1:
                new_end = covs[i][-1]
                if new_end > max_end:
                    max_id = i
                    max_end = new_end
        idxs.append(max_id)
        end = covs[max_id][-1]
    return idxs


def greedyHS(covs, n):
    uncovered = set(range(n))
    idxs = set()
    max_id = -1
    max_cov = 0
    for i in range(len(covs)):
        if len(covs[i]) > max_cov:
            max_id = i
            max_cov = len(covs[i])
    idxs.add(max_id)
    uncovered.difference_update(covs[max_id])
    while len(uncovered) > 0:
        max_id = -1
        max_cov = 0
        for i in range(len(covs)):
            if i not in idxs:
                cov_i = 0
                for j in covs[i]:
                    if j in uncovered:
                        cov_i += 1
                if cov_i > max_cov:
                    max_id = i
                    max_cov = cov_i
        idxs.add(max_id)
        uncovered.difference_update(covs[max_id])
    return idxs


def hittingSetRRM(k, dim, allS):
    gap = 0.001
    tau_max = 1.0
    tau_min = 0
    cur_idxs = set()
    while tau_max - tau_min > gap:
        tau = (tau_max + tau_min) / 2.0
        covs = list()
        for i in range(len(allS)):
            covs.append(list())
        for i in range(len(allS)):
            for j in range(len(allS)):
                if np.dot(allS[i].func_vec, allS[j].weights) > tau * allS[j].func_val:
                    covs[i].append(j)
        if dim == 2:
            idxs = greedyHS2D(covs, len(allS))
        else:
            idxs = greedyHS(covs, len(allS))
        if len(idxs) <= k:
            tau_min = tau
            cur_idxs = idxs
        else:
            tau_max = tau
    S2 = list()
    for idx in cur_idxs:
        S2.append(allS[idx])
    return S2


def deltaNetValidation(dim, size):
    delta_net = list()
    np.random.seed(17)
    for i in range(size):
        weights = np.absolute(np.random.normal(0, 1, size=dim))
        weights = weights / np.linalg.norm(weights)
        delta_net.append(weights)
    return delta_net


def maxRegretRatio(sol: list[Solution], data, attr, groups, normVector, dim, r):
    if dim == 2:
        validation = deltaNetValidation(dim, 100)
    else:
        validation = deltaNetValidation(dim, 1000)
    mrr = 0
    for v in validation:
        _, func_val, _ = greedy(r, v, normVector, data, attr, groups)
        max_val_sol = 0
        for s in sol:
            tempValue = np.dot(v, s.func_vec)
            if max_val_sol < tempValue:
                max_val_sol = tempValue
        tempRatio = max(1.0 - 1.0 * max_val_sol / func_val, 0)
        if mrr < tempRatio:
            mrr = tempRatio
    return mrr


def run(seed: int, k: int, r: int, data: list[set[int]], attr: list[int], groups: list[set[int]]):
    dim = len(groups)
    allSolutions = list()

    basis_vectors = basisVectors(dim)
    normVector = np.zeros(dim)
    d = 0
    for weights in basis_vectors:
        _, func_val, _ = greedy(r, weights, np.ones(dim), data, attr, groups)
        normVector[d] = float(func_val)
        d += 1
    if dim == 2:
        delta_net = deltaNet2D(dim, basis_vectors)
    else:
        delta_net = deltaNetmD(seed, dim, basis_vectors)
    for weights in delta_net:
        S, func_val, func_vals = greedy(r, weights, normVector, data, attr, groups)
        allSolutions.append(Solution(weights=weights, sol=S, func_val=float(func_val), func_vec=func_vals))
    S1 = epsKernel(seed, k, dim, allSolutions)
    S2 = hittingSetRRM(k, dim, allSolutions)
    P1 = list()
    for s in S1:
        P1.append(s.func_vec)
    mrr1 = maxRegretRatio(S1, data, attr, groups, normVector, dim, r)
    P2 = list()
    for s in S2:
        P2.append(s.func_vec)
    mrr2 = maxRegretRatio(S2, data, attr, groups, normVector, dim, r)
    if mrr1 < mrr2:
        S = [S1[i].sol for i in range(len(S1))]
    else:
        S = [S2[i].sol for i in range(len(S2))]
    return min(mrr1, mrr2), S
