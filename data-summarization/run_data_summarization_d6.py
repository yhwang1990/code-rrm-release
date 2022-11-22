import numpy as np
import pandas as pd

import data_summarization_HS_RRM
import data_summarization_RRMS_6D
import data_summarization_Polytope_6D


def readData(simPath, clusterPath, dim):
    simFile = open(simPath, 'r')
    num_items = len(simFile.readlines())
    simFile.close()
    simFile = open(simPath, 'r')
    sim_mat = np.zeros((num_items, num_items))
    a = 0
    for line in simFile.readlines():
        items = line.split()
        b = 0
        for item in items:
            sim_mat[a][b] = float(item)
            b += 1
        a += 1
    simFile.close()
    clusterFile = open(clusterPath, 'r')
    list_groups = []
    for d in range(dim):
        list_groups.append(set())
    d = 0
    for line in clusterFile.readlines():
        items = line.split()
        for item in items:
            list_groups[d].add(int(item))
        d += 1
        if d == dim:
            break
    clusterFile.close()
    sum_vals = np.zeros((num_items, dim))
    for i in range(num_items):
        for d in range(dim):
            for j in list_groups[d]:
                sum_vals[i][d] += sim_mat[i][j]
    return sim_mat, list_groups, sum_vals


if __name__ == "__main__":
    r = 10

    rows_list = []
    sim, clusters, sums = readData('./data/MovieLens.txt', './data/MovieLens_6.txt', 6)
    ratio, _ = data_summarization_RRMS_6D.SingleObj(r, sim, clusters, sums)
    rows_list.append({'data': 'MovieLens', 'alg': 'SingleObj', 'k': 6, 'd': 6, 'avg': ratio, 'std': 0, 'min': ratio, 'max': ratio})
    vals_k = list(range(1, 20))
    for ki in range(20, 51, 5):
        vals_k.append(ki)
    for k in vals_k:
        print('****************** k = %d *********************' % k)
        ratio, _ = data_summarization_Polytope_6D.Do(k, r, sim, clusters, sums)
        rows_list.append({'data': 'MovieLens', 'alg': 'Polytope', 'k': k, 'd': 6, 'avg': ratio, 'std': 0, 'min': ratio, 'max': ratio})
        mrr_hs_rrm = np.zeros(10)
        mrr_rrms = np.zeros(10)
        mrr_rrms_star = np.zeros(10)
        for seed in range(10):
            print('****************** seed = %d *********************' % seed)
            ratio, _ = data_summarization_HS_RRM.run(seed, k, r, sim, clusters, sums)
            mrr_hs_rrm[seed] = ratio
            ratio, _ = data_summarization_RRMS_6D.RRMS_Origin(seed, k, r, sim, clusters, sums)
            mrr_rrms[seed] = ratio
            ratio, _ = data_summarization_RRMS_6D.RRMS_Star(seed, k, r, sim, clusters, sums)
            mrr_rrms_star[seed] = ratio
        rows_list.append({'data': 'MovieLens', 'alg': 'HS-RRM', 'k': k, 'd': 6, 'avg': np.average(mrr_hs_rrm),
                          'std': np.std(mrr_hs_rrm), 'min': np.min(mrr_hs_rrm), 'max': np.max(mrr_hs_rrm)})
        rows_list.append({'data': 'MovieLens', 'alg': 'RRMS', 'k': k, 'd': 6, 'avg': np.average(mrr_rrms),
                          'std': np.std(mrr_rrms), 'min': np.min(mrr_rrms), 'max': np.max(mrr_rrms)})
        rows_list.append({'data': 'MovieLens', 'alg': 'RRMS-Star', 'k': k, 'd': 6, 'avg': np.average(mrr_rrms_star),
                          'std': np.std(mrr_rrms_star), 'min': np.min(mrr_rrms_star), 'max': np.max(mrr_rrms_star)})
    df = pd.DataFrame(rows_list)
    df.to_csv('./data_summarization_d6.csv')
