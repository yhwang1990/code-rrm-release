import numpy as np
import pandas as pd

import data_summarization_HS_RRM
import data_summarization_RRMS_2D
import data_summarization_Polytope_2D


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
    sim, clusters, sums = readData('./data/MovieLens.txt', './data/MovieLens_2.txt', 2)
    ratio, _ = data_summarization_RRMS_2D.SingleObj(r, sim, clusters, sums)
    rows_list.append({'data': 'MovieLens', 'alg': 'SingleObj', 'k': 2, 'd': 2, 'avg': ratio})
    vals_k = list(range(1, 21))
    for k in vals_k:
        print('****************** k = %d *********************' % k)
        ratio, _ = data_summarization_Polytope_2D.Do(k, r, sim, clusters, sums)
        rows_list.append({'data': 'MovieLens', 'alg': 'Polytope', 'k': k, 'd': 2, 'avg': ratio})
        ratio, _ = data_summarization_HS_RRM.run(0, k, r, sim, clusters, sums)
        rows_list.append({'data': 'MovieLens', 'alg': 'HS-RRM', 'k': k, 'd': 2, 'avg': ratio})
        ratio, _ = data_summarization_RRMS_2D.RRMS_Origin(k, r, sim, clusters, sums)
        rows_list.append({'data': 'MovieLens', 'alg': 'RRMS', 'k': k, 'd': 2, 'avg': ratio})
        ratio, _ = data_summarization_RRMS_2D.RRMS_Star(k, r, sim, clusters, sums)
        rows_list.append({'data': 'MovieLens', 'alg': 'RRMS-Star', 'k': k, 'd': 2, 'avg': ratio})

    df = pd.DataFrame(rows_list)
    df.to_csv('./data_summarization_d2.csv')
