import numpy as np
import pandas as pd

import max_cover_HS_RRM
import max_cover_RRMS_4D
import max_cover_Polytope_4D


def readData(dataPath: str, weightPath: str, dim: int, is_directed: bool):
    dataFile = open(dataPath, 'r')
    maxIndex = -1
    for line in dataFile.readlines():
        items = line.split()
        if int(items[0]) > maxIndex:
            maxIndex = int(items[0])
        if int(items[1]) > maxIndex:
            maxIndex = int(items[1])
    nodeNum = maxIndex + 1
    dataFile.close()

    dataFile = open(dataPath)
    graph = list()
    for node_id in range(nodeNum):
        graph.append(set())
        graph[node_id].add(node_id)
    for line in dataFile.readlines():
        items = line.split()
        graph[int(items[0])].add(int(items[1]))
        if not is_directed:
            graph[int(items[1])].add(int(items[0]))
    dataFile.close()

    list_attr = []
    list_group = []
    for i in range(nodeNum):
        list_attr.append(-1)
    for j in range(dim):
        list_group.append(set())
    with open(weightPath, 'r') as weightFile:
        line_no = 0
        for line in weightFile:
            tokens = line.split()
            for token in tokens:
                item_id = int(token)
                list_attr[item_id] = line_no
                list_group[line_no].add(item_id)
            line_no += 1
    return graph, list_attr, list_group


if __name__ == "__main__":
    r = 10

    data, attr, groups = readData('./data/email-Eu-core.txt', './data/email-Eu-core_4.txt', 4, is_directed=True)
    rows_list = []
    ratio, _ = max_cover_RRMS_4D.SingleObj(r, data, attr, groups)
    rows_list.append({'data': 'email-Eu-core', 'alg': 'SingleObj', 'k': 4, 'd': 4, 'avg': ratio, 'std': 0, 'min': ratio, 'max': ratio})
    vals_k = list(range(1, 20))
    for ki in range(20, 51, 5):
        vals_k.append(ki)
    for k in vals_k:
        print('****************** k = %d *********************' % k)
        ratio, _ = max_cover_Polytope_4D.Do(k, r, data, attr, groups)
        rows_list.append({'data': 'email-Eu-core', 'alg': 'Polytope', 'k': k, 'd': 4, 'avg': ratio, 'std': 0, 'min': ratio, 'max': ratio})
        mrr_hs_rrm = np.zeros(10)
        mrr_rrms = np.zeros(10)
        mrr_rrms_star = np.zeros(10)
        for seed in range(10):
            print('****************** seed = %d *********************' % seed)
            ratio, _ = max_cover_HS_RRM.run(seed, k, r, data, attr, groups)
            mrr_hs_rrm[seed] = ratio
            ratio, _ = max_cover_RRMS_4D.RRMS_Origin(seed, k, r, data, attr, groups)
            mrr_rrms[seed] = ratio
            ratio, _ = max_cover_RRMS_4D.RRMS_Star(seed, k, r, data, attr, groups)
            mrr_rrms_star[seed] = ratio
        rows_list.append({'data': 'email-Eu-core', 'alg': 'HS-RRM', 'k': k, 'd': 4, 'avg': np.average(mrr_hs_rrm),
                          'std': np.std(mrr_hs_rrm), 'min': np.min(mrr_hs_rrm), 'max': np.max(mrr_hs_rrm)})
        rows_list.append({'data': 'email-Eu-core', 'alg': 'RRMS', 'k': k, 'd': 4, 'avg': np.average(mrr_rrms),
                          'std': np.std(mrr_rrms), 'min': np.min(mrr_rrms), 'max': np.max(mrr_rrms)})
        rows_list.append({'data': 'email-Eu-core', 'alg': 'RRMS-Star', 'k': k, 'd': 4, 'avg': np.average(mrr_rrms_star),
                          'std': np.std(mrr_rrms_star), 'min': np.min(mrr_rrms_star), 'max': np.max(mrr_rrms_star)})
    df = pd.DataFrame(rows_list)
    df.to_csv('./max_cover_d4.csv')
