import max_cover_HS_RRM
import max_cover_RRMS_2D
import max_cover_Polytope_2D

import pandas as pd


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

    rows_list = []
    data, attr, groups = readData('./data/email-Eu-core.txt', './data/email-Eu-core_2.txt', 2, is_directed=True)
    ratio, _ = max_cover_RRMS_2D.SingleObj(r, data, attr, groups)
    rows_list.append({'data': 'email-Eu-core', 'alg': 'SingleObj', 'k': 2, 'd': 2, 'avg': ratio})
    vals_k = list(range(1, 21))
    for k in vals_k:
        print('****************** k = %d *********************' % k)
        ratio, _ = max_cover_Polytope_2D.Do(k, r, data, attr, groups)
        rows_list.append({'data': 'email-Eu-core', 'alg': 'Polytope', 'k': k, 'd': 2, 'avg': ratio})
        ratio, _ = max_cover_HS_RRM.run(0, k, r, data, attr, groups)
        rows_list.append({'data': 'email-Eu-core', 'alg': 'HS-RRM', 'k': k, 'd': 2, 'avg': ratio})
        ratio, _ = max_cover_RRMS_2D.RRMS_Origin(k, r, data, attr, groups)
        rows_list.append({'data': 'email-Eu-core', 'alg': 'RRMS', 'k': k, 'd': 2, 'avg': ratio})
        ratio, _ = max_cover_RRMS_2D.RRMS_Star(k, r, data, attr, groups)
        rows_list.append({'data': 'email-Eu-core', 'alg': 'RRMS-Star', 'k': k, 'd': 2, 'avg': ratio})

    df = pd.DataFrame(rows_list)
    df.to_csv('./max_cover_d2.csv')
