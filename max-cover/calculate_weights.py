import networkx as nx
from networkx.algorithms.community import asyn_fluidc

dataFile = open('./data/email-Eu-core.txt')
maxIndex = -1
minIndex = 100000000
for line in dataFile.readlines():
    items = line.split()
    if int(items[0]) > maxIndex:
        maxIndex = int(items[0])
    if int(items[0]) < minIndex:
        minIndex = int(items[0])
    if int(items[1]) > maxIndex:
        maxIndex = int(items[1])
    if int(items[1]) < minIndex:
        minIndex = int(items[1])
nodeNum = maxIndex - minIndex + 1
dataFile.close()

dataFile = open('./data/email-Eu-core.txt')
G = nx.Graph()
G.add_nodes_from(range(maxIndex - minIndex))
cnt = 0
for line in dataFile.readlines():
    items = line.split()
    if int(items[0]) != int(items[1]):
        G.add_edge(int(items[0]) - minIndex, int(items[1]) - minIndex)
    else:
        if int(items[0]) - minIndex != cnt:
            G.add_edge(int(items[0]) - minIndex, cnt)
            cnt += 1
        else:
            cnt += 1
            G.add_edge(int(items[0]) - minIndex, cnt)
            cnt += 1
dataFile.close()

for num_comm in range(2, 8):
    C = asyn_fluidc(G, k=num_comm)
    commFile = open('./data/email-Eu-core_' + str(num_comm) + '.txt', 'w')
    c_id = 0
    for comm in C:
        for i in comm:
            commFile.write(str(i) + ' ')
        commFile.write('\n')
        c_id += 1
    commFile.close()
