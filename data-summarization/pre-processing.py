import numpy as np
from sklearn.cluster import KMeans

fileVectors = open('./data/item-vectors.txt', 'r')
list_vectors = list()
for line in fileVectors:
    tokens = line.split()
    vec = [float(tokens[i]) for i in range(2, 27)]
    list_vectors.append(vec)
fileVectors.close()

S = np.zeros((len(list_vectors), len(list_vectors)))
for i in range(len(list_vectors)):
    for j in range(len(list_vectors)):
        S[i][j] = max((np.dot(list_vectors[i], list_vectors[j]) / (
                    np.linalg.norm(list_vectors[i]) * np.linalg.norm(list_vectors[j]))), 0.0)

np.savetxt(fname='./data/MovieLens.txt', X=S, fmt='%.5f', delimiter=' ', newline='\n')

fileVectors = open('./data/item-vectors.txt', 'r')
list_vectors = list()
for line in fileVectors:
    tokens = line.split()
    vec = [float(tokens[i]) for i in range(2, 27)]
    list_vectors.append(vec)
fileVectors.close()

X = np.array(list_vectors)
for i in range(len(X)):
    X[i] /= np.linalg.norm(X[i])

for c in range(2, 8):
    kmeans = KMeans(n_clusters=c, random_state=0).fit(X)
    clusters = []
    for i in range(c):
        clusters.append(list())
    item_id = 0
    for ll in kmeans.labels_:
        clusters[ll].append(item_id)
        item_id += 1
    fileCluster = open('./data/MovieLens_' + str(c) + '.txt', 'w')
    for i in range(c):
        for item_id in clusters[i]:
            fileCluster.write(str(item_id) + ' ')
        fileCluster.write('\n')
    fileCluster.close()
