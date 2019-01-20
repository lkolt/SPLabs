import numpy as np


def dist2(x1, x2):
    return np.sum([v ** 2 for v in (x1.ravel() - x2.ravel())])


def DaviesBouldin(X, k, clusters, centroids):
    def precalc(i):
        cluster_points = [X[j] for j in range(X.shape[0]) if clusters[j] == i]
        return np.sqrt(np.sum([dist2(centroids[i], point) ** 2 for point in cluster_points]) / len(cluster_points))

    def db_pair(i, j):
        return (precalc(i) + precalc(j)) / dist2(centroids[i], centroids[j])

    def db_cluster(i):
        return np.max([db_pair(i, j) for j in range(k) if i != j])

    return np.mean([db_cluster(i) for i in range(k)])


def CalinskiHarabasz(X, k, clusters, centroids):
    def get_sw(i):
        return np.sum([dist2(point, centroids[i]) for point in cluster_points[i]])

    cluster_points = [[X[j] for j in range(X.shape[0]) if clusters[j] == i] for i in range(k)]
    mean = np.mean(X, axis=0)
    sw = np.sum([get_sw(i) for i in range(k)])
    sb = np.sum([(len(cluster_points[i]) * dist2(centroids[i], mean)) for i in range(k)])
    return ((len(X) - k) * sb) / ((k - 1) * sw)


def RandIndex(TP, FN, FP, TN):
    return (TP + TN) / (TP + TN + FN + FP)


def FowlkesMallowsIndex(TP, FN, FP, TN):
    return TP / np.sqrt((TP + FP)*(TP + FN))


def get_metric(X, k, clusters, centroids, name):
    if name == 'DaviesBouldin':
        return DaviesBouldin(X, k, clusters, centroids)
    if name == 'CalinskiHarabasz':
        return  CalinskiHarabasz(X, k, clusters, centroids)
    return 0