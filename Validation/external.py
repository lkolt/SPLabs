import numpy as np
import matplotlib.pyplot as plt
from Validation.metrics import RandIndex, FowlkesMallowsIndex
from KMeans.main import k_means

# Consts
_clusters = np.arange(1, 9, 1)
iter_count = 200
launch_count = 3


def calc_clusters(X):
    clusters = []
    for k in _clusters:
        print(k, "clusters")
        _, clustered = k_means(X, k, iter_count, launch_count)
        clusters.append(clustered)

    return np.array(clusters)


def calc_metrics(ex_clusters, ac_clusters):
    n = len(ac_clusters)
    tp = sum(sum((ac_clusters[i] == ac_clusters[j]) and (ex_clusters[i] == ex_clusters[j]) and (i != j)
                  for j in range(n)) for i in range(n))
    fp = sum(sum((ac_clusters[i] == ac_clusters[j]) and not (ex_clusters[i] == ex_clusters[j]) and (i != j)
                  for j in range(n)) for i in range(n))
    tn = sum(sum(not (ac_clusters[i] == ac_clusters[j]) and (ex_clusters[i] == ex_clusters[j]) and (i != j)
                  for j in range(n)) for i in range(n))
    fn = sum(sum(not (ac_clusters[i] == ac_clusters[j]) and not (ex_clusters[i] == ex_clusters[j]) and (i != j)
                  for j in range(n)) for i in range(n))

    rand = RandIndex(tp, fn, fp, tn)
    fowlkes_mallows = FowlkesMallowsIndex(tp, fn, fp, tn)
    return np.array([rand, fowlkes_mallows])


def print_plot(score, name):
    plt.plot(_clusters, score, linestyle=':', linewidth=1, marker='s', label=name)
    plt.xticks(np.arange(min(_clusters), max(_clusters) + 1, 1.0))
    plt.xlabel('clusters')
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    data = np.genfromtxt("task_2_data_7.txt", delimiter=' ')
    y, X = np.hsplit(data, [1])
    clusters = calc_clusters(X)
    scores = np.array([calc_metrics(y, ac_clusters) for ac_clusters in clusters])

    print_plot(scores[:, 0], 'RandIndex')
    print_plot(scores[:, 1], 'Fowlkes-Mallows')

    best_n_of_clusters = [_clusters[i] for i in np.argmax(scores, axis=0)]
    print("Best Rand metric is", best_n_of_clusters[0])
    print("Best Fowlkes-Mallows is", best_n_of_clusters[0])
