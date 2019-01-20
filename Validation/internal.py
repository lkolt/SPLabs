import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from Validation.metrics import get_metric
from KMeans.main import k_means, get_path

# Consts
_clusters = np.arange(2, 9, 1)
iter_count = 200
launch_count = 3
image_name = "policemen"


def calc_clusters():
    image = np.array(Image.open(get_path(image_name)))
    X = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))

    data = dict()
    for k in _clusters:
        print(k, "clusters")
        centroids, clustered = k_means(X, k=k, iter_count=iter_count, launch_count=launch_count)
        data[k] = {'centroids': centroids, 'clusters': clustered}

    return X, data


def get_best_k_ch(n_clusters, scores):
    def delta(i):
        return (scores[i + 1] - scores[i]) - (scores[i] - scores[i - 1])

    deltas = [delta(i) for i in range(1, len(scores) - 1)]
    return n_clusters[1 + np.argmin(deltas)]


def compress(centroids, clusters):
    image = np.array(Image.open(get_path(image_name)))
    new_X = np.array([centroids[cluster_index] for cluster_index in clusters])
    new_image = new_X.astype(np.uint8).reshape(image.shape)
    Image.fromarray(new_image).save(get_path(image_name + '_compressed'))


def calc_metric(X, cluster_data, name):
    print("Calculating", name ,"scores...")
    scores =  [get_metric(X, k, cluster_data[k]['clusters'], cluster_data[k]['centroids'], name)
                 for k in _clusters]
    print("Done.")
    return scores


def print_plot(score, name):
    plt.plot(_clusters, score, linestyle=':', linewidth=1, marker='s', label=name)
    plt.xticks(np.arange(min(_clusters), max(_clusters) + 1, 1.0))
    plt.xlabel('clusters')
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    X, cluster_data = calc_clusters()
    db_scores = calc_metric(X, cluster_data, 'DaviesBouldin')
    ch_scores = calc_metric(X, cluster_data, 'CalinskiHarabasz')

    print_plot(db_scores, 'DaviesBouldin')
    print_plot(ch_scores, 'CalinskiHarabasz')

    best_k_db = _clusters[np.argmin(db_scores)]
    best_k_ch = get_best_k_ch(_clusters, ch_scores)
    print("Best Davies-Bouldin is", best_k_db)
    print("Best Calinski-Harabasz is", best_k_ch)

    compress(cluster_data[best_k_ch]['centroids'], cluster_data[best_k_ch]['clusters'])
