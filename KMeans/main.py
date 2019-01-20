from PIL import Image
import numpy as np
from numpy import random
from copy import *


pics_path = './pic/'
pics = ['lena', 'grain', 'peppers']
pics_format = '.jpg'


def get_path(pic_name):
    return pics_path + pic_name + pics_format


def k_means(X, k, iter_count, launch_count):
    n = X.shape[0]
    best_func = 1e18
    best_centroids = np.zeros(shape=n)
    best_clusters = np.zeros(shape=n)

    for launch in range(launch_count):
        print('\tLaunch:', launch, '\\', launch_count, ', cur_best_func: ', best_func)
        centroids = X[random.choice(range(n), size=k, replace=False)]
        centroids_new = deepcopy(centroids)
        clusters = np.zeros(n)
        distances = np.zeros((n, k))

        for iter in range(iter_count):
            for i in range(k):
                distances[:, i] = np.linalg.norm(X - centroids_new[i], axis=1)
            clusters = np.argmin(distances, axis=1)

            centers_old = deepcopy(centroids_new)
            for i in range(k):
                centroids_new[i] = np.mean(X[clusters == i], axis=0)
            if np.linalg.norm(centroids_new - centers_old) == 0:
                break

        cur_func = 0.
        for i, c in enumerate(centroids_new):
            cur_func += sum([np.linalg.norm(c - X[j]) for j in range(n) if clusters[j] == i])
        if cur_func < best_func:
            best_func = cur_func
            best_centroids = centroids_new
            best_clusters = clusters
    print('\tKMeans ended with best_func:', best_func)
    return best_centroids, best_clusters


def compress(input, k, output):
    image = np.array(Image.open(input))
    X = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))
    cent, clust = k_means(X, k, 200, 3)
    compress_image = np.vstack([cent[i] for i in clust]).reshape(image.shape)
    Image.fromarray(compress_image).save(output)


if __name__ == "__main__":
    for pic in pics:
        for k in np.arange(2, 9, 2):
            print('convert ' + pic + ', colors: ' + str(k))
            compress(get_path(pic), k, get_path(pic + '_compressed_' + str(k)))
