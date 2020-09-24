import numpy as np
from cluster.util import distance
from typing import Tuple


class Kmeans:
    def __init__(self,
                 n_clusters: int,          # number of clusters assignments
                 data: np.ndarray,         # data matrix (n x d)
                 iterations: int = 1000    # number of iterations to run.
                 ):  # unhappy face.
        self.n_clusters = n_clusters
        self.data = data
        self.iterations = iterations
        self.clusters = _initialize_clusters(self.n_clusters)  # define cluster dictionary for training data

        if not _verify_dim(self.data):  # check dimension size is valid
            raise ValueError("input data has dimensions exceeding 100")

        index = np.random.choice(self.data.shape[0], self.n_clusters, replace=False)
        self.centroids = self.data[index]

    def train(self, method: str = "l2", verbose: bool = True) -> Tuple[np.ndarray, dict]:
        iterations = self.iterations  # allows for multiple training runs if user desires

        if method == "l2":
            self._train_l2(iterations)
        elif method == "l1":
            self._train_l1(iterations, verbose=verbose)

        if verbose:
            print("Training done...")
        return self.centroids, self.clusters

    def predict(self, x: np.ndarray, method: str = "l2") -> Tuple[dict, np.ndarray]:
        """calculate closest distance between input matrix and cluster assignments"""
        if not _verify_dim(x):
            raise ValueError("input data has dimensions exceed 100")

        # initialize cluster dictionary.
        clusters = _initialize_clusters(self.n_clusters)

        if method == "l2":
            for i in range(len(x)):
                d = distance.l2norm_vector(x[i], self.centroids)
                c = np.argmin(d) + 1
                clusters[str(c)].append(x[i])
        elif method == "l1":
            for i in range(len(x)):
                d = distance.l1norm_vector(x[i], self.centroids)
                c = np.argmin(d) + 1
                clusters[str(c)].append(x[i])
        else:
            raise ValueError("Unrecognized method. Please choose between either 'l2' for l2-norm, or 'l1' for l1-norm")

        return clusters, self.centroids

    def _train_l1(self, iterations: int, verbose: bool):
        """run training using l1-norm"""

        while iterations != 0:
            for j in range(self.n_clusters):
                self.clusters[str(j + 1)] = []
            for i in range(len(self.data)):
                d = distance.l1norm_vector(self.data[i], self.centroids)
                c = np.argmin(d) + 1
                self.clusters[str(c)].append(self.data[i])

            old_centroids = self.centroids[:]
            for c in range(self.n_clusters):
                mean = np.mean(np.array(self.clusters[str(c + 1)]), axis=0)
                self.centroids[c] = mean

            iterations -= 1
            if np.all(old_centroids == self.centroids):
                break

    def _train_l2(self, iterations: int):
        """run training using l2-norm"""
        i = 0
        while iterations != 0:
            for j in range(self.n_clusters):
                self.clusters[str(j + 1)] = []
            for i in range(len(self.data)):
                d = distance.l2norm_vector(self.data[i], self.centroids)
                c = np.argmin(d) + 1
                self.clusters[str(c)].append(self.data[i])

            old_centroids = self.centroids.copy()
            for c in range(self.n_clusters):
                mean = np.mean(np.array(self.clusters[str(c + 1)]), axis=0)
                self.centroids[c] = mean

            print("\riterations: {}...".format(iterations))
            iterations -= 1
            i += 1
            if np.all(old_centroids == self.centroids):
                break

# -------------------------------------------------------------
# helper functions
# -------------------------------------------------------------


def _initialize_clusters(k: int) -> dict:
    return {str(i): [] for i in range(1, k + 1)}


def _verify_dim(x: np.ndarray) -> bool:
    if x.shape[1] > 100:
        return False
    else:
        return True
