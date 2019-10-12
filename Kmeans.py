import numpy as np
import matplotlib.pyplot as plt


class K_means():
    def __init__(self, data, n_clusters, iterations=50):
        self.data = data
        self.n_clusters = n_clusters
        self.iterations = iterations

    def kmeans(self):
        data = self.data
        k = self.n_clusters
        iterations = self.iterations
        cluster_assignments = K_means.assign_cluster(k)
        clusters = K_means.random_clusters(k)

        while iterations != 0:
            for j in range(k):
                cluster_assignments[str(j + 1)] = []

            for i in range(len(data)):
                curr_distance = K_means.euclidean_distance(data[i], clusters)
                cluster = np.argmin(curr_distance) + 1
                cluster_assignments[str(cluster)].append(data[i])
                
            for c in range(k):
                mean = np.mean(np.array(cluster_assignments[str(c + 1)]), axis=0)
                clusters[c] = mean

            iterations -= 1

        return cluster_assignments, clusters

    @classmethod
    def euclidean_distance(cls, point, center, axis=1):
        return np.linalg.norm(point - center, axis=axis)

    @staticmethod
    def assign_cluster(k):
        return {str(i): [] for i in range(1, k+1)}

    @staticmethod
    def random_clusters(c):
        return np.random.randn(c, 2)

    def show_points(self, clusters, centroids):
        colors = ['r', 'g', 'b', 'y', 'c', 'm']
        for k in range(len(centroids)):
            points = np.array(clusters[str(k + 1)])
            plt.scatter(points[:, 0], points[:, 1], c=colors[k])
        plt.show()


# Create test data (normal and uniform)
def generate_points(N, mu = 0, sigma = 2.5):
    return (np.random.normal(mu, sigma, N), np.random.normal(mu, sigma, N))


def generate_uniform(N):
    return (np.random.uniform(-1.5, 1.5, N), np.random.uniform(-1.5, 1.5, N))


if __name__ == "__main__":
    # create dummy data
    N = 500
    data = np.concatenate((generate_points(N), generate_points(N)), axis=1)
    data = data.T

    # display data
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()

    # display results of k-means
    K = K_means(data, 3)
    clusters, centroids = K.kmeans()

    K.show_points(clusters, centroids)