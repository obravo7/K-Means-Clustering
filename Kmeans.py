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
        cluster_assignments = self.assign_cluster(k)
        clusters = self.random_clusters(k)

        while iterations != 0:
            for j in range(k):
                cluster_assignments[str(j + 1)] = []

            for i in range(len(data)):
                curr_distance = self.euclidean_distance(data[i], clusters)
                cluster = np.argmin(curr_distance) + 1
                cluster_assignments[str(cluster)].append(data[i])
                
            for c in range(k):
                mean = np.mean(np.array(cluster_assignments[str(c + 1)]), axis=0)
                clusters[c] = mean

            iterations -= 1

        return cluster_assignments, clusters

    def euclidean_distance(self, point, center, axis=1):
        return np.linalg.norm(point - center, axis=axis)

    def assign_cluster(self, k):
        return {str(i): [] for i in range(1, k+1)}

    def random_clusters(self, c):
        return np.random.randn(c, 2)

    def show_points(self, clusters, centroids):
        colors = ['r', 'g', 'b', 'y', 'c', 'm']
        for k in range(len(centroids)):
            points = np.array(clusters[str(k + 1)])
            plt.scatter(points[:, 0], points[:, 1], c=colors[k])
            plt.show

# Create test data (normal and uniform)
def generate_points(N, mu = 0, sigma = 2.5):
    return (np.random.normal(mu, sigma, N), np.random.normal(mu, sigma, N))


def generate_uniform(N):
    return (np.random.uniform(-1.5, 1.5, N), np.random.uniform(-1.5, 1.5, N))

N = 250
pt = generate_points(N)
pt2 = generate_points(N)
c = np.concatenate((pt, pt2), axis=1)
data = c.T
plt.scatter(data[:, 0], data[:, 1])
plt.show()

k = 3
K = K_means(data, 3)
clusters, centroids = K.kmeans()
K.show_points(clusters, centroids)
