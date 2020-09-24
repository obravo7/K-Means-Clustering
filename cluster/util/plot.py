import matplotlib.pyplot as plt
import numpy as np


def scatter3d(clusters: dict,
              centroids: np.ndarray,
              title: str,
              save_file=None) -> None:
    ax = plt.axes(projection='3d')
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    for k in range(len(centroids)):
        # as float assures that points get read correctly. Otherwise error is raised for Iris data points.
        points = np.array(clusters[str(k + 1)], dtype=np.float)
        try:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors[k])
        except ValueError:  # empty cluster?
            pass

    ax.set_title(title)
    plt.show()
    if save_file is not None:
        plt.savefig(save_file)


def scatter2d(clusters: dict,
              centroids: np.ndarray,
              title: str) -> None:
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    for k in range(len(centroids)):
        points = np.array(clusters[str(k + 1)])
        plt.scatter(points[:, 0], points[:, 1], c=colors[k])
    plt.title(title)
    plt.show()


def plot3d_side_by_side(c1, c2, ct1, ct2):
    scatter3d(c1, ct1, title="K-Means: L2 Norm")
    scatter3d(c2, ct2, title="K-Means: L1 Norm")
    plt.tight_layout()
    plt.show()
