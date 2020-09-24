import cluster
import util
import numpy as np


def train_iris() -> None:
    """TRI coding challenge. Run training, save graphs, save files. """

    train_data, x_labels = util.load_iris_training_set("data/iris_train.csv")
    data = train_data.copy()  # used to recreate csv file.
    np.random.shuffle(train_data)
    t_size = int(len(train_data) * .80)
    x_train = train_data[:t_size, :]
    x_eval = train_data[t_size:, :]
    n_clusters = 3

    K_l2 = cluster.Kmeans(n_clusters=n_clusters, data=x_train)
    centroids, clusters = K_l2.train(method="l2")
    p_clusters, p_centroids = K_l2.predict(x_eval, method='l2')  # predict on the evaluation set.

    util.write_csv(clusters=[clusters, p_clusters],
                   data=data,
                   labels_gt=x_labels,
                   file_name="data/results/train_results_l2-run2.csv")

    cluster.plot.scatter3d(clusters, centroids, title="K-Means: L2 Norm", save_file="data/images/K-means-L2-Norm.png")

    K_l1 = cluster.Kmeans(n_clusters=n_clusters, data=x_train)
    centroids_l1, clusters_l1 = K_l1.train(method="l1")
    pcl_l1, pct_l1 = K_l1.predict(x_eval, method='l1')

    util.write_csv(clusters=[clusters_l1, pcl_l1],
                   data=data,
                   labels_gt=x_labels,
                   file_name="data/results/train_results_l1-run2.csv")

    cluster.plot.scatter3d(clusters_l1, centroids_l1, title="K-Means: L1 Norm", save_file="data/images/K-means-L1-Norm.png")

    x_test, labels_gt = util.load_iris_test_set("data/iris_test.csv")
    tl2_cl_pt, tl2_ct_pt = K_l2.predict(x_test, method='l2')

    util.write_csv(clusters=[tl2_cl_pt],
                   data=x_test,
                   labels_gt=list(labels_gt),
                   file_name="data/results/test_results_l2-run2.csv")

    cluster.plot.scatter3d(tl2_cl_pt, tl2_ct_pt, title="K-Means Test: L2 Norm",
                           save_file="data/images/K-means-test-L1-Norm.png")

    tl1_cl_pt, tl1_ct_pt = K_l1.predict(x_test, method="l1")

    util.write_csv(clusters=[tl1_cl_pt],
                   data=x_test,
                   labels_gt=list(labels_gt),
                   file_name="data/results/test_results_l1-run2.csv")

    cluster.plot.scatter3d(tl1_cl_pt, tl1_ct_pt, title="K-Means Test: L1 Norm",
                           save_file="data/images/K-means-test-L1-Norm.png")


def run_sample(n=1000, d=2) -> None:
    """
    Small util function used for debugging/testing purposes.

    Args:
        n:  (int) number of data points
        d: (int)  dimensions of data. Should be no more than 100.
    """
    n_clusters = 3
    data = util.generate_uniform(n, d)

    K1 = cluster.Kmeans(n_clusters=n_clusters, data=data, iterations=500)
    K2 = cluster.Kmeans(n_clusters=n_clusters, data=data, iterations=500)
    centroids, clusters = K1.train(method="l2")
    centroids2, clusters2 = K2.train(method="l1")
    print(f"centroids: {centroids}\n")
    print(f"centroids: {centroids2}\n")

    for item in clusters.keys():
        t1 = clusters[item]
        t2 = clusters2[item]  # should have the same items as t1
        print(f"(l2)-item key: {item}, shape: {np.array(t1).shape}")
        print(f"(l1)-item key: {item}, shape: {np.array(t2).shape}")

    # generate new data for prediction
    data_eval = util.generate_uniform(n=1000, d=3)

    cl_p_l2, ctr_p_l2 = K1.predict(data_eval, method="l2")
    cl_p_l1, ctr_p_l1 = K2.predict(data_eval, method="l1")

    # visualize
    if d >= 3:
        cluster.plot.scatter3d(clusters, centroids, title="K-Means Predict: L2 Norm")
        cluster.plot.scatter3d(clusters2, centroids2, title="K-Means Predict: L1 Norm")

        cluster.plot.scatter3d(cl_p_l2, ctr_p_l2, title="K-Means Predict: L2 Norm")
        cluster.plot.scatter3d(cl_p_l1, ctr_p_l1, title="K-Means Predict: L1 Norm")
    elif d == 2:
        cluster.plot.scatter2d(clusters, centroids, title="K-Means Predict: L2 Norm")
        cluster.plot.scatter2d(clusters2, centroids2, title="K-Means Predict: L1 Norm")

        cluster.plot.scatter2d(cl_p_l2, ctr_p_l2, title="K-Means Predict: L2 Norm")
        cluster.plot.scatter2d(cl_p_l1, ctr_p_l1, title="K-Means Predict: L1 Norm")


if __name__ == "__main__":
    # run_sample()
    train_iris()
