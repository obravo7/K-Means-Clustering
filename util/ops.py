import numpy as np
import pandas as pd
from typing import Tuple, List, Any
from collections import Counter


def generate_uniform(n, d) -> np.ndarray:
    """Generate uniform data of dim (nxd)"""
    x = np.array([np.random.uniform(-1.5, 1.5, n) for _ in range(d)]).T
    return x


def normalize(x):
    """normalize a numpy array."""
    xmax, xmin = x.max(), x.min()
    x = (x - xmin) / (xmax - xmin)
    return x


def load_iris_training_set(file_path: str) -> Tuple[np.ndarray, list]:
    iris_data = pd.read_csv(file_path, header=None)
    train_data = iris_data.values[:, 0: 4]
    train_labels = iris_data.values[:, 4]

    # shuffle / training split (80/20)
    # np.random.shuffle(train_data)
    # t_size = int(len(train_data) * .80)
    # x_train = train_data[:t_size, :]
    # x_eval = train_data[t_size:, :]
    # return x_train, x_eval, train_labels
    return train_data, train_labels


def load_iris_test_set(file_path: str, norm: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    iris_data = pd.read_csv(file_path, header=None)
    test_data = iris_data.values[:, 0: 4]
    labels = iris_data.values[:, 4]
    if norm:
        test_data = normalize(test_data)
    return test_data, labels


def get_label_mapping(clusters: List[dict], data: np.ndarray, labels_gt: list) -> dict:

    labels_pt = [float('Nan')] * data.shape[0]
    for cl in clusters:
        for cl_key in cl.keys():
            cluster_points = cl[cl_key]
            for cl_point in cluster_points:
                idx = np.where((data == cl_point).all(axis=1))[0]
                for i in idx:
                    labels_pt[i] = str(cl_key)

    matching = match_labels(gt=labels_gt, pt=labels_pt)

    return matching


def write_csv(clusters: List[dict], data: np.ndarray, labels_gt: list, file_name: str) -> None:
    """
    create a csv file that matches original train/test files, but
    with predicted classes included as a column; predicted output from K-means are matched with
    categories in ground truth labels.
    """
    # place predicted values into correct position based on data
    labels_pt = [float('Nan')] * data.shape[0]
    for cl in clusters:
        for cl_key in cl.keys():
            cluster_points = cl[cl_key]
            for cl_point in cluster_points:
                idx = np.where((data == cl_point).all(axis=1))[0]
                for i in idx:
                    labels_pt[i] = str(cl_key)

    matching = match_labels(gt=labels_gt, pt=labels_pt)
    print(f"matchings={matching}")  # debug

    # convert values in labels_pt to match categories in labels_gt
    predicted = []
    for val in labels_pt:
        try:
            predicted.append(matching[val])
        except KeyError:  # no matching...?!
            predicted.append(val)

    file_pd = pd.DataFrame({'1': data[:, 0], '2': data[:, 1], '3': data[:, 2], '4': data[:, 3], 'gt': labels_gt,
                            'pt': predicted})  # labels_pt
    file_pd.to_csv(file_name, header=False)


def match_labels(gt: list, pt: list) -> dict:
    """create a matching between the real labels and the predicted labels from K-means"""
    t = {}
    label_match = {}
    for i, j in zip(gt, pt):
        if i in t.keys():
            t[i].append(j)
        else:
            t[i] = [j]
    for item in t.keys():
        val = t[item]
        c = _most_common(val)
        label_match[c] = item
    return label_match


def _most_common(target) -> Any:
    counts = Counter(target)
    c = counts.most_common(1)[0]
    return c[0]


def reverse_dict(target: dict) -> dict:
    """reverse the mapping in a dictionary"""
    return {item: key for key, item in target.items()}
