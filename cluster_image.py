import cluster
from PIL import Image
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path).resize((256, 256), Image.ANTIALIAS)
    return np.array(image)


def unwind(target_image: np.ndarray) -> np.ndarray:
    assert len(target_image.shape) == 3
    return target_image.reshape((-1, 3))


pixels = {
    '1': [255, 0, 0],
    '2': [0, 0, 255],
    '3': [255, 255, 0],
    '4': [255, 192, 203],
    '5': [0, 128, 0]
}


def cluster_image(image: np.ndarray):
    data = unwind(image.copy())  # unwind but keep original
    mask = np.ones(data.shape, dtype=np.uint8) * 255
    kmeans = cluster.Kmeans(n_clusters=2, data=data, iterations=10000)
    centers, clusters = kmeans.train(method='l2')
    print(f'centers: {centers} \n')
    for item in clusters.keys():
        t1 = clusters[item]
        print(f"(l2)-item key: {item}, shape: {np.array(t1).shape} \n")

    predicted_labels = [float('Nan')] * data.shape[0]
    for cl_key in clusters.keys():  # slow
        cluster_points = clusters[cl_key]
        for cl_point in cluster_points:
            idx = np.where((data == cl_point).all(axis=1))[0]
            for i in idx:
                predicted_labels[i] = str(cl_key)

    for idx, point in enumerate(mask):
        p_val = predicted_labels[idx]
        p = pixels[p_val]
        mask[idx] = p

    mask = mask.reshape(image.shape)
    mask = Image.fromarray(mask, 'RGB')
    # mask = mask.resize((1024, 1024), Image.ANTIALIAS)
    mask.save("obama-mask-256-10000.png")

    return centers, clusters


if __name__ == "__main__":
    import time
    start = time.time()
    image = load_image("obama.jpg")
    _, _ = cluster_image(image)
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    t = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
    print("Done. Total time: {0}".format(t))
