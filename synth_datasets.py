from sklearn.datasets import make_blobs
features = [3, 6, 9, 18, 27]
sdev = [0.5, 1, 1.5, 2]

datasets = []
Xs = []
ys = []

for f in features:
    for d in sdev:
        X, y = make_blobs(n_samples=350, n_features=f,  centers=5, cluster_std=d, shuffle=False, random_state=42)
        Xs.append(X)
        ys.append(y)
