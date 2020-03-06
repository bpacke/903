from sklearn.datasets import make_blobs
from os import getcwd
samples = [175, 350, 700]
features = [3, 15, 35]
centers = [3, 6, 9]
stand_dev = [1, 3, 5]
randoms = [1, 2, 3]

blobs_made = []
Xs = []
ys = []
clusters = []

samples1 = []
samples2 = []
samples3 = []
features1 =[]
features2 = []
features3 = []
clusters1 = []
clusters2 = []
clusters3 = []
noise1 = []
noise2 = []
noise3 = []

ds_id = 0
for s in samples:
    for f in features:
        for c in centers:
            for d in stand_dev:
                for r in randoms:
                    X, y = make_blobs(n_samples=s, n_features=f,  centers=c, cluster_std=d, shuffle=False, random_state=r)
                    blobs_made.append(f'X, y = make_blobs(n_samples={s}, n_features={f}, centers={c}, cluster_std={d}, shuffle=False, '
                      f'random_state={r})  # {ds_id}')
                    clusters.append(c)
                    Xs.append(X)
                    ys.append(y)
                    if s == samples[0] : samples1.append(s)
                    if s == samples[1] : samples2.append(s)
                    if s == samples[2] : samples3.append(s)
                    if f == features[0]: features1.append(s)
                    if f == features[1]: features2.append(s)
                    if f == features[2]: features3.append(s)
                    if c == centers[0]: clusters1.append(s)
                    if c == centers[1]: clusters2.append(s)
                    if c == centers[2]: clusters3.append(s)
                    if d == stand_dev[0]: noise1.append(s)
                    if d == stand_dev[1]: noise2.append(s)
                    if d == stand_dev[2]: noise3.append(s)

                    ds_id += 1

# print(f'Datasets Count = {len(blobs_made)}')
# print(f'{getcwd()}/test_output/datasets.txt')
with open(f'{getcwd()}/test_output/datasets.txt', 'w') as f:
    for b in blobs_made:
        f.write(b + '\n')

