from datetime import datetime
from sklearn.metrics import *
from statistics import stdev, mean, median
from generated_datasets import *
from tabulate import tabulate
from os import getcwd, path
import getpass
import matplotlib.pyplot as plt
'''
IMPORT YOUR ALGORITHM HERE
e.g.
from sklearn.cluster import KMeans
'''

scores = []

verbose_test_output = open(f'{getcwd()}/{getpass.getuser()}_verbose_output.txt', 'w')
headers = 'dataset_id;adjusted_rand_score;balanced_accuracy_score;hamming_loss;completeness_score;homogeneity_score;time;confusion_matrix'.split(';')
table = []
# verbose_test_output.write('dataset_id;adjusted_rand_score;balanced_accuracy_score;confusion_matrix;hamming_loss;completeness_score;homogeneity_score;time\n')

adj_scores = []
bal_scores = []
conf_mats = []
hamm_scores = []
comp_scores = []
homo_scores = []
times = []


def run_tests(i, X, y, y_pred, time):
    adj = adjusted_rand_score(y, y_pred)
    bal = balanced_accuracy_score(y, y_pred)
    conf = confusion_matrix(y, y_pred)
    hamm = hamming_loss(y, y_pred)
    comp = completeness_score(y, y_pred)
    homo = homogeneity_score(y, y_pred)

    adj_scores.append(adj)
    bal_scores.append(bal)
    conf_mats.append(conf)
    hamm_scores.append(hamm)
    comp_scores.append(comp)
    homo_scores.append(homo)
    times.append(time)

    print(f'Adjusted Rand Score = {adj}')
    print(f'Balanced Accuracy Score = {bal}')
    print(f'Confusion Matrix = {conf}')
    print(f'Hamming Loss = {hamm}')
    print(f'Completeness Score = {comp}')
    print(f'Homogeneity Score = {homo}')

    table.append([i, adj, bal, hamm, comp, homo, time, conf.tolist()])


print(f'Start Time: {datetime.now()}')


for i in range(0, len(Xs)):
    print(('=' * 10) + f' Dataset {i} ' + ('=' * 10))
    print(blobs_made[i])
    X = Xs[i]
    y = ys[i]
    '''
    CREATE CLASSIFER HERE, FIT X AND SET y_pred TO LABELS
    e.g.
    clf = KMeans(n_clusters=clusters[i]).fit(X)
    '''
    start = datetime.now()
    clf.fit(X)
    y_pred = clf.labels_
    runtime = (datetime.now() - start).microseconds
    print(f'microseconds = {runtime}')
    run_tests(i, X, y, clf.labels_, runtime)


print(f'End Time: {datetime.now()}')

verbose_test_output.write(tabulate(table, headers=headers))
verbose_test_output.close()


def print_report(title, adj, bal, hamm, comp, homo, time):
    if path.exists('report.txt'):
        write_mode = 'a'
    else:
        write_mode = 'w'
    test_report = open(f'{getcwd()}/report.txt', write_mode)
    test_report.write('=' * 15 + title + '=' * 15 + '\n')
    headers = 'metric;min;max;mean;median;std_dev'.split(';')
    table = []
    table.append(['Adjusted Rand Score', min(adj), max(adj), mean(adj), median(adj), stdev(adj)])
    table.append(['Balanced Accuracy Score', min(bal), max(bal), mean(bal), median(bal), stdev(bal)])
    table.append(['Hamming Loss', min(hamm), max(hamm), mean(hamm), median(hamm), stdev(hamm)])
    table.append(['Completeness Score', min(comp), max(comp), mean(comp), median(comp), stdev(comp)])
    table.append(['Homogeneity Score', min(homo), max(homo), mean(homo), median(homo), stdev(homo)])
    table.append(['Time', min(time), max(time), mean(time), median(time), stdev(time)])
    test_report.write(tabulate(table, headers=headers))
    test_report.write('\n' * 3)
    test_report.close()


reports = [
    ['Sample Size = 175', [adj_scores[i] for i in samples1], [bal_scores[i] for i in samples1],
     [hamm_scores[i] for i in samples1], [comp_scores[i] for i in samples1], [homo_scores[i] for i in samples1],
     [times[i] for i in samples1]],
    ['Sample Size = 350', [adj_scores[i] for i in samples2], [bal_scores[i] for i in samples2],
     [hamm_scores[i] for i in samples2], [comp_scores[i] for i in samples2], [homo_scores[i] for i in samples2],
     [times[i] for i in samples2]],
    ['Sameple Size = 700', [adj_scores[i] for i in samples3], [bal_scores[i] for i in samples3],
     [hamm_scores[i] for i in samples3], [comp_scores[i] for i in samples3], [homo_scores[i] for i in samples3],
     [times[i] for i in samples3]],
    
    ['Feature Count = 3', [adj_scores[i] for i in features1], [bal_scores[i] for i in features1],
     [hamm_scores[i] for i in features1], [comp_scores[i] for i in features1], [homo_scores[i] for i in features1],
     [times[i] for i in features1]],
    ['Feature Count = 15', [adj_scores[i] for i in features2], [bal_scores[i] for i in features2],
     [hamm_scores[i] for i in features2], [comp_scores[i] for i in features2], [homo_scores[i] for i in features2],
     [times[i] for i in features2]],
    ['Feature Count = 32', [adj_scores[i] for i in features3], [bal_scores[i] for i in features3],
     [hamm_scores[i] for i in features3], [comp_scores[i] for i in features3], [homo_scores[i] for i in features3],
     [times[i] for i in features3]],

    ['Cluster Count = 3', [adj_scores[i] for i in clusters1], [bal_scores[i] for i in clusters1],
     [hamm_scores[i] for i in clusters1], [comp_scores[i] for i in clusters1], [homo_scores[i] for i in clusters1],
     [times[i] for i in clusters1]],
    ['Cluster Count = 6', [adj_scores[i] for i in clusters2], [bal_scores[i] for i in clusters2],
     [hamm_scores[i] for i in clusters2], [comp_scores[i] for i in clusters2], [homo_scores[i] for i in clusters2],
     [times[i] for i in clusters2]],
    ['Cluster Count = 9', [adj_scores[i] for i in clusters3], [bal_scores[i] for i in clusters3],
     [hamm_scores[i] for i in clusters3], [comp_scores[i] for i in clusters3], [homo_scores[i] for i in clusters3],
     [times[i] for i in clusters3]],

    ['Standard Deviation = 1', [adj_scores[i] for i in noise1], [bal_scores[i] for i in noise1],
     [hamm_scores[i] for i in noise1], [comp_scores[i] for i in noise1], [homo_scores[i] for i in noise1],
     [times[i] for i in noise1]],
    ['Standard Deviation = 3', [adj_scores[i] for i in noise2], [bal_scores[i] for i in noise2],
     [hamm_scores[i] for i in noise2], [comp_scores[i] for i in noise2], [homo_scores[i] for i in noise2],
     [times[i] for i in noise2]],
    ['Standard Deviation = 5', [adj_scores[i] for i in noise3], [bal_scores[i] for i in noise3],
     [hamm_scores[i] for i in noise3], [comp_scores[i] for i in noise3], [homo_scores[i] for i in noise3],
     [times[i] for i in noise3]]
]

for rep in reports:
    print_report(*rep)
s_labels = ['175 samples', '350 samples', '700 samples']
f_labels = ['3 features', '15 features', '32 features']
c_labels = ['3 clusters', '6 clusters', '9 clusters']
st_labels = ['σ = 1', 'σ = 3', 'σ = 5']

sadj = [[adj_scores[i] for i in samples1], [adj_scores[i] for i in samples2], [adj_scores[i] for i in samples3]]
fadj = [[adj_scores[i] for i in features1], [adj_scores[i] for i in features2], [adj_scores[i] for i in features3]]
cadj = [[adj_scores[i] for i in clusters1], [adj_scores[i] for i in clusters2], [adj_scores[i] for i in clusters3]]
stadj = [[adj_scores[i] for i in noise1], [adj_scores[i] for i in noise2], [adj_scores[i] for i in noise3]]


sbal = [[bal_scores[i] for i in samples1], [bal_scores[i] for i in samples2], [bal_scores[i] for i in samples3]]
fbal = [[bal_scores[i] for i in features1], [bal_scores[i] for i in features2], [bal_scores[i] for i in features3]]
cbal = [[bal_scores[i] for i in clusters1], [bal_scores[i] for i in clusters2], [bal_scores[i] for i in clusters3]]
stbal = [[bal_scores[i] for i in noise1], [bal_scores[i] for i in noise2], [bal_scores[i] for i in noise3]]

shamm = [[hamm_scores[i] for i in samples1], [hamm_scores[i] for i in samples2], [hamm_scores[i] for i in samples3]]
fhamm = [[hamm_scores[i] for i in features1], [hamm_scores[i] for i in features2], [hamm_scores[i] for i in features3]]
chamm = [[hamm_scores[i] for i in clusters1], [hamm_scores[i] for i in clusters2], [hamm_scores[i] for i in clusters3]]
sthamm = [[hamm_scores[i] for i in noise1], [hamm_scores[i] for i in noise2], [hamm_scores[i] for i in noise3]]

scomp = [[comp_scores[i] for i in samples1], [comp_scores[i] for i in samples2], [comp_scores[i] for i in samples3]]
fcomp = [[comp_scores[i] for i in features1], [comp_scores[i] for i in features2], [comp_scores[i] for i in features3]]
ccomp = [[comp_scores[i] for i in clusters1], [comp_scores[i] for i in clusters2], [comp_scores[i] for i in clusters3]]
stcomp= [[comp_scores[i] for i in noise1], [comp_scores[i] for i in noise2], [comp_scores[i] for i in noise3]]

shomo = [[homo_scores[i] for i in samples1], [homo_scores[i] for i in samples2], [homo_scores[i] for i in samples3]]
fhomo= [[homo_scores[i] for i in features1], [homo_scores[i] for i in features2], [homo_scores[i] for i in features3]]
chomo = [[homo_scores[i] for i in clusters1], [homo_scores[i] for i in clusters2], [homo_scores[i] for i in clusters3]]
sthomo = [[homo_scores[i] for i in noise1], [homo_scores[i] for i in noise2], [homo_scores[i] for i in noise3]]

stime = [[times[i] for i in samples1], [times[i] for i in samples2], [times[i] for i in samples3]]
ftime = [[times[i] for i in features1], [times[i] for i in features2], [times[i] for i in features3]]
ctime = [[times[i] for i in clusters1], [times[i] for i in clusters2], [times[i] for i in clusters3]]
sttime = [[times[i] for i in noise1], [times[i] for i in noise2], [times[i] for i in noise3]]

plot_data = [(sadj, s_labels, 'Adjusted Rand Score by Number of Samples'),
(fadj, f_labels, 'Adjusted Rand Score by Number of Features'),
(cadj, c_labels, 'Adjusted Rand Score by Number of Clusters'),
(stadj, st_labels, 'Adjusted Rand Score by Standard Deviation of Dataset'),
(sbal, s_labels, 'Balanced Accuracy Score by Number of Samples'),
(fbal, f_labels, 'Balanced Accuracy Score by Number of Features'),
(cbal, c_labels, 'Balanced Accuracy Score by Number of Clusters'),
(stbal, st_labels, 'Balanced Accuracy Score by Standard Deviation of Dataset'),
(shamm, s_labels, 'Hamming Loss by Number of Samples'),
(fhamm, f_labels, 'Hamming Loss by Number of Features'),
(chamm, c_labels, 'Hamming Loss by Number of Clusters'),
(sthamm, st_labels, 'Hamming Loss by Standard Deviation of Dataset'),
(scomp, s_labels, 'Completeness Score by Number of Samples'),
(fcomp, f_labels, 'Completeness Score by Number of Features'),
(ccomp, c_labels, 'Completeness Score by Number of Clusters'),
(stcomp, st_labels, 'Completeness Score by Standard Deviation of Dataset'),
(shomo, s_labels, 'Homogeneity Score by Number of Samples'),
(fhomo, f_labels, 'Homogeneity Score by Number of Features'),
(chomo, c_labels, 'Homogeneity Score by Number of Clusters'),
(sthomo, st_labels, 'Homogeneity Score by Standard Deviation of Dataset'),
(stime, s_labels, 'Time (μ seconds) by Number of Samples'),
(ftime, f_labels, 'Time (μ seconds) by Number of Features'),
(ctime, c_labels, 'Time (μ seconds) by Number of Clusters'),
(sttime, st_labels, 'Time (μ seconds) by Standard Deviation of Dataset')]

filenumber = 0
for p in plot_data:
    fig, ax = plt.subplots()
    ax.set_title(p[2])
    ax.boxplot(p[0], labels=p[1], showmeans=True)
    plt.savefig(str(filenumber) + '.png')
    filenumber += 1
print('DONE')
