from datetime import datetime
from sklearn.metrics import *
from statistics import stdev, mean, median
from generated_datasets import *
from tabulate import tabulate
from os import getcwd
import getpass
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
    test_report = open(f'{getcwd()}/test_report_{title}.txt', 'w')
    headers = 'metric;min;max;mean;median;std_dev'.split(';')
    table = []
    table.append(['Adjusted Rand Score', min(adj), max(adj), mean(adj), median(adj), stdev(adj)])
    table.append(['Balanced Accuracy Score', min(bal), max(bal), mean(bal), median(bal), stdev(bal)])
    table.append(['Hamming Loss', min(hamm), max(hamm), mean(hamm), median(hamm), stdev(hamm)])
    table.append(['Completeness Score', min(comp), max(comp), mean(comp), median(comp), stdev(comp)])
    table.append(['Homogeneity Score', min(homo), max(homo), mean(homo), median(homo), stdev(homo)])
    table.append(['Time', min(time), max(time), mean(time), median(time), stdev(time)])
    test_report.write(tabulate(table, headers=headers))
    test_report.close()


reports = [
    ['175 Samples', [adj_scores[i] for i in samples1], [bal_scores[i] for i in samples1],
     [hamm_scores[i] for i in samples1], [comp_scores[i] for i in samples1], [homo_scores[i] for i in samples1],
     [times[i] for i in samples1]],
    ['350 Samples', [adj_scores[i] for i in samples2], [bal_scores[i] for i in samples2],
     [hamm_scores[i] for i in samples2], [comp_scores[i] for i in samples2], [homo_scores[i] for i in samples2],
     [times[i] for i in samples2]],
    ['700 Samples', [adj_scores[i] for i in samples3], [bal_scores[i] for i in samples3],
     [hamm_scores[i] for i in samples3], [comp_scores[i] for i in samples3], [homo_scores[i] for i in samples3],
     [times[i] for i in samples3]],
    
    ['3_Features', [adj_scores[i] for i in features1], [bal_scores[i] for i in features1],
     [hamm_scores[i] for i in features1], [comp_scores[i] for i in features1], [homo_scores[i] for i in features1],
     [times[i] for i in features1]],
    ['15_Features', [adj_scores[i] for i in features2], [bal_scores[i] for i in features2],
     [hamm_scores[i] for i in features2], [comp_scores[i] for i in features2], [homo_scores[i] for i in features2],
     [times[i] for i in features2]],
    ['32_Features', [adj_scores[i] for i in features3], [bal_scores[i] for i in features3],
     [hamm_scores[i] for i in features3], [comp_scores[i] for i in features3], [homo_scores[i] for i in features3],
     [times[i] for i in features3]],

    ['3_Clusters', [adj_scores[i] for i in clusters1], [bal_scores[i] for i in clusters1],
     [hamm_scores[i] for i in clusters1], [comp_scores[i] for i in clusters1], [homo_scores[i] for i in clusters1],
     [times[i] for i in clusters1]],
    ['6_Clusters', [adj_scores[i] for i in clusters2], [bal_scores[i] for i in clusters2],
     [hamm_scores[i] for i in clusters2], [comp_scores[i] for i in clusters2], [homo_scores[i] for i in clusters2],
     [times[i] for i in clusters2]],
    ['9_Clusters', [adj_scores[i] for i in clusters3], [bal_scores[i] for i in clusters3],
     [hamm_scores[i] for i in clusters3], [comp_scores[i] for i in clusters3], [homo_scores[i] for i in clusters3],
     [times[i] for i in clusters3]],

    ['1_stddev', [adj_scores[i] for i in noise1], [bal_scores[i] for i in noise1],
     [hamm_scores[i] for i in noise1], [comp_scores[i] for i in noise1], [homo_scores[i] for i in noise1],
     [times[i] for i in noise1]],
    ['3_stddev', [adj_scores[i] for i in noise2], [bal_scores[i] for i in noise2],
     [hamm_scores[i] for i in noise2], [comp_scores[i] for i in noise2], [homo_scores[i] for i in noise2],
     [times[i] for i in noise2]],
    ['5_stddev', [adj_scores[i] for i in noise3], [bal_scores[i] for i in noise3],
     [hamm_scores[i] for i in noise3], [comp_scores[i] for i in noise3], [homo_scores[i] for i in noise3],
     [times[i] for i in noise3]]
]

for rep in reports:
    print_report(*rep)

print('DONE')
