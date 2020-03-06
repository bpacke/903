import sys
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import *
from collections import Counter


class Agg:
    def __init__(self, desired_clusters, metric='euclidean'):
        self.metrics_implemented = ['euclidean', 'manhattan', 'chebyshev']
        if metric not in self.metrics_implemented:
            raise NotImplementedError(f'Metric "{metric}" is not an implemented distance metric. '
                                      f'Available metrics:\n{self.metrics_implemented}')
        self.desired_clusters = desired_clusters
        self.metric = metric
        self.labels_ = np.array([])
        self.x_clusters = []

    def fit(self, x):
        x_clusters = [self.Cluster(z) for z in x]
        while len(x_clusters) > self.desired_clusters:
            centroids = [c.centroid for c in x_clusters]
            sci_distances = []
            if self.metric == 'euclidean':
                sci_distances = distance_matrix(centroids, centroids)
            elif self.metric == 'manhattan':
                sci_distances = pairwise_distances(centroids, centroids, metric='manhattan')
            elif self.metric == 'chebyshev':
                sci_distances = pairwise_distances(centroids, centroids, metric='chebyshev')
            sci_distance_minus_zeros = np.where(sci_distances == 0.0, sys.maxsize, sci_distances)
            nearest_clusters = np.where(sci_distance_minus_zeros == np.amin(sci_distance_minus_zeros))[0]
            nearest_clusters = tuple(nearest_clusters[0:2])
            if nearest_clusters[0] == nearest_clusters[1]:
                nearest_clusters = np.where(sci_distance_minus_zeros == np.amin(sci_distance_minus_zeros))[0]
                nearest_clusters = tuple([nearest_clusters[2], nearest_clusters[3]])
            x_clusters[nearest_clusters[0]] = x_clusters[nearest_clusters[0]].merge_to_new_cluster(
                x_clusters[nearest_clusters[1]])
            x_clusters = x_clusters[:nearest_clusters[1]] + x_clusters[nearest_clusters[1] + 1:]
        dp_cluster = []
        for clust in x_clusters:
            dp_cluster.append([(c, clust.id) for c in clust.data_points])

        dp_cluster = [item for sublist in dp_cluster for item in sublist]
        cluster = [item[1] for item in dp_cluster]
        cluster_ordered = []
        for x in x_clusters:
            for dp in x.data_points:
                cluster_ordered.append(dp.tolist() + [x.id])
        cluster_ordered = np.sort(np.array(cluster_ordered), axis=0)
        self.labels_ = cluster_ordered[:, -1]
        tags = list(set(self.labels_))
        tags.sort()
        self.labels_ = np.array(list(map(lambda x: tags.index(x), self.labels_)))
        pause = 'pause'
        # self.labels_ = [list(set(cluster)).index(z) for z in [item[1] for item in dp_cluster]]

    class Cluster:
        class_id = 0

        def __init__(self, point):
            Agg.Cluster.class_id += 1
            self.id = Agg.Cluster.class_id
            self.data_points = np.array([point])
            self.centroid = tuple(point)

        def merge_to_new_cluster(self, cluster):
            # self.data_points += cluster.data_points
            # test = list(self.data_points)
            self.data_points = np.concatenate((self.data_points, cluster.data_points))
            self.centroid = self.calculate_centroid()
            return self

        def calculate_centroid(self):
            try:
                dimes = self.data_points.shape[1]
            except IndexError:
                dimes = 1
            return tuple([np.average(self.data_points[:, i]) for i in range(dimes)])

        def __str__(self):
            return f'ID: {self.id}  -  Cluster Centroid: {self.centroid}\n' \
                   f'data_points {len(self.data_points)}: {self.data_points.tolist()} '

        def __eq__(self, other):
            return self.id == other.id and self.data_points == other.data_points
