import math

import numpy as np

UNCLASSIFIED = False
NOISE = -1


def _dist(p, q):
    return math.sqrt(np.power(p - q, 2).sum())


def _eps_neighborhood(p, q, eps):
    return _dist(p, q) < eps

# todo replace with R-tree
def _region_query(m, point_id, eps):
    n_points = m.shape[0]
    seeds = []
    for i in range(0, n_points):
        if _eps_neighborhood(m[point_id], m[i], eps):
            seeds.append(i)
    return seeds


def _expand_cluster(m, classifications, point_id, cluster_id, eps, min_points):
    seeds = _region_query(m, point_id, eps)
    if len(seeds) < min_points:
        classifications[point_id] = NOISE
        return False
    else:
        classifications[point_id] = cluster_id
        for seed_id in seeds:
            classifications[seed_id] = cluster_id

        while len(seeds) > 0:
            current_point = seeds.pop(0)
            results = _region_query(m, current_point, eps)
            if len(results) >= min_points:
                for i in range(0, len(results)):
                    result_point = results[i]
                    if classifications[result_point] == UNCLASSIFIED or \
                            classifications[result_point] == NOISE:
                        if classifications[result_point] == UNCLASSIFIED:
                            seeds.append(result_point)
                        classifications[result_point] = cluster_id
        return True


class DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(self, X):
        cluster_id = 1
        n_points = X.shape[0]
        classifications = [UNCLASSIFIED] * n_points
        for point_id in range(0, n_points):
            if X[point_id][0] > 2 and X[point_id][1] < 0:
                l = 2
            if classifications[point_id] == UNCLASSIFIED:
                if _expand_cluster(X, classifications, point_id, cluster_id, self.eps, self.min_samples):
                    cluster_id = cluster_id + 1
        return classifications
