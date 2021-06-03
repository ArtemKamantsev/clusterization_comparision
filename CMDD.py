import itertools

import numpy as np
from rtree import index

UNCLASSIFIED_ID = None


class DataSet:
    def __init__(self, dataset, k, is_brute):
        self.result = [UNCLASSIFIED_ID] * dataset.shape[0]
        if is_brute:
            self.__init_distances_brute(dataset, k)
        else:
            self.__init_distances(dataset, k)

    def __init_distances(self, dataset, k):
        rtree_property = index.Property()
        rtree_property.dimension = dataset.shape[1]
        rtree_index = index.Index(properties=rtree_property)
        for i, point in enumerate(dataset):
            rtree_index.add(i, (*point, *point))

        data = [[]] * dataset.shape[0]
        for i, point in enumerate(dataset):

            k_nearest = list(rtree_index.nearest((*point, *point), k + 1))
            if k_nearest[0] == i:  # removes current point
                k_nearest = k_nearest[1:]
            k_nearest = k_nearest[:k]  # removes excess if present
            data_item = [[0, 0] for i in range(k)]
            data_item[0] = [k_nearest[0], np.linalg.norm(point - dataset[k_nearest[0]])]
            for j, point_identifier in list(enumerate(k_nearest))[1:]:
                data_item[j] = [
                    point_identifier,
                    data_item[j - 1][1] + np.linalg.norm(point - dataset[point_identifier])
                ]
            data[i] = data_item

        self.data = data

    def __init_distances_brute(self, dataset, k):
        data = [None] * dataset.shape[0]
        for i, point in enumerate(dataset):
            distances = [(j, np.linalg.norm(point - p_j)) for j, p_j in enumerate(dataset) if j != i]
            distances = list(itertools.islice(sorted(distances, key=lambda item: item[1]), k))

            data_item = [[0, 0] for i in range(k)]
            data_item[0] = [distances[0][0], np.linalg.norm(point - dataset[distances[0][0]])]
            for j, (point_identifier, distance) in list(enumerate(distances))[1:]:
                data_item[j] = [
                    point_identifier,
                    data_item[j - 1][1] + distance
                ]
            data[i] = data_item

        self.data = data

    def region_query(self, point_identifier):
        return [t[0] for t in self.data[point_identifier]]

    def distance(self, from_point_id, k):
        return self.data[from_point_id][k - 1][1]

    def get_class_id(self, point_id):
        return self.result[point_id]

    def change_cluster_ids(self, point_identifiers, cluster_id):
        for point_identifier in point_identifiers:
            self.result[point_identifier] = cluster_id


class CMDD:
    def __init__(self, k, minpts, is_brute=False):
        if minpts > k:
            raise Exception('Invalid arguments. k can not be greater then minpts!')
        self.k = k
        self.minpts = minpts
        self.dataset = None
        self.is_brute = is_brute

    def fit(self, X):
        objects_count = X.shape[0]
        dataset = DataSet(X, self.k, self.is_brute)

        # calculate densities
        id_density = []
        for i in range(objects_count):
            id_density.append((
                i,
                dataset.distance(i, self.k)
            ))

        cluster_id = 1
        id_density_sorted = list(sorted(id_density, key=lambda t: t[1]))
        for point_identifier, point_k_den in id_density_sorted:
            if dataset.get_class_id(point_identifier) == UNCLASSIFIED_ID:
                self.expand_cluster(dataset, point_identifier, cluster_id, point_k_den)
                cluster_id += 1

        self.dataset = dataset

    def fit_predict(self, X):
        self.fit(X)

        return self.dataset.result

    def expand_cluster(self, dataset, point_identifier, cluster_id, point_k_den):
        # k_nn = dataset.region_query(point_identifier)
        seeds = [point_identifier]

        dataset.change_cluster_ids(seeds, cluster_id)
        while len(seeds) > 0:
            current_identifier = seeds.pop(0)
            result = dataset.region_query(current_identifier)
            if dataset.distance(current_identifier, self.minpts) <= point_k_den:
                for i, k_neighbour_id in enumerate(result):
                    if dataset.distance(current_identifier, i + 1) > point_k_den:
                        break
                    if dataset.get_class_id(k_neighbour_id) == UNCLASSIFIED_ID:
                        seeds.append(k_neighbour_id)
                        dataset.change_cluster_ids([k_neighbour_id], cluster_id)
