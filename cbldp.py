from rtree import index
import numpy as np

UNCLASSIFIED_ID = None


class DataSet:
    def __init__(self, dataset, k):
        self.dataset = dataset  # todo is it required?
        self.result = [UNCLASSIFIED_ID] * dataset.shape[0]
        self.__init_distances(k)

    def __init_distances(self, k):
        rtree_property = index.Property()
        rtree_property.dimension = self.dataset.shape[1]
        rtree_index = index.Index(properties=rtree_property)
        for i, point in enumerate(self.dataset):
            rtree_index.add(i, (*point, *point))

        data = [None] * self.dataset.shape[0]
        for i, point in enumerate(self.dataset):
            k_nearest = list(rtree_index.nearest((*point, *point), k + 1))[:k + 1]  # remove excess if present
            data_item = [[0, 0]] * (k + 1)
            data_item[0] = [k_nearest[0], 0]
            for j, point_identifier in list(enumerate(k_nearest))[1:]:
                data_item[j] = [
                    point_identifier,
                    data_item[j - 1][1] + np.linalg.norm(point - self.dataset[point_identifier])
                ]
            data[i] = data_item

        self.data = data

    def region_query(self, point_identifier):
        return [t[0] for t in self.data[point_identifier]]

    def distance(self, from_point_id, k):
        return self.data[from_point_id][k][1]

    def get_class_id(self, point_id):
        return self.result[point_id]

    def change_cluster_ids(self, point_identifiers, cluster_id):
        for point_identifier in point_identifiers:
            self.result[point_identifier] = cluster_id


def cbldp(raw_dataset, k=40, minpts=20):
    if minpts > k:
        raise Exception('Invalid arguments for cbldp function')

    objects_count = raw_dataset.shape[0]
    dataset = DataSet(raw_dataset, k)

    # calculate densities
    id_density = []
    for i in range(objects_count):
        id_density.append((
            i,
            dataset.distance(i, k)
        ))

    cluster_id = 1
    id_density_sorted = list(sorted(id_density, key=lambda t: t[1]))
    for point_identifier, point_k_den in id_density_sorted:
        if dataset.get_class_id(point_identifier) == UNCLASSIFIED_ID:
            expand_cluster(dataset, point_identifier, cluster_id, point_k_den, minpts)
            cluster_id += 1

    return dataset.result


def expand_cluster(dataset, point_identifier, cluster_id, point_k_den, minpts):
    seeds = dataset.region_query(point_identifier)
    dataset.change_cluster_ids([*seeds], cluster_id)
    while len(seeds) > 0:
        current_identifier = seeds.pop(0)
        result = dataset.region_query(current_identifier)
        if dataset.distance(current_identifier, minpts) <= point_k_den:
            for i, k_neighbour_id in enumerate(result):
                if dataset.distance(current_identifier, i) > point_k_den:
                    break
                if dataset.get_class_id(k_neighbour_id) == UNCLASSIFIED_ID:
                    seeds.append(k_neighbour_id)
                    dataset.change_cluster_ids([k_neighbour_id], cluster_id)
