from sklearn import cluster


def dbscan(dataset, eps=.3, min_samples=20):
    model = cluster.DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(dataset)

    return model.labels_
