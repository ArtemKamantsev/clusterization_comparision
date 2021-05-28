import numpy as np
import scipy
from scipy.sparse import linalg
from sklearn.cluster import KMeans
from sklearn.neighbors import radius_neighbors_graph


class Spectacl:
    def __init__(self,
                 affinity="radius_neighbors",
                 n_clusters=2,
                 epsilon=1.0,
                 normalize_adjacency=False,
                 clusterer=None,
                 random_state=42):
        # Assign the parameters to the object.
        self.epsilon = epsilon
        self.normalize_adjacency = normalize_adjacency
        self.n_clusters = n_clusters
        self.affinity = affinity

        # Manage the base clusterer
        if clusterer is None:
            self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=random_state)
        else:
            self.clusterer = clusterer

        # Predefine some properties, which will be updated, when the `fit` method has been called.
        self.Lambda = None
        self.V = None
        self.labels_ = None
        self.W = None
        self.Y = None

        self.__check_params__()

    def __check_params__(self):
        '''
        Checks, whether the class instance has been parametrized correctly.
        '''
        supported_affinities = ["radius_neighbors", "precomputed"]
        if self.affinity not in supported_affinities:
            raise ValueError("Illegal affinity supplied: {0}".format(supported_affinities))
        if self.affinity == "radius_neighbors" and self.epsilon < 0.0:
            raise ValueError("Illegal epsilon radius specified: {0}".format(str(self.epsilon)))
        if self.n_clusters < 2:
            raise ValueError("Illegal number of clusters specified: {0}".format(str(self.n_clusters)))
        if not callable(self.clusterer.fit_predict):
            raise ValueError("The provided base clusterer has no fit_predict method, which may be called.")

    def fit(self, X):
        '''
        Clusters the dataset or the precomputed adjacency matrix using the SpectAcl method.
        :param X: The dataset or the adjacency matrix to use.
        '''
        if self.affinity == "radius_neighbors":
            X = radius_neighbors_graph(X, radius=self.epsilon)
        if self.normalize_adjacency:
            d = np.sum(X, axis=1)
            d[d == 0] = 1
            d = np.power(d, -0.5)
            D = scipy.sparse.diags(np.squeeze(np.asarray(d)))
            X = D @ X @ D

        Lambda, V = linalg.eigsh(X, k=50, which="LM")
        Lambda, V = np.absolute(Lambda), np.absolute(V)

        base_labels = self.clusterer.fit_predict(V * np.power(Lambda, 0.5))

        # Create indicator matrix.
        Y = np.zeros((len(base_labels), self.n_clusters))
        for i in range(0, len(base_labels)):
            Y[i, base_labels[i]] = 1

        self.Lambda = Lambda
        self.V = V
        self.labels_ = base_labels
        self.W = X
        self.Y = Y

    def fit_predict(self, X):
        '''
        Clusters the dataset or the precomputed adjacency matrix using the SpectAcl method.

        In addition to the `fit` method, the cluster assignment vector is returned.

        :param X: The dataset or the adjacency matrix to use.
        :return: A vector of labels, representing the cluster assignment for each point.
        '''
        self.fit(X)
        return self.labels_
