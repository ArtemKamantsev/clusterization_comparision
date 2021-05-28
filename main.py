import time
from itertools import cycle, islice

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn import datasets
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from CMDD import CMDD
from Spectacl import Spectacl

ORIGINAL_title = 'Original'
DBSCAN_title = 'DBSCAN'
CMDD_title = 'CMDD'
Spectacl_title = 'SpectaCL'
Spectacl_Normalized_title = 'SpectaCL N'


def get_cores(labels_true, labels_predicted):
    # true_classes = list(set(labels_true))
    # predicted_classes = list(set(labels_predicted))
    # true_classes.sort()
    # predicted_classes.sort()
    # cost_matrix = []
    # for true_class in true_classes:
    #     cost_row = []
    #     for predicted_class in predicted_classes:
    #         true_class_binary = [1 if c == true_class else 0 for c in labels_true]
    #         predicted_class_binary = [1 if c == predicted_class else 0 for c in labels_predicted]
    #         precision = np.dot(true_class_binary, predicted_class_binary) / np.sum(predicted_class_binary)
    #         recall = np.dot(true_class_binary, predicted_class_binary) / np.sum(true_class_binary)
    #         if precision == recall == 0:
    #             cost_row.append(0)
    #         else:
    #             cost_row.append(
    #                 -2 * precision * recall / (precision + recall)
    #             )
    #     cost_matrix.append(cost_row)
    #
    # cost_matrix = np.array(cost_matrix)
    # row_ind, col_ind = linear_sum_assignment(cost_matrix)  # Hungarian algorithm
    # true_predicted_mapping = {true_classes[true_idx]: predicted_classes[predicted_idx]
    #                           for true_idx, predicted_idx in zip(row_ind, col_ind)
    #                           }
    # mapped = [true_predicted_mapping[label] if label in true_predicted_mapping else label for label in labels_true]
    # print('F-measure: %.3f' % metrics.f1_score(mapped, labels_predicted, average='micro'))
    # print("Normalized Mutual Information: %0.3f"
    #       % metrics.normalized_mutual_info_score(labels_true, labels_predicted))
    return {
        'ari': metrics.adjusted_rand_score(labels_true, labels_predicted)
    }


np.random.seed(0)

n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.6,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), np.zeros((n_samples,), dtype=int)

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state, )


def parse(v):
    try:
        return float(v)
    except:
        return 0


cancer_full = pd.read_csv('./datasets/breast-cancer-wisconsin.data', sep=',', converters={
    6: parse
}).to_numpy()
cancel_filtered = np.array(list(filter(lambda r: r[6] != 0, cancer_full)))
cancer = cancel_filtered[:, 1:-1], cancel_filtered[:, -1].astype(np.int32)

jain_full = pd.read_csv('./datasets/jain.txt', sep='	').to_numpy()
jain = jain_full[:, :2], jain_full[:, 2].astype(np.int32)

default_base = {
    DBSCAN_title: {
        'eps': .3,
        'min_samples': 50,
    },
    CMDD_title: {
        'k': 40,
        'minpts': 20,
    },
    Spectacl_title: {},
    Spectacl_Normalized_title: {},
}

datasets = {
    # ('Noisy circles', noisy_circles, {
    #     DBSCAN_title: {},
    #     CMDD_title: {},
    #     Spectacl_title: {
    #         'n_clusters': 2,
    #         'epsilon': 0.2,
    #     }
    # }),
    # ('Noisy moons', noisy_moons, {
    #     DBSCAN_title: {},
    #     CMDD_title: {},
    #     Spectacl_title: {}
    # }),
    'Varied': (varied, {
        DBSCAN_title: {'eps': .3, 'min_samples': 60},
        CMDD_title: {'k': 30, 'minpts': 12},
        Spectacl_title: {'n_clusters': 3, 'epsilon': .6},
        Spectacl_Normalized_title: {'n_clusters': 3, 'epsilon': .6, 'normalize_adjacency': True}
    }),
    # ('Anisotropy', aniso, {
    #     DBSCAN_title: {'eps': .15, 'min_samples': 20},
    #     CMDD_title: {},
    #     Spectacl_title: {}
    # }),
    # ('Blobs', blobs, {
    #     DBSCAN_title: {},
    #     CMDD_title: {},
    #     Spectacl_title: {},
    # }),
    # ('No structure', no_structure, {
    #     DBSCAN_title: {},
    #     CMDD_title: {},
    #     Spectacl_title: {},
    # }),
    # 'Jain': (jain, {
    #     DBSCAN_title: {},
    #     CMDD_title: {
    #         'k': 13,
    #         'minpts': 4,
    #     },
    #     Spectacl_title: {}
    # }),
    # 'Cancer': (cancer, {
    #     DBSCAN_title: {
    #         'eps': 1.55,
    #         'min_samples': 11
    #     },
    #     CMDD_title: {'k': 75, 'minpts': 12},
    #     Spectacl_title: {'n_clusters': 3, 'epsilon': 3.01},
    #     Spectacl_Normalized_title: {'n_clusters': 2, 'epsilon': 2.8, 'normalize_adjacency': True}
    # })
}
clustering_algorithms = {
    DBSCAN_title: DBSCAN,
    CMDD_title: CMDD,
    Spectacl_title: Spectacl,
    Spectacl_Normalized_title: Spectacl,
    ORIGINAL_title: None,
}


def fit():
    for k in range(1, 20):
        for m in range(1, 20):
            if m > k:
                break
            X, y = compound
            X = StandardScaler().fit_transform(X)
            model = CMDD(k=k, minpts=m)
            y_pred = model.fit_predict(X)
            # print(e, end=' ')
            scores = get_cores(y, y_pred)
            print('ARI: %.3f, %d, %d' % (scores['ari'], k, m))


def main():
    rows_count = len(datasets)
    cols_count = len(clustering_algorithms)
    fig = make_subplots(
        rows=rows_count,
        cols=cols_count,
        column_titles=list(clustering_algorithms.keys()),
        row_titles=list(datasets.keys()),
    )
    colors_range = ['#377eb8', '#ff7f00', '#4daf4a',
                    '#f781bf', '#a65628', '#984ea3',
                    '#999999', '#dede00']

    def output(X, y, y_predicted, row, column):
        scores = get_cores(y, y_predicted)
        print('ARI %0.3f' % scores['ari'])

        if X.shape[1] != 2:
            X = PCA(n_components=2).fit_transform(X)

        colors = list(islice(cycle(colors_range), int(max(y_predicted) + 1)))

        # add black color for outliers (if any)
        colors.append("#000000")
        colors = np.array(colors)

        fig.add_trace(
            go.Scatter(
                x=X[:, 0],
                y=X[:, 1],
                mode='markers',
                marker=dict(
                    color=colors[y_predicted],
                    size=4,
                ),
            ),
            row=row + 1, col=column + 1
        )
        fig.update_xaxes(title_text=('ARI score: %.3f' % scores["ari"]), row=row + 1, col=column + 1)

    for i_dataset, (title, (dataset, algo_params)) in enumerate(datasets.items()):
        print(title)

        X, y = dataset
        X = StandardScaler().fit_transform(X)

        for i_algorithm, key_algo in enumerate(clustering_algorithms):
            if key_algo != ORIGINAL_title:
                params = default_base[key_algo].copy()
                params.update(algo_params[key_algo])
                t0 = time.perf_counter()
                model = clustering_algorithms[key_algo](**params)
                y_pred = model.fit_predict(X)
                t1 = time.perf_counter()

                print(key_algo)
                output(X, y, y_pred, i_dataset, i_algorithm)
                print()
            else:
                output(X, y, y, i_dataset, i_algorithm)

    fig.update_layout({'showlegend': False})
    fig.show()


# main()
fit()
