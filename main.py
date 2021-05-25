import time
from itertools import cycle, islice

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn import datasets
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from dbscan import dbscan
from cbldp import cbldp

DBSCAN = 'DBSCAN'
CBLDP = 'CBLDP'


def print_measures(labels_true, labels_predicted):
    true_classes = list(set(labels_true))
    predicted_classes = list(set(labels_predicted))
    true_classes.sort()
    predicted_classes.sort()
    cost_matrix = []
    for true_class in true_classes:
        cost_row = []
        for predicted_class in predicted_classes:
            true_class_binary = [1 if c == true_class else 0 for c in labels_true]
            predicted_class_binary = [1 if c == predicted_class else 0 for c in labels_predicted]
            precision = np.dot(true_class_binary, predicted_class_binary) / np.sum(predicted_class_binary)
            recall = np.dot(true_class_binary, predicted_class_binary) / np.sum(true_class_binary)
            if precision == recall == 0:
                cost_row.append(0)
            else:
                cost_row.append(
                    -2 * precision * recall / (precision + recall)
                )
        cost_matrix.append(cost_row)

    cost_matrix = np.array(cost_matrix)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)  # Hungarian algorithm
    true_predicted_mapping = {true_classes[true_idx]: predicted_classes[predicted_idx]
                              for true_idx, predicted_idx in zip(row_ind, col_ind)
                              }
    f_measure = -1 * cost_matrix[row_ind, col_ind].sum() / len(true_predicted_mapping)
    accuracy = np.sum([1 if y_t in true_predicted_mapping and true_predicted_mapping[y_t] == y_p else 0
                       for y_t, y_p in zip(labels_true, labels_predicted)]) / len(labels_true)
    # mapped = [true_predicted_mapping[l] if l in true_predicted_mapping else l for l in labels_true]
    print('Accuracy: %.3f' % accuracy)
    print('F-measure: %.3f' % f_measure)
    # print('F-measure s: %.3f' % metrics.f1_score(labels_true, mapped, average='micro'))
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels_predicted))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels_predicted))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels_predicted))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels_predicted))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels_predicted))


np.random.seed(0)

n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.6,
                                      noise=.1)
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

compound_full = pd.read_csv('./datasets/Compound.txt', sep='	').to_numpy()
compound = compound_full[:, :2], compound_full[:, 2]

jain_full = pd.read_csv('./datasets/jain.txt', sep='	').to_numpy()
jain = jain_full[:, :2], jain_full[:, 2]

plt.figure(figsize=(9 * 2 + 3, 13))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.95, wspace=.05,
                    hspace=.01)

plot_num = 1

default_base = {
    DBSCAN: {
        'eps': .3,
        'min_samples': 50,
    },
    CBLDP: {
        'k': 40,
        'minpts': 20,
    }
}

datasets = [
    ('Noisy circles', noisy_circles, {
        DBSCAN: {},
        CBLDP: {},
    }),
    # ('Noisy moons', noisy_moons, {
    #     DBSCAN: {},
    #     CBLDP: {},
    # }),
    # ('Varied', varied, {
    #     DBSCAN: {'eps': .18, 'min_samples': 5},
    #     CBLDP: {},
    #
    # }),
    # ('Anisotropy', aniso, {
    #     DBSCAN: {'eps': .15, 'min_samples': 20},
    #     CBLDP: { },
    # }),
    # ('Blobs', blobs, {
    #     DBSCAN: {},
    #     CBLDP: {},
    # }),
    # ('No structure', no_structure, {
    #     DBSCAN: {},
    #     CBLDP: {},
    # })
    ('Compound', compound, {
        DBSCAN: {},
        CBLDP: {
            'k': 9,
            'minpts': 4,
        },
    }),
    ('Jain', jain, {
        DBSCAN: {},
        CBLDP: {
            'k': 15,
            'minpts': 4,
        },
    })
]
clustering_algorithms = [
    (DBSCAN, dbscan),
    (CBLDP, cbldp)
]

for i_dataset, (title, dataset, algo_params) in enumerate(datasets):
    print(title)

    X, y = dataset
    X = StandardScaler().fit_transform(X)

    for name, algorithm in clustering_algorithms:
        params = default_base[name].copy()
        params.update(algo_params[name])
        t0 = time.perf_counter()
        y_pred = algorithm(X, **params)
        t1 = time.perf_counter()

        print(name)
        print_measures(y, y_pred)

        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)

        colors = list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                    '#f781bf', '#a65628', '#984ea3',
                                    '#999999', '#e41a1c', '#dede00']),
                             int(max(y_pred) + 1)))
        # add black color for outliers (if any)
        colors.append("#000000")
        colors = np.array(colors)

        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1
    print()

plt.show()
