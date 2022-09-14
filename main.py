from itertools import cycle, islice
from statistics import mean, variance

import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from CMDD import CMDD
from Spectacl import Spectacl
from datasets import get_varied_dataset_generator, get_cancer_dataset

ORIGINAL_title = 'Оригінал'
DBSCAN_title = 'DBSCAN'
CMDD_title = 'CMDD'
CMDD_brute_title = 'CMDD_brute'
Spectacl_title = 'SpectACl'

varied_title = 'Синтетичні дані'
cancer_title = 'Breast Cancer Wisconsin'

np.random.seed(0)

random_state = 42
n_samples = 1500


def get_ami_score(labels_true, labels_predicted):
    return metrics.adjusted_rand_score(labels_true, labels_predicted)


data = {
    varied_title: (get_varied_dataset_generator(n_samples=n_samples), {
        DBSCAN_title: {'eps': .3, 'min_samples': 85},
        CMDD_title: {'k': 20, 'minpts': 7},
        Spectacl_title: {'n_clusters': 3, 'epsilon': .5},
    }),
    cancer_title: (get_cancer_dataset, {
        DBSCAN_title: {
            'eps': 1.55,
            'min_samples': 11
        },
        CMDD_title: {'k': 71, 'minpts': 12},
        Spectacl_title: {'n_clusters': 2, 'epsilon': 3.01},
    })
}

clustering_algorithms = {
    DBSCAN_title: DBSCAN,
    CMDD_title: CMDD,
    Spectacl_title: Spectacl,
    ORIGINAL_title: None,
}


def fit():
    for k in range(12, 18):
        score = 0
        iterations = 1
        for m in range(3, 8):
            if m > k:
                break
            X, y = get_varied_dataset_generator(n_samples)()
            X = StandardScaler().fit_transform(X)
            model = CMDD(minpts=m, k=k)
            y_pred = model.fit_predict(X)
            # score += get_ami_score(y, y_pred)
            score = get_ami_score(y, y_pred)
            print('AMI: %.3f, %d %d' % (score, k, m))

        # print('AMI: %.3f, %d %d' % (score / iterations, k, m))


def plot():
    rows_count = len(data)
    cols_count = len(clustering_algorithms)
    fig = make_subplots(
        rows=rows_count,
        cols=cols_count,
        column_titles=list(clustering_algorithms.keys()),
        row_titles=list(data.keys()),
    )
    colors_range = ['#377eb8', '#ff7f00', '#4daf4a',
                    '#f781bf', '#a65628', '#984ea3',
                    '#999999', '#dede00']
    transformer = PCA(n_components=2, random_state=random_state)

    def output(X, y, y_predicted, row, column):
        score = get_ami_score(y, y_predicted)
        print('AMI %0.3f' % score)

        if X.shape[1] != 2:
            X = transformer.fit_transform(X)

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
                    size=3,
                ),
            ),
            row=row + 1, col=column + 1
        )
        fig.update_xaxes(title_text=('AMI: %.3f' % score), row=row + 1, col=column + 1)

    for i_dataset, (title, (dataset_getter, algo_params)) in enumerate(data.items()):
        print(title)

        X, y = dataset_getter()
        X = StandardScaler().fit_transform(X)

        for i_algorithm, key_algo in enumerate(clustering_algorithms):
            if key_algo != ORIGINAL_title:
                model = clustering_algorithms[key_algo](**algo_params[key_algo])
                y_pred = model.fit_predict(X)
                # score = get_ami_score(y, y_pred)
                # if score < 0.8:
                #     print(key_algo)
                #     output(X, y, y_pred, i_dataset, i_algorithm)
                #     print()
                #     break
                # else:
                #     print(score)
                #     X, y = dataset_getter()
                #     X = StandardScaler().fit_transform(X)
                print(key_algo)
                output(X, y, y_pred, i_dataset, i_algorithm)
                print()
            else:
                output(X, y, y, i_dataset, i_algorithm)

    fig.update_layout({'showlegend': False})
    # plotly.offline.plot(fig, filename='clustering.html', show_link=False)
    fig.show()


def estimate_accuracy(data_dict, algo_dict, iterations):
    for i_dataset, (title, (dataset_getter, algo_params)) in enumerate(data_dict.items()):
        print(title)
        scores = [[] for i in range(len(algo_dict))]

        for i in range(iterations):
            print('iteration %d' % i)
            X, y = dataset_getter()
            X = StandardScaler().fit_transform(X)

            for i_algorithm, key_algo in enumerate(algo_dict):
                model = clustering_algorithms[key_algo](**algo_params[key_algo])
                y_pred = model.fit_predict(X)
                scores[i_algorithm].append(get_ami_score(y, y_pred))

        for i_algorithm, key_algo in enumerate(algo_dict):
            print(
                '%s: mean - %.3f, variance - %f' % (key_algo, mean(scores[i_algorithm]), variance(scores[i_algorithm])))


plot()
# fit()
# estimate_accuracy({
#     varied_title: data[varied_title],
# }, {
#     DBSCAN_title: DBSCAN,
#     CMDD_title: CMDD,
#     Spectacl_title: Spectacl,
# }, 100)
