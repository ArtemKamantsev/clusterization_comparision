import random

import numpy as np
import pandas as pd
from sklearn import datasets


# blobs with varied variances
def get_varied_dataset_generator(n_samples):
    INT_MAX = 2 ** 31 - 1

    def generator():
        return datasets.make_blobs(n_samples=n_samples,
                                   cluster_std=[1.0, 2, 0.5],
                                   centers=[[-8.95, - 5.46],
                                            [-4.59, 0.09],
                                            [1.94, 0.51]],
                                   random_state=random.randint(0, INT_MAX))

    return generator


def get_jain_dataset():
    jain_full = pd.read_csv('./datasets/jain.txt', sep='	').to_numpy()
    jain = jain_full[:, :2], jain_full[:, 2].astype(np.int32)

    return jain


def get_compound_dataset():
    jain_full = pd.read_csv('./datasets/Compound.txt', sep='	').to_numpy()
    jain = jain_full[:, :2], jain_full[:, 2].astype(np.int32)

    return jain

def get_cancer_dataset():
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

    return cancer
