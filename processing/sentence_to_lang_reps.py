# Script to create lanugage representations from sentence representations
#
# Then input files are assumed to contain gziped pickles containing dicts of
# the form:
#
#   { verse_id: np.ndarray }
#
# Where each array is a sentence representation of the given verse.

import sys
import pickle
import gzip
import os
from multiprocessing import Pool
from operator import itemgetter

import numpy as np

def get_doculect_representation(filename):
    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f)

    return np.mean(list(data.values()), axis=0)


def get_name(filename):
    return os.path.basename(filename).split('.')[0]


def get_embeddings(filenames):
    names = list(map(get_name, filenames))

    with Pool() as p:
        vectors = p.map(get_doculect_representation, filenames)

    return dict(zip(names, vectors))

def write_embeddings(filename, e):
    with open(filename, 'w', encoding='utf-8') as f:
        table = sorted(e.items(), key=itemgetter(0))
        print(f'{len(table)} {table[0][1].shape[0]}', file=f)
        for name, v in table:
            print(' '.join([name] + [f'{x:.5f}' for x in v]), file=f)

def main():
    output = sys.argv[1]
    assert not os.path.exists(output)
    filenames = sys.argv[2:]
    e = get_embeddings(filenames)
    write_embeddings(output, e)


if __name__ == '__main__':
    main()

