import sys
import csv
import statistics
from collections import defaultdict
import pickle
from multiprocessing import Pool, cpu_count
import os

import numpy as np
import scipy.spatial.distance
from sklearn.decomposition import TruncatedSVD
import Levenshtein
import umap

import mulres.config
import mulres.utils


CONCEPTS = '''
I you we one two person fish dog louse tree leaf skin blood bone horn ear eye
nose tooth tongue knee hand breast liver drink see hear die come sun star
water stone fire path mountain night full new name
'''.split()

def normalized_ed(s1, s2):
    return Levenshtein.distance(s1, s2) / max(len(s1), len(s2))

def mean_normalized_ed(forms1, forms2):
    # forms1 and forms2 are all the forms for a given concept, typically a
    # single form but may contain more.
    return statistics.mean((normalized_ed(s1, s2) for s1 in forms1
                                                  for s2 in forms2))

def language_distance(concepts1, concepts2):
    # Mean distance between equivalent forms
    dist = statistics.mean((mean_normalized_ed(forms1, forms2)
                            for forms1, forms2 in zip(concepts1, concepts2)
                            if forms1 and forms2))
    # Mean distance between unrelated forms
    base = statistics.mean((mean_normalized_ed(forms1, forms2)
                            for i, forms1 in enumerate(concepts1)
                            for j, forms2 in enumerate(concepts2)
                            if (i != j) and forms1 and forms2))
    return dist / base


def write_embeddings(filename, labels, m):
    with open(filename, 'w', encoding='utf-8') as f:
        print(len(labels), m.shape[1], file=f)
        for label, row in zip(labels, m):
            print(' '.join([label] + [str(round(x, 5)) for x in row]), file=f)


def main():
    text_info = mulres.utils.load_resources_table()
    isos_limit = {info['iso'] for name, info in text_info.items()
                  if info['preferred_source'] == 'no'
                  and os.path.exists(mulres.utils.encoded_filename(name))}

    print('Attempting to find data for', len(isos_limit), 'languages')

    os.makedirs(mulres.config.asjp_embeddings_path, exist_ok=True)

    full = os.path.join(mulres.config.asjp_embeddings_path, 'full.pickle')
    embeddings_umap = os.path.join(mulres.config.asjp_embeddings_path,
            'asjp_umap.vec')
    embeddings_svd = os.path.join(mulres.config.asjp_embeddings_path,
            'asjp_svd.vec')

    if not os.path.exists(full):
        iso_concepts = defaultdict(lambda: [set() for _ in CONCEPTS])
        with open(mulres.config.asjp_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                iso = row['iso']
                if iso and len(iso) == 3 and iso.isalpha():
                    for i, concept in enumerate(CONCEPTS):
                        forms = set(map(str.strip, row[concept].split(', ')))-{''}
                        iso_concepts[iso][i] |= forms

        isos = sorted(iso for iso, concept_forms in iso_concepts.items()
                          if sum((not forms) for forms in concept_forms) <= 10
                            and ((isos_limit is None) or (iso in isos_limit)))

        print(len(isos), 'languages with sufficient data')

        isos_forms = [iso_concepts[iso] for iso in isos]

        pairs = [(forms1, forms2) for i, forms1 in enumerate(isos_forms)
                                  for forms2 in isos_forms[i+1:]]

        print('Computing pairwise distance matrix ...')

        batch_size = len(pairs) // cpu_count()
        print(batch_size, 'pairs per core, ETA', batch_size//25, 'seconds')

        with Pool() as p:
            d = np.array(p.starmap(language_distance, pairs, batch_size))

        #d = np.array([language_distance(forms1, forms2)
        #              for i, forms1 in enumerate(isos_forms)
        #              for forms2 in isos_forms[:i]])

        with open(full, 'wb') as f:
            pickle.dump(isos, f)
            pickle.dump(d, f)

    else:
        print('Loading', full)
        with open(full, 'rb') as f:
            isos = pickle.load(f)
            d = pickle.load(f)

    m = scipy.spatial.distance.squareform(d)

    print('%dx%d matrix to reduce' % m.shape)

    transform = umap.UMAP(n_components=100, init='random')
    m_umap = transform.fit_transform(m)

    svd = TruncatedSVD(n_components=100)
    svd.fit(m)
    m_svd = svd.transform(m)

    write_embeddings(embeddings_umap, isos, m_umap)
    write_embeddings(embeddings_svd, isos, m_svd)

if __name__ == '__main__': main()

