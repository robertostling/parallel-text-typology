import sys
import csv
import gzip
import pickle
from multiprocessing import Pool, cpu_count
from collections import Counter, defaultdict
import os
import statistics
from operator import itemgetter
import pprint

import Levenshtein
import numpy as np
import scipy.spatial.distance
import umap
from sklearn.decomposition import TruncatedSVD

from create_asjp_vectors import write_embeddings, language_distance

import mulres.config
import mulres.utils
import mulres.transliterate

# TODO: make create_asjp_vectors.py use this function too, or move it there
# rather
def get_distance_matrix(forms_table):
    pairs = [(concepts1, concepts2) for i, concepts1 in enumerate(forms_table)
             for concepts2 in forms_table[i+1:]]

    batch_size = max(1, len(pairs) // cpu_count())
    print(len(pairs), 'pairs of languages')
    print(batch_size, 'pairs per core, computing pairwise distances...')
    with Pool() as p:
        d = np.array(p.starmap(language_distance, pairs, batch_size))

    return d


def get_forms(target_text, use_sources='eng-x-bible-newworld2013'):
    """Get aligned forms of source text lemmas

    Args:
        target_text: text_table entry for the target text
        use_sources: if None, include all sources, otherwise this is a set
                     of source text names to include in the table

    Returns:
        dict { (source_name, source_lemma): target_form }
    """
    aligned_filename = mulres.utils.aligned_filename(target_text['name'])
    lemma_translation = {}
    with gzip.open(aligned_filename, 'rb') as f:
        source_names = pickle.load(f)
        candidates_list = pickle.load(f)
        for source_name, lemma_candidates in zip(source_names, candidates_list):
            if use_sources is not None and source_name not in use_sources:
                continue
            for lemma, candidates in lemma_candidates.items():
                if not candidates: continue
                form, bayes_factor, _ = candidates[0]
                # Names are not very informative, and most of the source
                # languages capitalize them, so we use this heuristic to avoid
                # them.
                if lemma[0].isupper(): continue
                if bayes_factor > 50:
                    # Ignore word boundary markers
                    form = form.replace('#', '')
                    lemma_translation[(source_name, lemma)] = form
    return lemma_translation


def transliterate(s):
    if not s: return s
    return mulres.transliterate.remove_distinctions(
            mulres.transliterate.normalize(s))


def generate_forms(csv_filename):
    text_table = mulres.utils.load_resources_table()

    os.makedirs(mulres.config.lexical_path, exist_ok=True)

    target_texts = [
            info for name, info in text_table.items()
            if os.path.exists(mulres.utils.aligned_filename(name))]

    print(len(target_texts), 'texts to compute lexical similarity between')

    with Pool() as p:
        target_forms = p.map(get_forms, target_texts)

    source_lemma_count = Counter()

    for info, forms in zip(target_texts, target_forms):
        source_lemma_count.update(forms.keys())

    threshold = 0.75*len(target_forms)

    common_lemmas = sorted(
            source_lemma
            for source_lemma, n in source_lemma_count.items()
            if n >= threshold)

    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        for info, forms in zip(target_texts, target_forms):
            fields = [transliterate(forms.get(source_lemma, ''))
                      for source_lemma in common_lemmas]
            writer.writerow([info['name']] + fields)

    #pprint.pprint(source_lemma_count)
    #pprint.pprint(common_lemmas)

def main():
    csv_filename = os.path.join(mulres.config.lexical_path, 'forms.csv')
    matrix_filename = os.path.join(mulres.config.lexical_path, 'matrix.pickle')
    umap_filename = os.path.join(mulres.config.lexical_path, 'lexical_umap.vec')
    svd_filename = os.path.join(mulres.config.lexical_path, 'lexical_svd.vec')

    if not os.path.exists(csv_filename):
        print('Generating', csv_filename)
        generate_forms(csv_filename)

    if not os.path.exists(matrix_filename):
        print('Generating', matrix_filename)
        with open(csv_filename, newline='') as f:
            reader = csv.reader(f)
            forms_table = []
            names = []
            for row in reader:
                names.append(row[0])
                forms_table.append(
                        [[] if not form else [form] for form in row[1:]])

        d = get_distance_matrix(forms_table)
        del forms_table
        with open(matrix_filename, 'wb') as f:
            pickle.dump(names, f)
            pickle.dump(d, f)
    else:
        with open(matrix_filename, 'rb') as f:
            names = pickle.load(f)
            d = pickle.load(f)


    m = scipy.spatial.distance.squareform(d)

    #if m.shape[0] < 12:
    #    print(names)
    #    print(m)

    print('Reducing %dx%d distance matrix with UMAP' % m.shape)
    transform = umap.UMAP(n_components=100, init='random')
    m_umap = transform.fit_transform(m)
    write_embeddings(umap_filename, names, m_umap)

    print('Reducing %dx%d distance matrix with SVD' % m.shape)
    svd = TruncatedSVD(n_components=100)
    svd.fit(m)
    m_svd = svd.transform(m)
    write_embeddings(svd_filename, names, m_svd)



if __name__ == '__main__': main()

