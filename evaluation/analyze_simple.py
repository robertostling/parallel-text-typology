# Tool to analyze the .tsv outputs of simple_language_vectors.py
# Example:
#
# python3 evaluation/analyze_simple.py log-balanced-l21e-3.tsv \
#       | tee log-balanced-l21e-3.tsv.table
# 
# This will create log-balanced-l21e-3.tsv.predictions containing a
# tab-separated text file with the following columns:
#
# 1. name of vectors (.vec input filename without extension)
# 2. feature name (from URIEL, e.g. S_TEND_PREFIX)
# 3. source database for *training* ("uriel", or "proj" for Bible-projected)
# 4. doculect (Bible name, e.g. "eng-x-bible-kingjames")
# 5. ISO 639-3 code (first 3 letters of doculect name, e.g. "eng")
# 6. family (Glottolog family identifier, or language Glottocode for isolates)
# 7. count of 0-value classifications
# 8. count of 1-value classifications
# 9. URIEL feature value
#
# To get a binary classification comparable to column 9, compute 0 if column 7
# is higher than column 8, 1 otherwise.
#
# NOTE: the output printed to stdout can be used for sanity checks, but
# accuracy figures are not balanced and should not be used directly.
# Use the script analyze_predictions.py on the output of this script for a
# more careful analysis.

from collections import namedtuple, defaultdict, Counter
from operator import itemgetter
import sys
import itertools

import numpy as np

from langinfo.glottolog import Glottolog
from family import family_from_iso

from analyze_predictions import combined_f1


Entry = namedtuple('Entry',
        'name feature db doculect iso family sample train_size '
        'classifier baseline target')


def read_data(filename):
    table = []

    with open(filename) as f:
        for line in f:
            vecs, feature, db, doculect, family, sample, train_size, \
                    classifier, baseline, target = line.split()
            yield Entry(
                name=vecs.split('.')[0],
                feature=feature,
                db=db,
                doculect=doculect,
                iso=doculect[:3],
                family=family,
                sample=int(sample),
                train_size=int(train_size),
                classifier=float(classifier),
                baseline=float(baseline),
                target=int(target))


def get_paired_accuracies(table):
    # Assume sub-table with a single (name, feature, db) value
    sample_classifier_results = defaultdict(list)
    sample_baseline_results = defaultdict(list)
    samples = sorted({e.sample for e in table})
    assert samples == list(range(len(samples)))
    families = sorted({e.family for e in table})
    doculects = sorted({e.doculect for e in table})
    assert len(doculects) >= len(families)
    family_idx = {family:i for i,family in enumerate(families)}
    doculect_idx = {doculect:i for i,doculect in enumerate(doculects)}
    cols = len(doculects)
    classifier_acc = np.empty((len(samples), cols), dtype=float)
    baseline_acc = np.empty((len(samples), cols), dtype=float)
    weights = np.zeros_like(baseline_acc)
    sample_families = [Counter() for _ in samples]
    for e in table:
        sample_families[e.sample][e.family] += 1
    family_count = [len(counts) for counts in sample_families]

    # Family-weighted confusion matrices
    classifier_c = np.zeros((len(samples), 2, 2), dtype=float)
    baseline_c = np.zeros((len(samples), 2, 2), dtype=float)

    for e in table:
        # Compute binary score for a single language classification
        i = e.sample
        j = doculect_idx[e.doculect]
        w = 1.0 / (sample_families[i][e.family] * family_count[i])
        weights[i, j] = weights.shape[1] / (
                sample_families[i][e.family] * family_count[i])
        classifier_c[i, e.target, int(round(e.classifier))] += w
        baseline_c[i, e.target, int(round(e.baseline))] += w
        classifier_acc[i, j] = float(int(round(e.classifier)) == e.target)
        baseline_acc[i, j] = float(int(round(e.baseline)) == e.target)

    return dict(
            classifier=classifier_acc,
            baseline=baseline_acc,
            classifier_confusion=classifier_c,
            baseline_confusion=baseline_c,
            families=families,
            weights=weights)


def get_doculect_predictions(table, binarize=True):
    targets = {}
    predictions = defaultdict(list)
    for e in table:
        language = (e.doculect, e.iso, e.family)
        if language in targets:
            assert targets[language] == e.target
        else:
            targets[language] = e.target
        predictions[language].append(int(round(e.classifier)))
    if binarize:
        predictions = {
                language: int(round(np.mean(pred)))
                for language, pred in predictions.items()}
    else:
        predictions = {
                language: (pred.count(0), pred.count(1))
                for language, pred in predictions.items()}
    return predictions, targets


def analyze(table_iter, predf=None, sumf=None):
    for (name, feature, db), sub_table in itertools.groupby(
            table_iter, key=lambda e: (e.name, e.feature, e.db)):
        sub_table = list(sub_table)
        if predf is not None:
            pred, targets = get_doculect_predictions(sub_table, binarize=False)
            for (doculect, iso, family), (n_0, n_1) in sorted(
                    pred.items(),
                    key=itemgetter(0)):
                target = targets[(doculect, iso, family)]
                print(  f'{name}\t{feature}\t{db}\t'
                        f'{doculect}\t{iso}\t{family}\t'
                        f'{n_0}\t{n_1}\t{target}',
                        file=predf)
            predf.flush()
        r = get_paired_accuracies(sub_table)
        def family_mean(m):
            return np.mean(m * r['weights'], axis=1)
        print(name, feature, db)
        diff = r['classifier'] - r['baseline']
        #diff_dist = np.mean(diff, axis=1)
        diff_dist = family_mean(diff)
        diff_q = 100*np.percentile(diff_dist, [1, 5])
        #baseline_dist = np.mean(r['baseline'], axis=1)
        #classifier_dist = np.mean(r['classifier'], axis=1)
        baseline_dist = family_mean(r['baseline'])
        classifier_dist = family_mean(r['classifier'])
        b_q = 100*np.percentile(baseline_dist, [2.5, 50, 97.5, 95, 99])
        c_q = 100*np.percentile(classifier_dist, [2.5, 50, 97.5, 95, 99])
        better = np.mean(diff_dist > 0)
        train_size = np.mean([e.train_size for e in sub_table])
        if c_q[1] > b_q[4]: # diff_q[0] > 0:
            significance = '**'
        elif c_q[1] > b_q[3]: # diff_q[1] > 0:
            significance = '*'
        else:
            significance = ''
        baseline_f1s = [combined_f1(c) for c in r['baseline_confusion']]
        classifier_f1s = [combined_f1(c) for c in r['classifier_confusion']]
        bf1_q = 100*np.percentile(baseline_f1s, [1, 50, 99])
        cf1_q = 100*np.percentile(classifier_f1s, [1, 50, 99])
        print(f'{significance:3s}'
              f'N = {train_size:.1f}  P(c>b) = {100*better:.1f}%  '
              f'A_c ~ {c_q[1]:.1f} ({c_q[0]:.1f}/{c_q[2]:.1f})  '
              f'A_b ~ {b_q[1]:.1f} ({b_q[0]:.1f}/{b_q[2]:.1f})')
        print(f'   F1_c ~ {cf1_q[1]:.1f} ({cf1_q[0]:.1f}/{cf1_q[2]:.1f})'
              f'  F1_b ~ {bf1_q[1]:.1f} ({bf1_q[0]:.1f}/{bf1_q[2]:.1f})')
        if sumf is not None:
            print('\t'.join([
                name, feature, db,
                str(cf1_q[0]), str(cf1_q[1]), str(cf1_q[2]),
                str(bf1_q[0]), str(bf1_q[1]), str(bf1_q[2]),
                    ]), file=sumf, flush=True)
        print(flush=True)


if __name__ == '__main__':
    table_filename = sys.argv[1]
    with open(table_filename + '.predictions', 'w') as predf:
        with open(table_filename + '.summary', 'w') as sumf:
            analyze(read_data(table_filename), predf=predf, sumf=sumf)

