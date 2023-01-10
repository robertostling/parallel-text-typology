# This is the script used in the November 2022 version of the manuscript.
#
# time python3 evaluation/evaluate_language_vectors.py language-vectors/iso-bibles.txt log-new-balanced.tsv language-vectors/word_level.vec language-vectors/reinflect_verb.vec language-vectors/ostling2017_all.lang language-vectors/malaviya_mtvec.lang language-vectors/malaviya_mtcell.lang language-vectors/asjp_svd.lang language-vectors/character_level.vec language-vectors/lexical_svd.vec language-vectors/nmt_from_eng.vec language-vectors/nmt_to_eng.vec language-vectors/reinflect_all.vec language-vectors/reinflect_noun.vec language-vectors/word_encoding.vec

import pickle
import sys
import pprint
from collections import defaultdict, Counter, namedtuple
import random
import os.path
from multiprocessing import Pool
import time

import numpy as np
import scipy.stats
import sklearn.preprocessing
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import statsmodels.stats.api as sms
import statsmodels
from langinfo.glottolog import Glottolog


import mulres.config
import mulres.utils
import mulres.embeddings


# Ensure balanced (50/50) label distributions in classifiers.
balance = True
#balance = False

# Perform cross-validation in the training set to determine regularization
# for each individual classifier. This is typically a bad idea since it often
# results in too little regularization.
cross_validate = False
#cross_validate = True

# If cross_validate is False, use this amount of fixed L2 regularization:
fixed_regularization = 1e-3

# Use naive leave-one-out cross validation, simulating the proper in terms of
# sample sizes but always sampling from the full population.
# This is for comparison only, and will yield overly optimistic results.
naive_loocv = False
#naive_loocv = True

# If True, only languages where URIEL agrees with the annotation projection
# will be included. There is a theoretical risk that this will bias the
# results, so we do not currently use it. If False, use all languages with
# data in URIEL.
filter_by_agreement = False

# Number of random samples for each classification
n_samples = 401 # Reduce to e.g. 9 for faster evaluation


def read_representations(filename, restrict_isos=None):
    e = mulres.embeddings.Embeddings(filename)
    e_matrix = np.array([e[k] for k in sorted(e.embeddings.keys())])
    #print('Standard deviations:', np.std(e_matrix, axis=0))
    e_matrix = sklearn.preprocessing.scale(e_matrix)
    e = dict(zip(sorted(e.embeddings.keys()), e_matrix))

    return {k: v.astype(np.float64)
            for k, v in e.items()
            if restrict_isos is None or k[:3] in restrict_isos}


URIEL_EXCLUSIVE = [
        feature_set.split('/') for feature_set in '''
S_NEGATIVE_WORD_AFTER_SUBJECT/S_NEGATIVE_WORD_BEFORE_SUBJECT
S_NEGATIVE_WORD_AFTER_OBJECT/S_NEGATIVE_WORD_BEFORE_OBJECT
S_RELATIVE_AFTER_NOUN/S_RELATIVE_BEFORE_NOUN
S_DEMONSTRATIVE_WORD_AFTER_NOUN/S_DEMONSTRATIVE_WORD_BEFORE_NOUN
S_NUMERAL_AFTER_NOUN/S_NUMERAL_BEFORE_NOUN
S_NEGATIVE_WORD_AFTER_VERB/S_NEGATIVE_WORD_BEFORE_VERB
S_POSSESSOR_AFTER_NOUN/S_POSSESSOR_BEFORE_NOUN
S_ADPOSITION_AFTER_NOUN/S_ADPOSITION_BEFORE_NOUN
S_ADJECTIVE_AFTER_NOUN/S_ADJECTIVE_BEFORE_NOUN
S_SUBJECT_AFTER_OBJECT/S_SUBJECT_BEFORE_OBJECT
S_SUBJECT_AFTER_VERB/S_SUBJECT_BEFORE_VERB
S_OBJECT_AFTER_VERB/S_OBJECT_BEFORE_VERB
S_DEFINITE_AFFIX/S_DEFINITE_WORD
S_POSSESSIVE_PREFIX/S_POSSESSIVE_SUFFIX
S_PLURAL_PREFIX/S_PLURAL_SUFFIX
S_TAM_PREFIX/S_TAM_SUFFIX
S_CASE_PREFIX/S_CASE_SUFFIX
S_TEND_PREFIX/S_TEND_SUFFIX
S_POLARQ_WORD/S_POLARQ_AFFIX
S_NEGATIVE_AFFIX/S_NEGATIVE_WORD
S_DEMONSTRATIVE_PREFIX/S_DEMONSTRATIVE_SUFFIX
S_NEGATIVE_PREFIX/S_NEGATIVE_SUFFIX
'''.split()]


def ensure_exclusive(feature1, feature2):
    common_keys = frozenset(feature1.keys()) & frozenset(feature2.keys())
    accept = {k for k in common_keys
            if sorted([int(feature1[k]), int(feature2[k])]) == [0, 1]}
    for k in list(feature1.keys()):
        if k not in accept:
            del feature1[k]
    for k in list(feature2.keys()):
        if k not in accept:
            del feature2[k]



ClassificationResult = namedtuple('ClassificationResult',
    'test_doculect y_pred y_base train_sizes')


def classify(test_doculect, train_folds, x_map, y_map):
    y_base = []
    y_pred = []
    for sample_idx, train_doculects in enumerate(train_folds):
        y = np.array([y_map[doculect] for doculect in train_doculects])
        x = np.array([x_map[doculect] for doculect in train_doculects])
        test_x = x_map[test_doculect]

        # scikit-learn logistic regression only supports binary labels
        # This means we have no choice but to discretize
        y = np.round(y)
        y_counts = Counter(y)
        if balance:
            cond = bool(random.randint(0, 1))
        else:
            cond = y_counts[0] > y_counts[1]
        dummy_predictions = np.array([
            1.0 if cond else 0.0,
            0.0 if cond else 1.0])
        if len(y_counts) < 2 or min(y_counts.values()) < 5:
            # Dummy classifier for the majority class
            predictions = dummy_predictions
            predictions_shuf = dummy_predictions
        else:
            if cross_validate:
                clf = LogisticRegressionCV(
                        Cs=5,
                        penalty='l2',
                        solver='liblinear',
                        class_weight=(
                            'balanced' if balance else None)
                        ).fit(x, y)
            else:
                clf = LogisticRegression(
                        C=fixed_regularization,
                        penalty='l2',
                        solver='liblinear',
                        class_weight=(
                            'balanced' if balance else None)
                        ).fit(x, y)

            predictions = clf.predict_proba([test_x])[0]

            y_shuf = y.copy()
            np.random.shuffle(y_shuf)
            if cross_validate:
                clf_shuf = LogisticRegressionCV(
                        Cs=5,
                        penalty='l2',
                        solver='liblinear',
                        class_weight=(
                            'balanced' if balance else None)
                        ).fit(x, y_shuf)
            else:
                clf_shuf = LogisticRegression(
                        C=fixed_regularization,
                        penalty='l2',
                        solver='liblinear',
                        class_weight=(
                            'balanced' if balance else None)
                        ).fit(x, y_shuf)
            predictions_shuf = clf_shuf.predict_proba([test_x])[0]

        y_base.append(predictions_shuf)
        y_pred.append(predictions)

    return ClassificationResult(
            test_doculect=test_doculect,
            y_pred=np.array(y_pred),
            y_base=np.array(y_base),
            train_sizes=[len(train_doculects)
                         for train_doculects in train_folds])


def main():
    iso_filename = sys.argv[1]
    out_filename = sys.argv[2]
    in_filenames = sys.argv[3:]

    assert not os.path.exists(out_filename)

    with open(mulres.config.typology_db_path, 'rb') as f:
        text_info = pickle.load(f)
        independent = pickle.load(f)
        uriel_syntax = pickle.load(f)
        proj_syntax = pickle.load(f)
        proj_cont_syntax = pickle.load(f)

    with open(iso_filename) as f:
        restrict_isos = set(f.read().split())

    iso_glottocode = {}
    glottocode_iso = {}
    iso_family = {}
    for iso in list(restrict_isos):
        try:
            l = Glottolog[iso]
            if l.id not in independent:
                restrict_isos.remove(iso)
            else:
                iso_glottocode[iso] = l.id
                glottocode_iso[l.id] = iso
                family = l.family_id if l.family_id is not None else l.id
                iso_family[iso] = family
        except KeyError:
            restrict_isos.remove(iso)

    
    def sample_language_per_family(e, restrict_doculects=None):
        family_doculects = defaultdict(lambda: defaultdict(list))
        for doculect in e.keys():
            if restrict_doculects is not None \
                    and doculect not in restrict_doculects:
                continue
            family = iso_family[doculect[:3]]
            family_doculects[family][doculect[:3]].append(doculect)
        return [random.choice(random.choice(list(iso_doculects.values())))
                for iso_doculects in family_doculects.values()]


    def sample_folds(e, test_iso, restrict_doculects=None, n_folds=1,
                     naive=False):
        candidate_glottocodes = frozenset(independent[iso_glottocode[test_iso]])
        candidate_doculects = [
                k for k in e.keys()
                if iso_glottocode[k[:3]] in candidate_glottocodes \
                        and (restrict_doculects is None
                             or k in restrict_doculects)]
        candidate_families = defaultdict(list)
        for doculect in candidate_doculects:
            candidate_families[iso_family[doculect[:3]]].append(doculect)
        if naive:
            all_doculects = [
                    k for k in e.keys()
                    if restrict_doculects is None
                       or k in restrict_doculects]
            return [random.sample(all_doculects, len(candidate_families))
                    for _ in range(n_folds)]
        else:
            return [[random.choice(doculects)
                     for doculects in candidate_families.values()]
                    for _ in range(n_folds)]

    
    print(f'Restricting evaluation to {len(restrict_isos)} languages')

    #features = [
    #        'S_TEND_PREFIX',
    #        'S_OBJECT_AFTER_VERB',
    #        'S_NUMERAL_AFTER_NOUN',
    #        'S_ADJECTIVE_AFTER_NOUN', 
    #        'S_ADPOSITION_AFTER_NOUN',
    #        #'S_SUBJECT_AFTER_VERB',
    #]

    #features = sorted(uriel_syntax.keys())
    features = [feature_set[0] for feature_set in URIEL_EXCLUSIVE]

    def get_families(doculects):
        return {iso_family[doculect[:3]] for doculect in doculects}


    for feature_set in URIEL_EXCLUSIVE:
        if len(feature_set) < 2:
            continue
        uriel_targets = uriel_syntax.get(feature_set[0])
        uriel_other = uriel_syntax.get(feature_set[1])
        if None not in (uriel_targets, uriel_other):
            ensure_exclusive(uriel_targets, uriel_other)

    def extend_to_bare_iso(d):
        # A dict of the form
        # {'eng-x-bible-abc': x,
        #  'fra-x-bible-xyz': y,
        #  ...}
        # will be extended to include:
        # {'eng': x,
        #  'fra': y,
        #  'eng-x-bible-abc': x,
        #  'fra-x-bible-xyz': y,
        #  ...
        #  }
        # This is a workaround to make sure doculect identifiers can be both
        # bare ISO codes and full identifiers with the first three letters
        # being the ISO code.
        #
        # In case multiple items for the same ISO code exists, use the mean
        # of individual doculects.
        #iso_d = {}
        iso_samples = defaultdict(list)
        for k, v in d.items():
            #if k[:3] not in iso_d:
            #    iso_d[k[:3]] = v
            iso_samples[k[:3]].append(v)
        iso_d = {iso: np.mean(vs) for iso, vs in iso_samples.items()}
        iso_d.update(d)
        return iso_d

    with open(out_filename, 'w') as dumpf:
        for filename in in_filenames:
            e = read_representations(filename, restrict_isos=restrict_isos)

            vector_isos = {k[:3] for k in e.keys()}

            for feature in features:
                # Does this feature have projected values?
                has_proj = feature in proj_cont_syntax

                cont_feats = None
                if has_proj:
                    cont_feats = extend_to_bare_iso(proj_cont_syntax[feature])
                #proj_feats = extend_to_bare_iso(proj_syntax[feature])

                # No point working on features we can not evaluate
                if feature not in uriel_syntax:
                    continue
                uriel_feats = extend_to_bare_iso(uriel_syntax[feature])

                # Only include doculects in the test set if:
                #   a) they occur in URIEL
                #   b) the absolute difference between the projected ratio and
                #      URIEL's binary value is at most confidence_threshold
                #
                # Note condition (b) is ignored if no projected values are
                # available for this feature.
                #
                # 1.0 --> no agreement needed
                # 0.5 --> both must agree
                # less than 0.5 --> both must agree and the the dominance must be
                #   sufficiently strong
                if has_proj and filter_by_agreement:
                    confidence_threshold = 1.0/3
                    confident_doculects = {
                        doculect for doculect in e.keys()
                        if doculect in uriel_feats
                        and doculect in cont_feats
                        and abs(uriel_feats[doculect]-cont_feats[doculect]) <
                            confidence_threshold
                        and doculect[:3] in iso_glottocode}
                else:
                    confident_doculects = {
                        doculect for doculect in e.keys()
                        if doculect[:3] in uriel_feats
                        and doculect[:3] in iso_glottocode}

                if len(get_families(confident_doculects)) < 50:
                    print(feature, 'has below 50 families represented')
                    continue

                # If there are projected values, we should classify both with
                # and without them.
                proj_options = [False]
                if has_proj: proj_options.append(True)

                test_doculects = sorted(confident_doculects)

                for use_proj in proj_options:
                    print(f'{os.path.basename(filename)} '
                          f'{feature}/{use_proj} ({len(test_doculects)})',
                          flush=True)

                    # Classification tasks to be performed in parallel
                    tasks = []
                    for test_doculect in test_doculects:
                        train_folds = sample_folds(
                                e, test_doculect[:3],
                                n_folds=n_samples,
                                naive=naive_loocv,
                                restrict_doculects=
                                    (cont_feats if use_proj
                                        else uriel_feats).keys()
                                )
                        tasks.append((
                            test_doculect, train_folds,
                            e, (cont_feats if use_proj else uriel_feats)))

                    with Pool() as p:
                        solved_tasks = p.starmap(classify, tasks)

                    for r in solved_tasks:
                        for sample_idx, (y_pred, y_base, n_train) \
                        in enumerate(zip(
                            r.y_pred, r.y_base, r.train_sizes)):
                            print('\t'.join([
                                os.path.basename(filename),
                                feature,
                                'proj' if use_proj else 'uriel',
                                r.test_doculect,
                                str(iso_family[r.test_doculect[:3]]),
                                str(sample_idx),
                                str(n_train),
                                f'{y_pred[1]:.4f}',
                                f'{y_base[1]:.4f}',
                                f'{int(uriel_feats[r.test_doculect])}']),
                                file=dumpf)

                    dumpf.flush()


if __name__ == '__main__': main()

