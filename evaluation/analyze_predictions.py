# Script to analyze the .predictions output of analyze_simple.py
# It also requires the .summary file from that script.
#
#   python3 evaluation/analyze_predictions.py log-balanced-l21e-3.tsv
#
# Will also write figures to data/figures/barplots (hardcoded)
# Remember to adjust f1_threshold! Should be set to 0 when generating figures,
# but increase the value to 0.7 or so to remove clutter from the text output.

from collections import namedtuple, defaultdict, Counter, OrderedDict
import sys
import itertools
import math
import pickle
from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plt

import mulres.utils
import mulres.config

# increase this to get a bit cleaner text output
# for generating figures it should be zero
f1_threshold = 0.70
# For less clutter, set this to 0 to only print those alternative explanations
# that produce a *higher* F1 score. Otherwise, leave it at a relatively small
# value to see which alternative explanations are at least close.
comparison_threshold = 0.10

with open(mulres.config.typology_db_path, 'rb') as f:
    text_info = pickle.load(f)
    pickle.load(f) # independent -- not needed here
    pickle.load(f) # uriel_syntax -- not mutually exclusive, so not compatible
                   # with the output of simple_language_vectors.py
    proj_syntax = pickle.load(f)
    cont_syntax = pickle.load(f)

Result = namedtuple('Result',
        'name feature db doculect iso family n_0 n_1 prediction target')


def read_data(filename):
    with open(filename) as f:
        for line in f:
            name, feature, db, doculect, iso, family, n_0, n_1, target = \
                    line.split()
            yield Result(
                    name=name,
                    feature=feature,
                    db=db,
                    doculect=doculect,
                    iso=iso,
                    family=family,
                    n_0=int(n_0),
                    n_1=int(n_1),
                    prediction=int(int(n_1)>=int(n_0)),
                    target=int(target))


def get_accuracy(c):
    assert c.shape == (2, 2)
    if c.sum() == 0: return 0.0
    return (c[0,0] + c[1,1]) / c.sum()


def get_f1(c):
    if c[1,1] + c[0,1] == 0:
        return 0.0
    if c[1,1] + c[1,0] == 0:
        return 0.0
    p = c[1,1] / (c[1,1] + c[0,1])
    r = c[1,1] / (c[1,1] + c[1,0])
    if p == 0 or r == 0:
        return 0.0
    return 2*p*r / (p+r)


def combined_f1(c):
    assert c.shape == (2, 2)
    c2 = np.array([[c[1-i, 1-j] for j in (0, 1)] for i in (0, 1)])
    # Possible alternative:
    #return min([get_f1(c), get_f1(c2)])
    return np.mean([get_f1(c), get_f1(c2)])


def get_weights(difs, balance='family'):
    if balance in ('family', 'iso'):
        key = itemgetter(1 if balance == 'iso' else 2)
        family_count = Counter()
        for dif in difs:
            family_count[key(dif)] += 1
        return {dif: 1.0/(len(family_count)*family_count[key(dif)])
                for dif in difs}
    elif balance == 'doculect':
        return {dif: 1.0/len(difs) for dif in difs}
    else:
        raise NotImplementedError(f'Invalid balance parameter: {balance}')


def evaluate_projection(difs, feature, targets):
    difs = [(doculect, iso, family) for doculect, iso, family in difs
            if doculect in proj_syntax[feature]]
    weights = get_weights(difs)
    c = np.zeros((2, 2), dtype=float)
    for dif in difs:
        doculect, iso, family = dif
        w = weights[dif]
        # print(targets[dif], proj_syntax[feature][doculect])
        c[targets[dif], proj_syntax[feature][doculect]] += w
    return c

def analyze(vectors_name, table):
    # Analyze the results for all languages in a single experiment.
    # result.name should be constant for all items in table.
    feature_db_fam_f1 = dict()
    feature_db_proj_f1 = dict()

    feature_doculect_predictions = defaultdict(dict)
    feature_doculect_target = defaultdict(dict)
    for r in table:
        dif = (r.doculect, r.iso, r.family)
        feature_doculect_predictions[(r.feature, r.db)][dif] = r.prediction
        feature_doculect_target[r.feature][dif] = r.target


    feature_short = OrderedDict(
            S_NUMERAL_AFTER_NOUN=('NumN', 'NNum'),
            S_ADJECTIVE_AFTER_NOUN=('AdjN', 'NAdj'),
            S_RELATIVE_AFTER_NOUN=('RelN', 'NRel'),
            S_POSSESSOR_AFTER_NOUN=('PossN', 'NPoss'),
            S_OBJECT_AFTER_VERB=('OV', 'VO'),
            S_ADPOSITION_AFTER_NOUN=('Prep', 'Post'))


    def get_profile(dif):
        row = []
        for full, names in feature_short.items():
            v = feature_doculect_target[full].get(dif)
            row.append('?'*len(names[0]) if v is None else names[v])
        return ' '.join(row)

    def get_profile_dict(dif):
        d = {}
        for full, names in feature_short.items():
            k = '/'.join(names)
            v = feature_doculect_target[full].get(dif)
            if v is None:
                d[k] = None
            else:
                d[k] = names[v]
        return d
             

    for (feature, db), doculect_predictions in \
            feature_doculect_predictions.items():
        difs = sorted(doculect_predictions.keys())
        weights = get_weights(difs)
        targets = feature_doculect_target[feature]
        accuracy = sum(weights[dif]*int(prediction == targets[dif])
                        for dif, prediction in doculect_predictions.items())
        confusion = np.zeros((2, 2), dtype=float)
        for dif, prediction in doculect_predictions.items():
            confusion[targets[dif], prediction] += weights[dif]
        accuracy = (confusion[0,0] + confusion[1,1]) / confusion.sum()
        f1 = combined_f1(confusion)
        feature_db_fam_f1[(feature, db)] = f1
        # Started this, but it's probably better to use the experiments
        # directly so we can put the bar at the 99th percentile or whatever
        # dummy_c = np.zeros((2, 2), dtype=float)
        # for dif, target in targets.items():
        #     w = weights[dif]
        #     dummy_c[target, 0] += 0.5*w
        #     dummy_c[target, 1] += 0.5*w
        # feature_db_dummy_f1[(feature, db)] = combined_f1(dummy_c)
        if (f1 < f1_threshold) and (feature not in proj_syntax):
            continue
        print(f'--{vectors_name} {feature} {db} '
              f'{100*accuracy:.1f}% (n = {len(targets)}) '
              f'F1 = {f1:.3f}')
        #print(np.round(100*confusion, 1))
        print(f'  {confusion[0,0]:.3f} {confusion[0,1]:.3f}')
        print(f'  {confusion[1,0]:.3f} {confusion[1,1]:.3f}')

        # Does this set of embeddings only include ISO codes? If False,
        # doculect is a Bible name
        only_iso = len(difs[0][0]) == 3

        if feature in proj_syntax and not only_iso:
            proj_c = evaluate_projection(difs, feature, targets)
            print(f'  PROJECTION BASELINE (n = {len(difs)}): '
                  f'F1 = {combined_f1(proj_c):.3f}  '
                  f'Acc = {100*get_accuracy(proj_c):.1f}%')
            #print(difs[:4], feature, list(targets.items())[:4])
            print(f'    {proj_c[0,0]:.3f} {proj_c[0,1]:.3f}')
            print(f'    {proj_c[1,0]:.3f} {proj_c[1,1]:.3f}')
            feature_db_proj_f1[(feature, db)] = combined_f1(proj_c)
            # Dirty hack to get comparable numbers, copy-pasted from
            # evaluate_projection
            difs = [(doculect, iso, family) for doculect, iso, family in difs
                    if doculect in proj_syntax[feature]]
            weights = get_weights(difs)
            confusion = np.zeros((2, 2), dtype=float)
            for dif in difs:
                prediction = doculect_predictions[dif]
                confusion[targets[dif], prediction] += weights[dif]
            print(f'  COMPARABLE F1 = {combined_f1(confusion):.3f}  '
                  f'Acc = {100*get_accuracy(confusion):.1f}')
            # Compute 3-way confusion tensor: URIEL x projection x prediction
            c3w_difs = defaultdict(list)
            c3w = np.zeros((2, 2, 2), dtype=float)
            for dif in difs:
                target = targets[dif]
                projected = proj_syntax[feature][dif[0]]
                predicted = doculect_predictions[dif]
                c3w[target, projected, predicted] += weights[dif]
                c3w_difs[(target, projected, predicted)].append(dif)
            print(np.round(100*c3w, 1))
            print('  F1 relative to projected labels: '
                  f'{combined_f1(c3w.sum(axis=0)):.3f}')
            for b1 in (0, 1):
                for b2 in (0, 1):
                    for b3 in (0, 1):
                        print(f'  URIEL: {b1}, proj: {b2}, classifier: {b3}:')
                        feature_family_counts = defaultdict(
                                lambda: defaultdict(Counter))
                        for dif in sorted(c3w_difs[(b1,b2,b3)],
                                          key=itemgetter(2)):
                            doculect, _, family = dif
                            for k, v in get_profile_dict(dif).items():
                                if v is not None:
                                    feature_family_counts[k][family][v] += 1
                            print(f'    <{get_profile(dif)}> {family} {doculect}')
                        for k, family_counts in feature_family_counts.items():
                            total_counts = Counter()
                            for family, counts in family_counts.items():
                                for v, n in counts.items():
                                    total_counts[v] += n/sum(counts.values())
                            for v, c in total_counts.items():
                                print(f'    {v} {c:.2f} families')

            c3w_collapsed = np.array(
                    [[c3w[0, 0, 0], c3w[1, 0, 0]],
                     [c3w[0, 1, 1], c3w[1, 1, 1]]])
            disagreeing = c3w_difs[(0, 1, 1)] + c3w_difs[(1, 0, 0)]
            agreeing = c3w_difs[(0, 0, 0)] + c3w_difs[(1, 1, 1)]
            print('  F1 in proj=classifier subset: '
                  f'{combined_f1(c3w_collapsed):.3f}')
            print(f'  {100*len(agreeing)/len(difs):.1f}% agree (doculects)')
            print(f'  {100*c3w_collapsed.sum()/c3w.sum():.1f}% agree (families)')
            print(f'  {len(difs)} doculects and {len(set(dif[2] for dif in difs))} families total')


        for other_feature, other_targets in feature_doculect_target.items():
            # Let's see if other_feature explains the predictions even better
            # than feature
            if feature == other_feature:
                continue
            joint_dif = sorted(
                    set(targets.keys()) &
                    set(other_targets.keys()))
            joint_weights = get_weights(joint_dif)
            this_confusion = np.zeros((2, 2), dtype=float)
            other_confusion = np.zeros((2, 2), dtype=float)
            confusion = np.zeros((2, 2), dtype=float)
            confusion_difs = [[[], []], [[], []]]
            for dif in joint_dif:
                w = joint_weights[dif]
                this_confusion[
                        targets[dif], doculect_predictions[dif]] += w
                other_confusion[
                        other_targets[dif], doculect_predictions[dif]] += w
                this_match = int(
                        doculect_predictions[dif] == targets[dif])
                other_match = int(
                        doculect_predictions[dif] == other_targets[dif])
                confusion[this_match, other_match] += w
                confusion_difs[this_match][other_match].append(dif)
            other_accuracy = confusion[:,1].sum()
            this_accuracy = confusion[1,:].sum()
            # Note that the predictions could be accurate but the labels
            # inverted!
            other_f1 = max(combined_f1(other_confusion),
                           combined_f1(other_confusion[:,::-1]))
            this_f1 = combined_f1(this_confusion)
            fix_rate = confusion[0,1] / confusion[0,:].sum()
            #ruin_rate = confusion[1,0] / confusion[1,:].sum()
            improve_rate = confusion[0,1] / (confusion[0,1]+confusion[1,0])
            if other_f1 >= this_f1 - comparison_threshold:
                #assert other_accuracy >= this_accuracy
                print(f'    {other_feature} '
                      f'({other_f1:.3f} vs {this_f1:.3f} ({feature})'
                      #f'{100*other_accuracy:.1f}% > '
                      #f'{100*this_accuracy:.1f}% (n = {len(joint_dif)}, '
                      #f'F1: {combined_f1(this_confusion):.3f} --> '
                      #f'{combined_f1(other_confusion):.3f} '
                      #f'ruined {100*ruin_rate:.1f}% '
                      #f'improved {100*improve_rate:.1f}% '
                      #f'fixed {100*fix_rate:.0f}%'
                      ')')
                #print(np.round(100*this_confusion, 1))
                #print(np.round(100*other_confusion, 1))

    return dict(feature_db_fam_f1=feature_db_fam_f1,
                feature_db_proj_f1=feature_db_proj_f1)


def plot_figures(name_results, vectors_baseline_f1):
    all_features = [
            (('S_ADPOSITION_AFTER_NOUN', 'uriel'), 'Pre/postpositions'),
            (('S_ADPOSITION_AFTER_NOUN', 'proj'), 'Pre/postpositions (projected)'),
            (('S_OBJECT_AFTER_VERB', 'uriel'), 'OV/VO'),
            (('S_OBJECT_AFTER_VERB', 'proj'), 'OV/VO (projected)'),
            (('S_TEND_PREFIX', 'uriel'), 'AffixPos'),
            (('S_TEND_PREFIX', 'proj'), 'AffixPos (projected)'),
            (('S_NUMERAL_AFTER_NOUN', 'uriel'), 'NumN/NNum'),
            (('S_NUMERAL_AFTER_NOUN', 'proj'), 'NumN/NNum (projected)'),
            (('S_POSSESSIVE_PREFIX', 'uriel'), 'PossAffixPos'),
            (('S_PLURAL_PREFIX', 'uriel'), 'PluralAffixPos'),
            (('S_NEGATIVE_PREFIX', 'uriel'), 'NegAffixPos'),
            (('S_TAM_PREFIX', 'uriel'), 'TAMAffixPos'),
            (('S_CASE_PREFIX', 'uriel'), 'CaseAffixPos'),
            (('S_POSSESSOR_AFTER_NOUN', 'uriel'), 'PossN/NPoss'),
            (('S_RELATIVE_AFTER_NOUN', 'uriel'), 'RelN/NRel'),
            (('S_NEGATIVE_WORD_AFTER_VERB', 'uriel'), 'NegV/VNeg'),
            (('S_SUBJECT_AFTER_VERB', 'uriel'), 'SV/VS'),
            (('S_SUBJECT_AFTER_VERB', 'proj'), 'SV/VS (projected)'),
            (('S_ADJECTIVE_AFTER_NOUN', 'uriel'), 'AdjN/NAdj'),
            (('S_ADJECTIVE_AFTER_NOUN', 'proj'), 'AdjN/NAdj (projected)'),
            (('S_DEMONSTRATIVE_WORD_AFTER_NOUN', 'uriel'), 'DemN/NDem'),
            (('S_NEGATIVE_AFFIX', 'uriel'), 'NegAffix?'),
            #(('S_', 'uriel'), ''),
            ]
    vector_groups = [
            [
                ('asjp_svd', 'ASJP'),
                ('lexical_svd', 'Lexical'),
            ],
            [
                ('nmt_to_eng', 'NMTx2eng'),
                ('nmt_from_eng', 'NMTeng2x'),
                ('malaviya_mtcell', 'MTCell'),
                ('malaviya_mtvec', 'MTVec'),
            ],
            [
                ('ostling2017_all', 'Ö&T'),
                ('character_level', 'CharLM'),
            ],
            [
                ('word_level', 'WordLM'),
            ],
            [
                ('reinflect_noun', 'Reinflect-Noun'),
                ('reinflect_verb', 'Reinflect-Verb'),
                ('word_encoding', 'Encoder'),
            ]
        ]
    group_attribs = [
            dict(hatch='///'),
            dict(hatch='\\\\\\'),
            dict(hatch='xxx'),
            dict(hatch='+++'),
            dict(hatch='|||'),
            dict(hatch='...'),
            ]
    color_sequence = [f'C{i}' for i in range(20)]
    # ['green', 'blue', 'magenta', 'black', 'yellow']
    n_vectors = sum(map(len, vector_groups))
    width = 0.7/n_vectors
    xgroupspace = width*0.5
    for features in all_features:
        features = [features]
        plt.clf()
        fig, ax = plt.subplots()
        ax.set_ylim([0.3, 1.0])
        xoff = -0.5*(width*n_vectors)
        for vector_group, bar_attribs in zip(vector_groups, group_attribs):
            for (vector_name, vector_label), bar_color in zip(
                    vector_group, color_sequence):
                feature_db_fam_f1 = name_results[vector_name]['feature_db_fam_f1']
                feature_db_proj_f1 = name_results[vector_name]['feature_db_proj_f1']
                xs = np.arange(len(features)).astype(float)
                ys = [feature_db_fam_f1.get(feature_db, 0.0)
                      for feature_db, _ in features]
                rects = ax.bar(xs+xoff, ys, width, label=vector_label,
                        color=bar_color,
                        **bar_attribs, edgecolor='white')
                # From Matplotlib 3.4.0 this should be possible:
                #ax.bar_label(
                #        rects, labels=[vector_label], rotation=90,
                #        label_type='center')
                # Something like this to show the baseline / projection level
                for x_mid, (feature_db, _) in zip(xs, features):
                    proj_f1 = feature_db_proj_f1.get(feature_db)
                    if proj_f1 is not None:
                        ax.plot([x_mid-0.45, x_mid+0.45], [proj_f1, proj_f1], 'k--')
                    # The choice of character_level here is arbitrary, but we
                    # should pick one that uses the Bible corpus
                    # The actual baseline will be slightly different for other
                    # language vectors on other subsets of the data
                    base_f1 = vectors_baseline_f1['character_level'].get((feature_db))
                    if base_f1 is not None:
                        # Use 99th percentile (see analyze_simple.py) from
                        # dummy classifier monte carlo samples as baseline
                        # level
                        ax.plot([x_mid-0.45, x_mid+0.45],
                                [base_f1[2], base_f1[2]],
                                'k:')

                xoff += width
            xoff += xgroupspace
        if len(features) >= 2:
            plt.xticks(ticks=xs, labels=[label for _, label in features])
        else:
            plt.xticks(ticks=[])
            # This is taken care of by subfigure captions in the paper:
            #ax.set_xlabel(features[0][1])
        ax.set_ylabel('F-score (mean of both classes)')
        ax.legend(loc='upper left')
        fig.tight_layout()
        plt.savefig(f'data/figures/barplots/{"-".join(features[0][0])}.pdf')
        plt.close()


def main():
    filename = sys.argv[1] + '.predictions'
    summary_filename = sys.argv[1] + '.summary'
    name_results = {}
    for name, sub_table in itertools.groupby(read_data(filename),
            key=lambda r: r.name):
        sub_table = list(sub_table)
        name_results[name] = analyze(name, sub_table)

    with open(summary_filename) as f:
        classifier_f1 = defaultdict(dict)
        baseline_f1 = defaultdict(dict)
        for line in f:
            name, feature, db, *rest = line.strip().split('\t')
            rest = list(map(float, rest))
            assert len(rest) == 6
            classifier_f1[name][(feature, db)] = 0.01*np.array(rest[:3])
            baseline_f1[name][(feature, db)] = 0.01*np.array(rest[3:])

    plot_figures(name_results, baseline_f1)


if __name__ == '__main__':
    main()

