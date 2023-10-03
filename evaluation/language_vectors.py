import os
import json
from collections import defaultdict
import pickle

import lang2vec.lang2vec as l2v
from langinfo.glottolog import Glottolog

import mulres.utils

NIKOLAEV_NEIGHBORS = mulres.utils.load_contact_graph()

def glottolog_ancestry(language):
    ancestry = tuple()
    try:
        l = Glottolog[language]
        while True:
            ancestry = ancestry + (l,)
            l = Glottolog[l.ancestor(1).id]
            if l.family_id is None:
                return ancestry + (l,)
    except KeyError:
        return tuple()



def read_wordorder_table(name):
    filename = mulres.utils.wordorder_filename(name)
    if not os.path.exists(filename): return None
    table = {}
    with open(filename, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            assert len(fields) == 5
            # (dep_pos, head_pos, deprel)
            pos_rel = tuple(fields[:3])
            # head-initial and head-final counts
            n_hi, n_hf = fields[3:5]
            table[pos_rel] = (int(n_hi), int(n_hf))
    return table


def get_wordorder_features(table, discretize):
    def get_relation(dep_pos, head_pos, deprel):
        n_hi, n_hf = table.get((dep_pos, head_pos, deprel), (0, 0))
        # If counts are too low, back off to non-core projections
        # TODO: consider the bias introduced by this fallback
        if n_hi + n_hf < 16:
            n_hi, n_hf = table.get(
                    (dep_pos.rstrip(':CORE'),
                        head_pos.rstrip(':CORE'),
                        deprel),
                    (0, 0))
        return n_hi, n_hf

    features = {}
    feature_specs = [
            'NOUN VERB obj S_OBJECT_AFTER_VERB head-initial',
            'ADJ:CORE NOUN amod S_ADJECTIVE_AFTER_NOUN head-initial',
            'ADP NOUN case S_ADPOSITION_AFTER_NOUN head-initial',
            'NUM:CORE NOUN nummod S_NUMERAL_AFTER_NOUN head-initial',
            'NOUN VERB obl S_OBLIQUE_AFTER_VERB head-initial',
            'NOUN VERB nsubj S_SUBJECT_AFTER_VERB head-initial',
            # Note to note: sync with evaluation/language_vectors.py
            # Note: added this one recently (should actually use acl:relcl but
            # that is not projected), if there are problems consider removing:
            'VERB NOUN acl S_RELATIVE_AFTER_NOUN head-initial',
            ]

    for spec in feature_specs:
        dep_pos, head_pos, deprel, feature, order = spec.split()
        n_hi, n_hf = get_relation(dep_pos, head_pos, deprel)
        n = n_hi + n_hf
        if n >= 16:
            ratio = (n_hi/n if order == 'head-initial' else n_hf/n)
            value = int(ratio >= 0.5) if discretize else ratio
            features[feature] = value

    return features


def read_wordorder_features(text_info, discretize=True):
    feature_doculect_values = defaultdict(dict)
    for name, info in text_info.items():
        table = read_wordorder_table(name)
        if table is None: continue
        doculect_features = get_wordorder_features(table, discretize)
        for feature, ratio in doculect_features.items():
            feature_doculect_values[feature][name] = ratio
    return dict(feature_doculect_values)


def read_affix_features(discretize=True):
    with open(mulres.config.affixation_path, 'r') as f:
        data = json.load(f)

    # WALS chapter 26 has ten features which are weighed together for the
    # classification.
    # Nominal: 2+1+1+1 = 5 points
    # Verbal: 2+2+1+1+1+1 = 8 points
    # These weights are used here to very roughly approximate the WALS
    # classification.

    pos_weight = { 'NOUN': 5/13,
                   'VERB': 8/13 }
    tend_suffix = {}

    for doculect, pos_affixes in data.items():
        suffix_score = 0.0
        prefix_score = 0.0
        for pos, affixes in pos_affixes.items():
            n_prefixes = len(affixes['prefixes'])
            n_suffixes = len(affixes['suffixes'])
            suffixing = n_suffixes >= n_prefixes*2
            prefixing = n_prefixes >= n_suffixes*2
            if suffixing:
                suffix_score += pos_weight[pos]
            elif prefixing:
                prefix_score += pos_weight[pos]
            else:
                suffix_score += 0.5 * pos_weight[pos]
                prefix_score += 0.5 * pos_weight[pos]

        score = suffix_score / (suffix_score+prefix_score)
        tend_suffix[doculect] = int(score > 0.5) if discretize else score

    return {'S_TEND_SUFFIX': tend_suffix,
            'S_TEND_PREFIX': {k:1-x for k,x in tend_suffix.items()}}


def get_lang2vec_targets(text_info, categories, exclusive=None):
    """Import data from URIEL

    Args:
        text_info: from mulres.utils.load_resources_table()
        categories: URIEL categories to load features from
        exclusive: None or a dict object, which is used for each
                   feature to look up a tuple of features which are
                   supposed to be mutually exclusive. If not, the
                   feature is dropped for that particular language.

    Returns:
        dict { feature_name: { doculect: value } }
    """
    uriel_isos = set(l2v.LANGUAGES)
    texts = sorted(text_info.keys())
    l2v_dict = l2v.get_features(
            [text_info[k]['iso'] for k in texts
             if text_info[k]['iso'] in uriel_isos],
            categories,
            header=True)
    all_features = l2v_dict['CODE']
    del l2v_dict['CODE']

    r = { feature: {
            k: l2v_dict[text_info[k]['iso']][feature_i]
            for k in texts
            if text_info[k]['iso'] in l2v_dict
                and l2v_dict[text_info[k]['iso']][feature_i] != '--'}
        for feature_i, feature in enumerate(all_features)}

    if exclusive is not None:
        for feature, values in r.items():
            if feature not in exclusive: continue
            other_features = exclusive[feature]
            for name, value in list(values.items()):
                all_values = [r[f].get(name) for f in other_features] + [value]
                if None in all_values:
                    # there are some cases where this happens, for
                    # subject/verb and object/verb order (Ethnologue giving
                    # multiple basic word orders?)
                    # e.g. ign, kki, mcb, ssw
                    consistent = False
                else:
                    all_values.sort()
                    # all values should be 0 except one
                    consistent = all_values[-2] == 0 and all_values[-1] == 1
                if not consistent:
                    for f in (feature,) + other_features:
                        if name in r[f]:
                            del r[f][name]

    return r


def is_independent(l1, l2):
    if l1.family_id == l2.family_id: return False
    if l1.macroarea == l2.macroarea: return False
    neighbors = NIKOLAEV_NEIGHBORS.get(l1.id, set())
    if l2.id in neighbors: return False
    return True


def independent_languages(all_languages):
    all_languages_glottolog = {}
    for language in all_languages:
        try:
            l = Glottolog[language]
            l.family_id
            l.macroarea
            all_languages_glottolog[language] = l
        except KeyError:
            # Skip languages not in Glottolog
            pass
        except AttributeError:
            # Skip languages without family_id and macroarea defined
            pass
    independent = {
            language1: [
                language2
                for language2, l2 in all_languages_glottolog.items()
                if is_independent(l1, l2)]
            for language1, l1 in all_languages_glottolog.items()}
    return independent

