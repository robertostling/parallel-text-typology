import numpy as np
import lang2vec.lang2vec as l2v

import mulres.utils

import language_vectors 
import evaluate_language_vectors 
import analyze_predictions
from family import family_from_iso

def to_bernoulli(d):
    return {k:np.array([1.0 if v < 0.5 else 0.0, 1.0 if v >= 0.5 else 0.0])
            for k,v in d.items()}


feature_table_code=dict(
        S_OBJECT_AFTER_VERB=r'\featlabel{OV} & \textsc{noun}/\textsc{propn} $\xleftarrow{\text{obj}}$ \textsc{verb}',
        S_ADJECTIVE_AFTER_NOUN=r'\featlabel{AdjN} & \textsc{adj*} $\xleftarrow{\text{amod}}$ \textsc{noun}',
        S_RELATIVE_AFTER_NOUN=r'\featlabel{RelN} & \textsc{verb} $\xleftarrow{\text{acl}}$ \textsc{noun}',
        S_NUMERAL_AFTER_NOUN=r'\featlabel{NumN} & \textsc{num*} $\xleftarrow{\text{nummod}}$ \textsc{noun}',
        S_OBLIQUE_AFTER_VERB=r'\featlabel{XV} & \textsc{noun}/\textsc{propn} $\xleftarrow{\text{obl}}$ \textsc{verb}',
        S_SUBJECT_AFTER_VERB=r'\featlabel{SV} & \textsc{noun}/\textsc{propn} $\xleftarrow{\text{nsubj}}$ \textsc{verb}',
        S_ADPOSITION_AFTER_NOUN=r'\featlabel{AdpN} & \textsc{adp} $\xleftarrow{\text{case}}$ \textsc{noun}',
        S_TEND_PREFIX=r'\featlabel{Prefix} & Prefixing',
        S_TEND_SUFFIX=r'\featlabel{Suffix} & Suffixing',
)

URIEL_EXCLUSIVE_FEATURES = [category.split('+') for category in '''
    S_TEND_SUFFIX+S_BEFORE_PREFIX
    S_RELATIVE_AFTER_NOUN+S_RELATIVE_BEFORE_NOUN
    S_ADJECTIVE_AFTER_NOUN+S_ADJECTIVE_BEFORE_NOUN
    S_NUMERAL_AFTER_NOUN+S_NUMERAL_BEFORE_NOUN
    S_ADPOSITION_AFTER_NOUN+S_ADPOSITION_BEFORE_NOUN
    S_OBJECT_AFTER_VERB+S_OBJECT_BEFORE_VERB
    S_SUBJECT_AFTER_VERB+S_SUBJECT_BEFORE_VERB
    S_OBLIQUE_AFTER_VERB+S_OBLIQUE_BEFORE_VERB

'''.split()]

def main():
    for use_exclusive in (False, True):
        if not use_exclusive:
            print(r'\multicolumn{6}{c}{\textbf{Non-exclusive}} \\')
        else:
            print(r'\multicolumn{6}{c}{\textbf{Exclusive}} \\')

        text_info = mulres.utils.load_resources_table()
        uriel_syntax = language_vectors.get_lang2vec_targets(
            text_info,
            l2v.fs_union(['syntax_wals', 'syntax_ethnologue']))
        proj_syntax = language_vectors.read_wordorder_features(text_info)
        proj_syntax.update(language_vectors.read_affix_features())

        if use_exclusive:
            #for category, features in language_vectors.URIEL_CATEGORIES.items():
            #    for feature_set in features:
            for feature_set in URIEL_EXCLUSIVE_FEATURES:
                feature = feature_set[0]
                uriel_targets = uriel_syntax.get(feature)
                proj_targets = proj_syntax.get(feature)
                if len(feature_set) > 1:
                    uriel_other = uriel_syntax.get(feature_set[1])
                    if None not in (uriel_targets, uriel_other):
                        evaluate_language_vectors.ensure_exclusive(
                                uriel_targets, uriel_other)
                    proj_other = proj_syntax.get(feature_set[1])
                    if None not in (proj_targets, proj_other):
                        before = len(proj_targets), len(proj_other)
                        evaluate_language_vectors.ensure_exclusive(
                                proj_targets, proj_other)
                        after = len(proj_targets), len(proj_other)
                        assert before == after

        all_family_agreements = []
        all_lang_agreements = []
        all_doculect_agreements = []

        def doculect_to_dif(doculect):
            return (doculect, doculect[:3], family_from_iso(doculect[:3]))

        for feature, proj_x in proj_syntax.items():
            print(feature)
            uriel_x = uriel_syntax[feature]
            doculects = sorted(set(uriel_x.keys()) & set(proj_x.keys()))
            c = np.zeros((2, 2), dtype=float)
            iso_c = np.zeros((2, 2), dtype=float)
            family_c = np.zeros((2, 2), dtype=float)
            difs = list(map(doculect_to_dif, doculects))
            iso_weights = analyze_predictions.get_weights(difs, balance='iso')
            family_weights = analyze_predictions.get_weights(difs, balance='family')
            for dif in difs:
                doculect, iso, family = dif
                proj = proj_x[doculect]
                target = int(uriel_x[doculect])
                c[target, proj] += 1.0
                iso_c[target, proj] += iso_weights[dif]
                family_c[target, proj] += family_weights[dif]
            doc_a = analyze_predictions.get_accuracy(c)
            iso_a = analyze_predictions.get_accuracy(iso_c)
            fam_a = analyze_predictions.get_accuracy(family_c)
            doc_f1 = analyze_predictions.combined_f1(c)
            iso_f1 = analyze_predictions.combined_f1(iso_c)
            fam_f1 = analyze_predictions.combined_f1(family_c)
            all_family_agreements.append(fam_a)
            all_lang_agreements.append(iso_a)
            all_doculect_agreements.append(doc_a)
            print(f'Doculect: {100*doc_a:.1f}%  {doc_f1:.3f}')
            print(f'Language: {100*iso_a:.1f}%  {iso_f1:.3f}')
            print(f'Family:   {100*fam_a:.1f}%  {fam_f1:.3f}')
            data = f'{100*iso_a:.1f}\\% & {iso_f1:.3f} & ' \
                   f'{100*fam_a:.1f}\\% & {fam_f1:.3f} \\\\'
            if feature in feature_table_code:
                print(feature_table_code[feature] + ' & ' + data)
            print()

        print(f'Mean per-family acc:   {100*np.mean(all_family_agreements):.1f}')
        print(f'Mean per-language acc: {100*np.mean(all_lang_agreements):.1f}')
        print(f'Mean per-doculect acc: {100*np.mean(all_doculect_agreements):.1f}')

if __name__ == '__main__': main()

