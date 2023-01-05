import sys
import gzip
import pickle
from multiprocessing import Pool
from collections import Counter, defaultdict
import os
import statistics
from operator import itemgetter

import Levenshtein
import numpy as np

import mulres.config
import mulres.utils
from mulres.empfile import EncodedMPF

def lemma_alignments(source_empf, target_empf, lemma_candidates):
    assert source_empf.verse_ids == target_empf.verse_ids

    lemma_translations = defaultdict(Counter)

    for verse_i, verse in enumerate(source_empf.verse_ids):
        if source_empf.sentences[verse_i] is None: continue
        if target_empf.sentences[verse_i] is None: continue

        # target n-gram mapped to list of indexes in target verse
        ngram_positions = defaultdict(list)

        target_tokens = [target_empf.sentences_vocab[token_i]
                         for token_i in target_empf.sentences[verse_i]]
        source_tokens = [source_empf.sentences_vocab[token_i]
                         for token_i in source_empf.sentences[verse_i]]
        source_lemmas = [source_empf.lemma_vocab[token_i]
                         for token_i in source_empf.lemma[verse_i]]
        source_pos = [source_empf.pos_vocab[token_i]
                      for token_i in source_empf.pos[verse_i]]

        assert len(source_tokens) == len(source_lemmas)

        target_links = [None]*len(target_tokens)

        for token_idx, token in enumerate(target_tokens):
            # token plus boundary markers
            s = '#' + token + '#'
            for i in range(len(s)-1):
                for j in range(i+2, len(s)+1):
                    ngram = s[i:j]
                    ngram_positions[ngram].append(token_idx)

        for lemma_i, (lemma, pos, plain) in enumerate(
                zip(source_lemmas, source_pos, source_tokens)):
            candidates = lemma_candidates[lemma]
            # candidates should be sorted by decreasing saved_per_token
            for form, bayes_factor, saved_per_token in candidates:
                # TODO: adjust thresholds?
                if bayes_factor < 0.0:
                    continue
                elif saved_per_token < 0.7 and bayes_factor < 100:
                    continue
                elif saved_per_token < 0.2:
                    break
                # the scores of this form seems acceptable, find the first
                # unaligned match for it
                for idx in ngram_positions[form]:
                    link = target_links[idx]
                    if link is None or saved_per_token > link[-1]:
                        target_links[idx] = (lemma, pos, saved_per_token)
                        break

        for target, link in zip(target_tokens, target_links):
            if link is not None:
                lemma, pos, _ = link
                lemma_translations[(pos, lemma)][target] += 1

    return dict(lemma_translations)


def similarity(s1, s2):
    d = Levenshtein.distance(s1, s2)
    n1 = len(s1)
    n2 = len(s2)
    return d / (n1 + n2)


def mean_similarity(forms1, forms2):
    return statistics.mean(
            [similarity(form1, form2)
                    for form1 in forms1
                    for form2 in forms2])

def guess_paradigms(target_text):
    target_empf = EncodedMPF(
            mulres.utils.encoded_filename(target_text['name']))
    aligned_filename = mulres.utils.aligned_filename(target_text['name'])
    with gzip.open(aligned_filename, 'rb') as f:
        source_names = pickle.load(f)
        candidates_list = pickle.load(f)

    pos_paradigms = defaultdict(list)
    form_pos_count = defaultdict(Counter)
    for source_name, lemma_candidates in zip(source_names, candidates_list):
        source_empf = EncodedMPF(mulres.utils.encoded_filename(source_name))
        alignments = lemma_alignments(
                source_empf, target_empf, lemma_candidates)

        for (pos, lemma), form_counts in alignments.items():
            if pos not in ('NOUN', 'VERB'): continue
            if len(form_counts) < 2: continue
            for form in form_counts:
                form_pos_count[form][pos] += 1
            pos_paradigms[pos].append(frozenset(form_counts.keys()))

    form_pos = {form: pos_count.most_common(1)[0][0]
                for form, pos_count in form_pos_count.items()}

    paradigms_filename = mulres.utils.paradigms_filename(target_text['name'])
    with gzip.open(paradigms_filename, 'wt', encoding='utf-8') as f:
        for pos, paradigms in pos_paradigms.items():
            # Assign forms to their majority POS
            paradigms = [{form for form in paradigm if form_pos[form] == pos}
                         for paradigm in paradigms]

            # Exclude large outlier paradigm candidates (probably noise)
            max_len = np.percentile(list(map(len, paradigms)), 90)
            paradigms = [paradigm for paradigm in paradigms
                         if len(paradigm) <= max_len]

            vocab = {form for paradigm in paradigms
                          for form in paradigm}

            cooc_count = Counter()
            for paradigm in paradigms:
                for form1 in paradigm:
                    for form2 in paradigm:
                        if form1 < form2:
                            cooc_count[(form1, form2)] += 1
                        elif form2 > form1:
                            cooc_count[(form2, form1)] += 1

            cluster_of = {form: [form] for form in vocab}
            cooc_count = sorted(
                    cooc_count.items(), key=itemgetter(1), reverse=True)

            for (form1, form2), n in cooc_count:
                if n < 4: break
                forms1 = cluster_of[form1]
                forms2 = cluster_of[form2]
                if forms1 is forms2: continue
                s = mean_similarity(cluster_of[form1], cluster_of[form2])
                if s < 0.3:
                    forms = forms1 + forms2
                    for form in forms:
                        cluster_of[form] = forms

            for cluster in sorted(
                    {tuple(forms) for forms in cluster_of.values()}):
                # Size-1 clusters are what defines isolating languages, so
                # they should be kept
                #if len(cluster) > 1:
                print(pos, ' '.join(sorted(cluster)), file=f)


def main():
    text_table = mulres.utils.load_resources_table()

    os.makedirs(mulres.config.paradigms_path, exist_ok=True)

    target_texts = [
            info for name, info in text_table.items()
            if os.path.exists(mulres.utils.encoded_filename(name)) and
               os.path.exists(mulres.utils.aligned_filename(name))]

    print(len(target_texts), 'texts to guess paradigms for')

    with Pool() as p:
        p.map(guess_paradigms, target_texts)


if __name__ == '__main__': main()

