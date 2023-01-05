import sys
from multiprocessing import Pool
from operator import itemgetter
from collections import defaultdict
import time
import os.path

import numpy as np

from mulres.empfile import EncodedMPF, betabinomial_similarity
from mulres.utils import (load_verse_table, load_resources_table,
                          encoded_filename, aligned_filename)
import mulres.config


def load_empf(filename):
    empf = EncodedMPF(filename)
    if hasattr(empf, 'lemma'):
        # If this file is lemmatized, create dummy n-grams from lemmas
        empf.make_lemma_ngrams()
        empf.count_ngrams(annotation='lemma')
        empf.make_ngram_positions(
                np.arange(len(empf.ngram_list)), annotation='lemma')
    else:
        empf.make_ngrams()
        empf.count_ngrams()
    return empf


def align_raw(target_text, source_texts):
    if os.path.exists(aligned_filename(target_text['name'])):
        return target_text, None

    target_empf = load_empf(encoded_filename(target_text['name']))

    def align_ngram(source_empf, query_ngram_i):
        query_verses = {
            verse_i
            for verse_i, _, _ in source_empf.ngram_positions[query_ngram_i]}

        real_query_verses, (result, count) = \
                target_empf.find_ngrams_from_verses(
                        query_verses,
                        low_limit=2, high_limit=2)

        l = len(real_query_verses)
        n_items = len(target_empf.ngram_list)
        total = source_empf.n_verses

        ngram_verse_count = target_empf.ngram_verse_count
        ks = target_empf.ngram_verse_count[result]

        table = []
        for (ngram_i, both, k) in zip(result, count, ks):
            # In rare cases k needs to be adjusted down because we only use an
            # approximation, this should only happen for n-grams that occur in
            # nearly every verse.
            k = min(k, total)
            k = min(k, total-l+both)
            bayes_factor = betabinomial_similarity(total, both, k, l, n_items)
            saved_per_token = \
                    betabinomial_similarity(total, both, k, l, 1) / both
            table.append((
                    target_empf.ngram_list[ngram_i],
                    bayes_factor,
                    saved_per_token))

        table.sort(key=itemgetter(2), reverse=True)
        return table

    source_candidates = []

    load_t = 0.0
    align_t = 0.0
    for source_text in source_texts:
        t0 = time.time()
        source_empf = load_empf(encoded_filename(source_text['name']))
        load_t += (time.time() - t0)
        assert 'lemma' in source_empf.available_annotations, source_text
        
        t0 = time.time()
        source_candidates.append({
                source_empf.ngram_list[ngram_i]:
                    align_ngram(source_empf, ngram_i)
                for ngram_i in range(len(source_empf.ngram_list))})
        align_t += (time.time() - t0)

    # 91 seconds to load
    # 117 seconds to align
    #print(target_text['name'], round(load_t, 2), round(align_t, 2))

    return target_text, source_candidates


def star_align_raw(args):
    return align_raw(*args)


def main():
    import gzip
    import glob
    import pickle

    os.makedirs(mulres.config.aligned_path, exist_ok=True)

    text_table = load_resources_table()

    source_texts = [
            info for name, info in text_table.items()
            if info['preferred_source'] == 'yes' and
                'ud' in info and
                os.path.exists(encoded_filename(name))]

    target_texts = [
            info for name, info in text_table.items()
            if info['preferred_source'] == 'no' and
                os.path.exists(encoded_filename(name))]

    print('Aligning', len(target_texts), 'targets with', len(source_texts),
            'sources', file=sys.stderr)

    t0 = time.time()
    with Pool() as p:
        tasks = [(target_text, source_texts) for target_text in target_texts]
        for target_text, source_candidates in p.imap(star_align_raw, tasks, 1):
        #for target_text, source_candidates in map(star_align_raw, tasks[:1]):
            if source_candidates is None:
                print('Skipping', target_text['name'], file=sys.stderr)
                continue
            out_filename = aligned_filename(target_text['name'])
            with gzip.open(out_filename, 'wb') as f:
                pickle.dump(
                        [source_text['name'] for source_text in source_texts],
                        f)
                pickle.dump(source_candidates, f)
            print('Wrote', target_text['name'], file=sys.stderr)
    print('Aligned files in', round(time.time()-t0, 2), 's', file=sys.stderr)


if __name__ == '__main__': main()

