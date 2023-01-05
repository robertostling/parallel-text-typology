# TODO: special heuristic for hapax legomena? Not covered at all currently and
# may be misaligned. (nålsöga)


import sys
from multiprocessing import Pool
from operator import itemgetter
from collections import namedtuple, defaultdict
import time
import os.path

import numpy as np

from empfile import EncodedMPF, betabinomial_similarity

Candidate = namedtuple('Candidate', 'index bayes_factor saved_per_token')

def load_lemma_empf(filename, verse_ids):
    empf = EncodedMPF(filename, verse_ids)
    empf.make_dummy_ngrams()
    empf.count_ngrams()
    empf.make_ngram_positions(np.arange(len(empf.ngram_list)))
    return empf


def load_raw_empf(filename, verse_ids):
    empf = EncodedMPF(filename, verse_ids)
    empf.make_ngrams()
    empf.count_ngrams()
    return empf

def read_verse_ids(filename):
    with open(filename) as f: return list(map(str.strip, f))


def align_raw(target_filename, source_filenames, verse_ids):
    target_empf = load_raw_empf(target_filename, verse_ids)

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

    for source_filename in source_filenames:
        source_empf = load_lemma_empf(source_filename, verse_ids)
        source_candidates.append({
                source_empf.ngram_list[ngram_i]:
                    align_ngram(source_empf, ngram_i)
                for ngram_i in range(len(source_empf.ngram_list))})

    return target_filename, source_candidates


def star_align_raw(args):
    return align_raw(*args)


def main():
    import gzip
    import glob
    import pickle

    verse_ids = read_verse_ids('data/sentences.txt')
    #verse_idx = {verse: i for i, verse in enumerate(verse_ids)}

    out_dir = sys.argv[1]
    target_dir = sys.argv[2]
    source_filenames = sys.argv[3:]

    assert os.path.isdir(out_dir)
    assert os.path.isdir(target_dir)

    target_filenames = glob.glob(os.path.join(target_dir, '*.pickle.gz'))

    # TODO call align_raw(), save alignments, use them when loading CoNLL-U
    # files to project or map word forms to lemmas
    # TODO transliterate and compare string similarity for proper names

    # XXX
    #target_filenames = target_filenames[:2]

    print('Aligning', len(target_filenames), 'targets', file=sys.stderr)

    t0 = time.time()
    with Pool() as p:
        tasks = [(target_filename, source_filenames, verse_ids)
                 for target_filename in target_filenames]
        for filename, source_candidates in p.imap(star_align_raw, tasks, 1):
            out_filename = os.path.join(
                    out_dir,
                    os.path.basename(filename).split('.', 1)[0] + '.align.gz')
            with gzip.open(out_filename, 'wb') as f:
                pickle.dump([os.path.basename(source_filename)
                    for source_filename in source_filenames], f)
                pickle.dump(source_candidates, f)
    print('Aligned files in', round(time.time()-t0, 2), 's', file=sys.stderr)

    # def align_raw(target_filename, source_filenames, verse_ids):

    #t0 = time.time()
    #with Pool() as p:
    #    source_empfs = p.starmap(
    #            load_lemma_empf,
    #            [(filename, verse_ids) for filename in source_filenames],
    #            1)

    #target_empf = load_raw_empf(target_filename, verse_ids)

    #print('Loaded files in', round(time.time()-t0, 2), 's', file=sys.stderr)

    #source_empfs.sort(key=lambda empf: empf.name)

    #name_source_empf = {empf.name: empf for empf in source_empfs}

    ## map from (source_empf.name, ngram_i) to list of potentially matching
    ## ngrams in target_empf
    #align_cache = {}

    #def cached_align(source_empf, query_ngram_i):
    #    key = (source_empf.name, query_ngram_i)
    #    if key in align_cache: return align_cache[key]

    #    query_verses = {
    #        verse_i
    #        for verse_i, _, _ in source_empf.ngram_positions[query_ngram_i]}

    #    real_query_verses, (result, count) = \
    #            target_empf.find_ngrams_from_verses(
    #                    query_verses,
    #                    low_limit=2, high_limit=2)

    #    l = len(real_query_verses)
    #    n_items = len(target_empf.ngram_list)
    #    total = source_empf.n_verses

    #    ngram_verse_count = target_empf.ngram_verse_count
    #    ks = target_empf.ngram_verse_count[result]

    #    table = []
    #    for (ngram_i, both, k) in zip(result, count, ks):
    #        bayes_factor = betabinomial_similarity(total, both, k, l, n_items)
    #        saved_per_token = \
    #                betabinomial_similarity(total, both, k, l, 1) / both
    #        table.append(Candidate(
    #            ngram_i, bayes_factor=bayes_factor,
    #            saved_per_token=saved_per_token))

    #    table.sort(key=lambda x: -x.saved_per_token)

    #    align_cache[key] = table
    #    return table

    #t0 = time.time()
    #for verse_i, verse in enumerate(verse_ids):
    #    if target_empf.verse_tokens[verse_i] is None:
    #        continue

    #    #if not verse.startswith('4001902'): continue
    #    print(verse)

    #    ngram_tokens = defaultdict(list)
    #    for token_pos, token_i in enumerate(target_empf.verse_tokens[verse_i]):
    #        offset = target_empf.token_ngrams_offset[token_i]
    #        count = target_empf.token_ngrams_count[token_i]
    #        for i in range(offset, offset+count):
    #            ngram_i = target_empf.token_ngrams_table[i]
    #            ngram_tokens[ngram_i].append(token_pos)

    #    # TODO: number of sources know, use integer indexing instead of dict
    #    # in inner loop
    #    token_sources = [{} for _ in target_empf.verse_tokens[verse_i]]

    #    for source_empf in source_empfs:
    #        if source_empf.verse_tokens[verse_i] is None:
    #            continue
    #        for source_token_pos, token_i in enumerate(
    #                source_empf.verse_tokens[verse_i]):
    #            candidates = cached_align(source_empf, token_i)
    #            for candidate in candidates:
    #                # Threshold of 1 bit (~0.693 nats)
    #                if candidate.saved_per_token < 0.693:
    #                    break
    #                if candidate.index in ngram_tokens:
    #                    for target_token_pos in ngram_tokens[candidate.index]:
    #                        sources_match = token_sources[target_token_pos]
    #                        if source_empf.name in sources_match:
    #                            _, old_candidate = \
    #                                    sources_match[source_empf.name]
    #                            if candidate.saved_per_token > \
    #                                    old_candidate.saved_per_token:
    #                                sources_match[source_empf.name] = \
    #                                        (token_i, candidate)
    #                        else:
    #                            sources_match[source_empf.name] = \
    #                                    (token_i, candidate)
    #                    break

    #    for token_pos, token_i in enumerate(target_empf.verse_tokens[verse_i]):
    #        if token_sources[token_pos]:
    #            best_saved_per_token = max(
    #                    candidate.saved_per_token
    #                    for _, candidate in token_sources[token_pos].values())

    #            # TODO:  continue from here....
    #            annotations = [
    #                (source_name[:3],
    #                name_source_empf[source_name].ngram_list[source_token_i],
    #                target_empf.ngram_list[candidate.index])
    #                for source_name, (source_token_i, candidate)
    #                    in token_sources[token_pos].items()
    #                    if candidate.saved_per_token >=
    #                        best_saved_per_token*0.5]

    #            links = [
    #                '%s:%s=%s/%.2f' % (
    #                    source_name[:3],
    #                    name_source_empf[source_name].ngram_list[source_token_i],
    #                    target_empf.ngram_list[candidate.index],
    #                    candidate.saved_per_token)
    #                for source_name, (source_token_i, candidate)
    #                    in token_sources[token_pos].items()
    #                    if candidate.saved_per_token >=
    #                        best_saved_per_token*0.5]

    #            print(target_empf.word_list[token_i]+'['+' '.join(links)+']')
    #        else:
    #            print(target_empf.word_list[token_i])
    #    print()

    #print('Done in', round(time.time()-t0, 2), 's')

if __name__ == '__main__': main()


