import copy
import sys
import pickle
import gzip
from collections import defaultdict, Counter
import math
import os.path
import statistics
from multiprocessing import Pool
import random

import numpy as np

import numba
from numba import njit, typed

from conllu import read_conllu


@njit(numba.int32[:]( \
    numba.typeof(0), \
    numba.int32[:], numba.int32[:], numba.int32[:], \
    numba.int32[:], numba.int32[:], numba.int32[:]))
def _get_verse_unique_ngrams(
        verse_i,
        token_ngrams_table, token_ngrams_offset, token_ngrams_count,
        verse_tokens_table, verse_tokens_offset, verse_tokens_count):
    """Return an array of the unique n-grams in verse with index verse_i"""

    tokens_offset = verse_tokens_offset[verse_i]
    tokens_count = verse_tokens_count[verse_i]

    # Compute number of n-gram tokens
    n_ngram_tokens = 0
    for i in range(tokens_offset, tokens_offset+tokens_count):
        token_i = verse_tokens_table[i]
        ngrams_count = token_ngrams_count[token_i]
        n_ngram_tokens += ngrams_count

    # Empty array to hold all n-grams in this verse
    verse_ngrams = np.empty((n_ngram_tokens,), np.int32)
    k = 0
    for i in range(tokens_offset, tokens_offset+tokens_count):
        token_i = verse_tokens_table[i]
        ngrams_offset = token_ngrams_offset[token_i]
        ngrams_count = token_ngrams_count[token_i]
        for j in range(ngrams_offset, ngrams_offset+ngrams_count):
            ngram_i = token_ngrams_table[j]
            verse_ngrams[k] = ngram_i
            k += 1

    return np.unique(verse_ngrams)


@njit(numba.int32[:]( \
    numba.typeof(0), \
    numba.typeof(0), numba.typeof(0), \
    numba.int32[:], numba.int32[:], \
    numba.int32[:], numba.int32[:], numba.int32[:], \
    numba.int32[:], numba.int32[:], numba.int32[:]))
def _get_verse_unique_ngrams_constrained(
        verse_i,
        min_count, max_count,
        ngram_verse_count, word_verse_count,
        token_ngrams_table, token_ngrams_offset, token_ngrams_count,
        verse_tokens_table, verse_tokens_offset, verse_tokens_count):
    """Return an array of the unique n-grams in verse with index verse_i
    
    Args:
        min_count: int, ignore if zero, otherwise only include n-grams
            guaranteed to occur in at least this many verses
        max_count: int, ignore if zero, otherwise only include n-grams
            guaranteed to occur in at most this many verses
    """

    tokens_offset = verse_tokens_offset[verse_i]
    tokens_count = verse_tokens_count[verse_i]

    # Compute number of n-gram tokens (conservative estimate)
    n_ngram_tokens = 0
    for i in range(tokens_offset, tokens_offset+tokens_count):
        token_i = verse_tokens_table[i]
        if word_verse_count[token_i] > max_count:
            continue
        ngrams_count = token_ngrams_count[token_i]
        n_ngram_tokens += ngrams_count

    # Empty array to hold all n-grams in this verse
    verse_ngrams = np.empty((n_ngram_tokens,), np.int32)
    k = 0
    for i in range(tokens_offset, tokens_offset+tokens_count):
        token_i = verse_tokens_table[i]
        if word_verse_count[token_i] > max_count:
            continue
        ngrams_offset = token_ngrams_offset[token_i]
        ngrams_count = token_ngrams_count[token_i]
        for j in range(ngrams_offset, ngrams_offset+ngrams_count):
            ngram_i = token_ngrams_table[j]
            count = ngram_verse_count[ngram_i]
            if min_count <= count <= max_count:
                verse_ngrams[k] = ngram_i
                k += 1

    return np.unique(verse_ngrams[:k])


@njit(numba.int32[:]( \
    numba.typeof(0), \
    numba.int32[:], numba.int32[:], numba.int32[:], \
    numba.int32[:], numba.int32[:], numba.int32[:]))
def _count_ngrams(
        n_ngrams,
        token_ngrams_table, token_ngrams_offset, token_ngrams_count,
        verse_tokens_table, verse_tokens_offset, verse_tokens_count):
    ngram_verse_count = np.zeros((n_ngrams,), np.int32)

    for verse_i in range(verse_tokens_offset.shape[0]):
        verse_unique_ngrams = _get_verse_unique_ngrams(
            verse_i,
            token_ngrams_table, token_ngrams_offset, token_ngrams_count,
            verse_tokens_table, verse_tokens_offset, verse_tokens_count)

        for ngram_i in verse_unique_ngrams:
            ngram_verse_count[ngram_i] += 1

    return ngram_verse_count


@njit(( \
    numba.int32[:], \
    numba.int32[:], numba.int32[:], \
    numba.int32[:], numba.int32[:], numba.int32[:], \
    numba.int32[:], numba.int32[:], numba.int32[:]))
def _find_ngrams_from_verses(
        target_verses,
        ngram_verse_count, word_verse_count,
        token_ngrams_table, token_ngrams_offset, token_ngrams_count,
        verse_tokens_table, verse_tokens_offset, verse_tokens_count):
    """
    Args:
        target_verses: np.ndarray with verse indexes where the target concept
            is mentioned, this must exclude any verses not present in the
            current text
        ngram_verse_count: np.ndarray containing the number of verses in the
            current text that a given n-gram occurs in
        [rest as in _get_verse_unique_ngrams]
    """
    n_target_verses = target_verses.shape[0]
    min_count = (n_target_verses // 3) + 1
    max_count = (n_target_verses * 3)

    candidates_arrays = typed.List()
    remaining = typed.List()
    remaining_count = typed.List()

    for verse_i in target_verses:
        verse_unique_ngrams = _get_verse_unique_ngrams_constrained(
            verse_i,
            min_count, max_count,
            ngram_verse_count, word_verse_count,
            token_ngrams_table, token_ngrams_offset, token_ngrams_count,
            verse_tokens_table, verse_tokens_offset, verse_tokens_count)
        candidates_arrays.append(verse_unique_ngrams)

    candidates = np.array([x for one_array in candidates_arrays
                             for x in one_array])
    if len(candidates) == 0:
        return remaining, remaining_count
    candidates.sort()
    last_ngram_i = candidates[0]
    count = 1
    for i in range(1, len(candidates)):
        ngram_i = candidates[i]
        if ngram_i != last_ngram_i:
            total_count = ngram_verse_count[last_ngram_i]
            #assert count <= total_count
            if total_count <= max_count and min_count <= count:
                remaining.append(last_ngram_i)
                remaining_count.append(count)
            last_ngram_i = ngram_i
            count = 1
        else:
            count += 1

    total_count = ngram_verse_count[last_ngram_i]
    if total_count <= max_count and min_count <= count:
        remaining.append(last_ngram_i)
        remaining_count.append(count)

    return remaining, remaining_count


#@njit(( \
#    numba.typeof({0}), \
#    numba.int32[:], numba.int32[:], numba.int32[:], \
#    numba.int32[:], numba.int32[:], numba.int32[:]))
def _make_ngram_positions(
        include_ngrams,
        token_ngrams_table, token_ngrams_offset, token_ngrams_count,
        verse_tokens_table, verse_tokens_offset, verse_tokens_count):
    """Return a dict mapping ngrams to lists of (verse, token) positions
    
    Only ngrams in the set include_ngrams will be included. This contains
    integer indexes, as in token_ngrams_table.
    """
    ngram_positions = {}
    for verse_i in range(verse_tokens_offset.shape[0]):
        tokens_offset = verse_tokens_offset[verse_i]
        tokens_count = verse_tokens_count[verse_i]

        for i in range(tokens_offset, tokens_offset+tokens_count):
            token_i = verse_tokens_table[i]
            ngrams_offset = token_ngrams_offset[token_i]
            ngrams_count = token_ngrams_count[token_i]
            for j in range(ngrams_offset, ngrams_offset+ngrams_count):
                ngram_i = token_ngrams_table[j]
                if ngram_i in include_ngrams:
                    position = (verse_i, i-tokens_offset,
                                (i-tokens_offset)/tokens_count)
                    if ngram_i in ngram_positions:
                        ngram_positions[ngram_i].append(position)
                    else:
                        ngram_positions[ngram_i] = [position]
    return ngram_positions


class EncodedMPF:
    """Efficient representation of parallel text file

    Attributes:
        n_verses: int, number of verses in this text (i.e. number of non-None
            elements in verse_tokens)
        word_list: list of str, word types that make up the vocabulary
        verse_ids: list of str, containing the verse IDs present in this
            corpus but *NOT* necessarily in this text
        verse_tokens: list of same length as verse_ids, containing either None
            (if the verse is not present in this text) or an np.ndarray with
            indexes into word_list of the tokens that make up this verse
        verse_tokens_table, verse_tokens_offset, verse_tokens_count:
            analogue to token_ngrams_* below, but containing the tokens of
            each verse. For verses that are not present in this text, the
            corresponding verse_tokens_count entry is 0.
    Attributes created by make_ngrams():
        ngram_list: list of str containing character n-grams
        token_ngrams_table: np.ndarray, contains indexes into ngram_list
        token_ngrams_offset: np.ndarray, same size as word_list, containing
            indexes into token_ngrams_table so that the n-grams of
            word word_list[i] start at offset token_nrgams_offset[i] in
            token_ngrams_table
        token_ngrams_count: np.ndarray, corresponds to token_ngrams_offset
            but contains the number of n-grams in the word
    """
    def __init__(self, filename, verse_ids):
        """Read data created by MPFile.write_numpy"""
        with gzip.open(filename, 'rb') as f:
            self.verse_ids = verse_ids
            self.word_list = pickle.load(f)
            self.verse_tokens = pickle.load(f)

        self.name = os.path.basename(filename).split('.', 1)[0]

        self.verse_idx = {
                verse_ids[i]: i for i, tokens in enumerate(self.verse_tokens)
                if tokens is not None}
        self.n_verses = sum(int(tokens is not None)
                            for tokens in self.verse_tokens)
        self.verse_tokens_table = np.concatenate(
                [tokens for tokens in self.verse_tokens if tokens is not None])
        index = 0
        self.n_tokens = sum(len(tokens) for tokens in self.verse_tokens
                            if tokens is not None)
        verse_tokens_offset = []
        verse_tokens_count = []
        for tokens in self.verse_tokens:
            verse_tokens_offset.append(index)
            verse_tokens_count.append(0 if tokens is None else len(tokens))
            index += verse_tokens_count[-1]
        self.verse_tokens_offset = np.array(
                verse_tokens_offset, dtype=np.int32)
        self.verse_tokens_count = np.array(
                verse_tokens_count, dtype=np.int32)

        word_verse_count = np.zeros((len(self.word_list),), dtype=np.int32)
        for tokens in self.verse_tokens:
            if tokens is None: continue
            for token_i in frozenset(tokens):
                word_verse_count[token_i] += 1
        self.word_verse_count = word_verse_count


    def make_ngrams(self):
        ngram_idx = {}
        ngram_list = []
        token_ngrams_offset = []
        token_ngrams_count = []
        ngram_table = []
        ngram_count = [] # number of tokens where the n-gram occurs
        for token_i, token in enumerate(self.word_list):
            s = '#'+token+'#'
            n = len(s)
            token_ngrams_offset.append(len(ngram_table))
            token_ngrams_count.append((n*(n-1))//2)
            for i in range(n-1):
                for j in range(i+2, n+1):
                    ngram = s[i:j]
                    ngram_i = ngram_idx.get(ngram)
                    if ngram_i is None:
                        ngram_i = len(ngram_idx)
                        ngram_idx[ngram] = ngram_i
                        ngram_list.append(ngram)
                        ngram_count.append(0)
                    ngram_table.append(ngram_i)
                    ngram_count[ngram_i] += 1

        # Only keep the longest n-gram in case of identical distributions
        ignore = set()
        for ngram in sorted(ngram_list, key=len, reverse=True):
            if ngram in ignore: continue
            n = len(ngram)
            ngram_i = ngram_idx[ngram]
            count = ngram_count[ngram_i]
            for i in range(n-1):
                for j in range(i+2, n+1):
                    if i == 0 and j == n: continue
                    sub_ngram = ngram[i:j]
                    sub_ngram_i = ngram_idx[sub_ngram]
                    if ngram_count[sub_ngram_i] == count:
                        ignore.add(sub_ngram)

        #print(self.name, 'removed', len(ignore), 'of', len(ngram_table),
        #        'n-grams')

        new_ngram_list = sorted(set(ngram_list)-ignore)
        new_ngram_idx = {ngram: i for i, ngram in enumerate(new_ngram_list)}
        new_ngram_table = []
        new_offset = 0
        for token_i in range(len(token_ngrams_offset)):
            offset = token_ngrams_offset[token_i]
            n = token_ngrams_count[token_i]
            new_n = 0
            for i in range(offset, offset+n):
                ngram_i = ngram_table[i]
                ngram = ngram_list[ngram_i]
                if ngram not in ignore:
                    new_ngram_table.append(new_ngram_idx[ngram])
                    new_n += 1
            token_ngrams_offset[token_i] = new_offset
            token_ngrams_count[token_i] = new_n
            new_offset += new_n

        total_ngrams = 0
        for token_i in self.verse_tokens_table:
            total_ngrams += token_ngrams_count[token_i]
        #print(self.name, 'has', total_ngrams, 'n-gram tokens')

        ngram_table = new_ngram_table
        ngram_list = new_ngram_list

        self.token_ngrams_table = np.array(ngram_table, dtype=np.int32)
        self.token_ngrams_offset = np.array(
                token_ngrams_offset, dtype=np.int32)
        self.token_ngrams_count = np.array(
                token_ngrams_count, dtype=np.int32)
        self.ngram_list = ngram_list

    def make_dummy_ngrams(self):
        """Like make_ngrams() but only full words are copied"""
        self.ngram_list = list(self.word_list)
        self.token_ngrams_table = np.arange(len(self.ngram_list),
                dtype=np.int32)
        self.token_ngrams_offset = np.arange(len(self.ngram_list),
                dtype=np.int32)
        self.token_ngrams_count = np.ones((len(self.ngram_list),),
                dtype=np.int32)

    def count_ngrams(self):
        self.ngram_verse_count = _count_ngrams(
            len(self.ngram_list),
            self.token_ngrams_table,
            self.token_ngrams_offset, self.token_ngrams_count,
            self.verse_tokens_table,
            self.verse_tokens_offset, self.verse_tokens_count)

    def make_ngram_positions(self, include_ngrams):
        self.ngram_positions = _make_ngram_positions(
            include_ngrams,
            self.token_ngrams_table,
            self.token_ngrams_offset, self.token_ngrams_count,
            self.verse_tokens_table,
            self.verse_tokens_offset, self.verse_tokens_count)

    def find_ngrams_from_verses(self, target_verses):
        target_verses = np.unique(np.array(
                [verse_i for verse_i in target_verses
                         if self.verse_tokens[verse_i] is not None],
                dtype=np.int32))

        return _find_ngrams_from_verses(
            target_verses,
            self.ngram_verse_count, self.word_verse_count,
            self.token_ngrams_table,
            self.token_ngrams_offset, self.token_ngrams_count,
            self.verse_tokens_table,
            self.verse_tokens_offset, self.verse_tokens_count)


def read_verse_ids(filename):
    with open(filename) as f: return list(map(str.strip, f))


def is_word(s):
    for c in s:
        if c.isalnum(): return True
    return False


def read_lemma_distributions(filename):
    interesting_feats = set(
            'Number Case Degree Definite Tense Aspect Mood '
            'Evident Polarity Person'.split())
    conllu_sentences = read_conllu(filename)
    verse_lemmas = defaultdict(list)
    lemma_verses = defaultdict(list)
    verse = None
    for metadata, sentence in conllu_sentences:
        verse = metadata.get('verse', verse)
        for fields in sentence:
            if is_word(fields[1]):
                misc = dict(kv.split('=') for kv in fields[9].split('|')
                            if '=' in kv)
                verse = misc.get('ref', verse)
                lemma = fields[2]
                pos = fields[3]
                verse_lemmas[verse].append((pos, lemma))
                # This mostly adds noise:
                #for kv in fields[5].split('|'):
                #    kvs = kv.split('=')
                #    if len(kvs) == 2:
                #        k, v = kvs
                #        if k in interesting_feats:
                #            verse_lemmas[verse].append(('MARK', kv))

    for verse, lemmas in verse_lemmas.items():
        for i, (pos, lemma) in enumerate(lemmas):
            lemma_verses[(pos, lemma)].append((verse, i/(len(lemmas)-1)))

    return lemma_verses


def logll_dirichlet_multinomial(alpha, n, x):
    assert len(alpha) == len(x)
    assert n == sum(x)
    z = math.lgamma(sum(alpha)) - math.lgamma(n + sum(alpha))
    return z + sum(math.lgamma(x[k] + alpha[k]) - math.lgamma(alpha[k])
                   for k in range(len(x)))


def betabinomial_similarity(total, both, k, l, n_items):
    """Similarity score equal to log of P(equivalent)/P(independent)

    Args:
        total: total number of verses
        both: verses with both items
        k: verses with item 1 (e.g. cluster distribution)
        l: verses with item 2 (e.g. n-gram distribution)
        n_items: number of items to choose from (used as uniform prior)
    """
    assert total >= k, (total, both, k, l)
    assert k >= both, (total, both, k, l)
    assert l >= both, (total, both, k, l)
    assert total >= (k+l-both), (total, both, k, l)
    log_p_independent = logll_dirichlet_multinomial(
            [1.0, 1.0], total, [k, total-k]) + \
                        logll_dirichlet_multinomial(
            [1.0, 1.0], total, [l, total-l])
    log_p_joint = logll_dirichlet_multinomial(
            [1.0, 1.0, 1.0, 1.0], total,
            [k-both, l-both, both, total-(k+l-both)])
    log_p_prior = -math.log(n_items)
    return log_p_prior + log_p_joint - log_p_independent


def positional_similarity(a, b):
    """Refined similarity measure given full distributions

    verse_i, rel_pos are estimated positions (verse index, relative position
    of token within the verse, range [0, 1]) of a concept occurrence.

    a is the reference distribution, and also includes the probability p
    in the range [0, 1] indicating how likely we estimate that the particular
    concept is expressed at this location

    Args:
        a: dict of { int verse_i: list (float rel_pos, float p) }
        b: list of (int verse_i, float rel_pos)
    """
    # Lists in a will be modified, so we need copy
    a_copy = copy.deepcopy(a)
    sum_score = 0.0
    for verse_i, b_pos in b:
        if verse_i in a_copy:
            a_list = a_copy[verse_i]
            if not a_list: continue
            scores = [p*(1.0-abs(a_pos-b_pos)) for a_pos, p in a_list]
            best_match = np.argmin(scores)
            a_list.pop(best_match)
            sum_score += scores[best_match]
    # Ratio of (weighted) occurrences identified that were correct
    precision = sum_score / len(b)
    # Ratio of (weighted) occurrences correctly identified
    recall = sum_score / sum(p for positions in a.values()
                               for _, p in positions)
    # precision = min(1.0, precision)
    # recall = min(1.0, recall)
    assert 0.0 <= precision <= 1.0, (precision, a, b)
    assert 0.0 <= recall <= 1.0, (recall, a, b)
    return 0.0 if precision == 0 and recall == 0 else \
            2*precision*recall / (precision+recall)


def find_cluster_candidates(out_dir, filename, verse_ids, dummy_ngrams,
                            cluster_dists, string_similarity):
    """Find candidate n-grams of a set of clusters in one translation

    Args:
        out_dir: directory to write pickled output
        filename: passed to EncodedMPF
        verse_ids: passed to EncodedMPF (set of str)
        dummy_ngrams: if True, use full words rather than char n-grams
        dist: list of lists of (verse_i, rel_pos, p)
        string_similarity: list of bools, indicating whether string similarity
                           should count for each cluster
    Returns:
        list of lists of tuples
            (ngram, (n_joint_verses, n_ngram_verses, n_target_verses))
        where the ngram is a str, remaining are integer counts
    """
    empf = EncodedMPF(filename, verse_ids)
    if dummy_ngrams: empf.make_dummy_ngrams()
    else: empf.make_ngrams()
    empf.count_ngrams()
    def find_ngrams(target_verses):
        result, count = empf.find_ngrams_from_verses(target_verses)
        ngram_stats = {}
        ngram_stats_by_length = []
        n_target_verses = sum(int(empf.verse_tokens[verse_i] is not None)
                              for verse_i in target_verses)
        for i in range(len(result)):
            ngram_i = result[i]
            n_joint_verses = count[i]
            n_ngram_verses = empf.ngram_verse_count[result[i]]
            min_verses = (max(n_ngram_verses, n_target_verses) // 3) + 1
            max_verses = (min(n_ngram_verses, n_target_verses) * 3)
            if not (min_verses <= n_joint_verses <= max_verses):
                continue
            counts = (n_joint_verses, n_ngram_verses, n_target_verses)
            ngram = empf.ngram_list[ngram_i]
            ngram_stats[ngram] = counts
            # NOTE: ugly hack because ngram_stats needs to be indexed by ngram
            # str only, while ngram_stats_by_length must contain ngram_i
            ngram_stats_by_length.append((ngram_i, ngram, counts))

        ngram_stats_by_length.sort(key=lambda t: len(t[1]), reverse=True)
        final_ngrams = []
        for ngram_i, ngram, counts in ngram_stats_by_length:
            score = betabinomial_similarity(
                    *((empf.n_verses,) + counts + (len(empf.ngram_list),)))
            if score < 0.0: continue
            final_ngrams.append((ngram_i, counts, score))
        final_ngrams.sort(key=lambda t: -t[-1])
        return final_ngrams


    cluster_verses = []
    for dist in cluster_dists:
        verse_p_sum = Counter()
        # Consider a verse to contain the concept of this cluster iff the
        # expected number of occurrences is at least 0.5
        for verse_i, _, p in dist:
            verse_p_sum[verse_i] += p
        cluster_verses.append([
            verse_i for verse_i, p_sum in verse_p_sum.items() if p_sum >= 0.5])

    cluster_ngrams = [find_ngrams(target_verses)
                      for target_verses in cluster_verses]

    relevant_ngrams = {ngram_i for ngrams in cluster_ngrams
                               for ngram_i, counts, score in ngrams}


    empf.make_ngram_positions(relevant_ngrams)

    # Data suitable as first argument to positional_similarity()
    cluster_a = []
    for dist in cluster_dists:
        a = {}
        for verse_i, rel_pos, p in dist:
            if verse_i not in a: a[verse_i] = []
            a[verse_i].append((rel_pos, p))
        cluster_a.append(a)

    out_filename = os.path.join(out_dir, empf.name + '.align.gz')

    #result = [[(empf.ngram_list[ngram_i], counts, score,
    #          empf.ngram_positions[ngram_i])
    #         for ngram_i, counts, score in ngrams]
    #        for ngrams in cluster_ngrams]

    result = [[(empf.ngram_list[ngram_i], score,
                positional_similarity(a,
                    [(verse_i, rel_pos) for verse_i, _, rel_pos
                        in empf.ngram_positions[ngram_i]]))
               for ngram_i, counts, score in ngrams]
              for ngrams, a in zip(cluster_ngrams, cluster_a)]

    with gzip.open(out_filename, 'wb') as f:
        pickle.dump(result, f)

    print('Wrote', out_filename)


def load_lemma_empf(filename, verse_ids):
    empf = EncodedMPF(filename, verse_ids)
    empf.make_dummy_ngrams()
    empf.count_ngrams()
    empf.make_ngram_positions(np.arange(len(empf.ngram_list)))
    return empf


def main():
    import time
    import pprint
    import glob

    verse_ids = read_verse_ids('data/sentences.txt')
    verse_idx = {verse: i for i, verse in enumerate(verse_ids)}

    lemma_dir = sys.argv[1]
    raw_dir = sys.argv[2]
    align_dir = sys.argv[3]

    assert os.path.isdir(align_dir)

    lemma_filenames = glob.glob(os.path.join(lemma_dir, '*.pickle.gz'))
    raw_filenames = glob.glob(os.path.join(raw_dir, '*.pickle.gz'))

    initial_concepts_filename = 'initial-concepts.pickle'

    if os.path.exists(initial_concepts_filename):
        with open(initial_concepts_filename, 'rb') as f:
            empf_names = pickle.load(f)
            assert verse_ids == pickle.load(f)
            str_concepts = pickle.load(f)
    else:
        print('Reading annotated texts')
        t0 = time.time()
        with Pool() as p:
            lemma_empfs = p.starmap(
                    load_lemma_empf,
                    [(filename, verse_ids) for filename in lemma_filenames],
                    1)
        print('Done in', round(time.time()-t0, 2), 's')

        vocabulary_sizes = [len(empf.word_list) for empf in lemma_empfs]
        print('Total vocabulary size:', sum(vocabulary_sizes))
        print('Individual sizes:',
                [len(empf.word_list) for empf in lemma_empfs])

        potential_concepts = set()
        concepts = []

        for empf_i, empf in enumerate(lemma_empfs):
            for word_i, n in enumerate(empf.word_verse_count):
                if n >= 3:
                    # Note: since we have used make_dummy_ngrams() on these
                    # empf objects, word_i is equivalent to ngram_i
                    potential_concepts.add((empf_i, word_i))

        print('Potential concept-defining lemmas:', len(potential_concepts))

        pair_n_verses = { (empf1_i, empf2_i):
                sum(int(empf1.verse_tokens[verse_i] is not None and
                        empf2.verse_tokens[verse_i] is not None)
                    for verse_i in range(len(verse_ids)))
                for empf1_i, empf1 in enumerate(lemma_empfs)
                for empf2_i, empf2 in enumerate(lemma_empfs) }

        while potential_concepts:
            empf_i, word_i = potential_concepts.pop()
            concepts.append([(empf_i, word_i)])
            empf = lemma_empfs[empf_i]
            for other_empf_i, other_empf in enumerate(lemma_empfs):
                if other_empf_i == empf_i: continue
                target_verses = [verse_i for verse_i, _, _ in
                        lemma_empfs[empf_i].ngram_positions[word_i]
                        if other_empf.verse_tokens[verse_i] is not None]
                n_target_verses = len(set(target_verses))
                n_verses = pair_n_verses[(empf_i, other_empf_i)]
                result, count = other_empf.find_ngrams_from_verses(
                        target_verses)
                best_score = 0.0
                best_ngram_i = None
                for ngram_i, n_joint_verses in zip(result, count):
                    if (other_empf_i, ngram_i) not in potential_concepts:
                        continue
                    n_ngram_verses = other_empf.ngram_verse_count[ngram_i]
                    min_verses = (max(n_ngram_verses, n_target_verses)//3) + 1
                    max_verses = (min(n_ngram_verses, n_target_verses)*3)
                    if not (min_verses <= n_joint_verses <= max_verses):
                        continue
                    score = betabinomial_similarity(
                            n_verses, n_joint_verses, n_ngram_verses,
                            n_target_verses, len(other_empf.ngram_list))
                    if score >= min(100, best_score):
                        best_score = score
                        best_ngram_i = ngram_i
                if best_ngram_i is not None:
                    concepts[-1].append((other_empf_i, best_ngram_i))
                    try:
                        potential_concepts.remove((other_empf_i, best_ngram_i))
                    except KeyError:
                        pass

            #print(len(potential_concepts), 'left')
            #print(len(concepts), len(concepts[-1]), len(potential_concepts))
            #print('   ', ' '.join(lemma_empfs[empf_i].name[:3]+':'+lemma_empfs[empf_i].ngram_list[ngram_i]
            #    for empf_i, ngram_i in concepts[-1]))

        str_concepts = [
                [(empf_i, lemma_empfs[empf_i].ngram_list[ngram_i],
                    lemma_empfs[empf_i].ngram_positions[ngram_i])
                    for empf_i, ngram_i in members]
                for members in concepts
                if len(members) >= 2]

        with open(initial_concepts_filename, 'wb') as f:
            pickle.dump([empf.name for empf in lemma_empfs], f)
            #pickle.dump([[tokens is not None for tokens in empf.verse_tokens]
            #             for empf in lemma_empfs], f)
            pickle.dump(verse_ids, f)
            pickle.dump(str_concepts, f)

        print('Wrote', len(str_concepts), 'concepts')


    print('Total number of concepts:', len(str_concepts))
    #for i in range(2, 11):
    #    print('  with at least', i, 'members:',
    #            sum(int(len(members) >= i) for members in str_concepts))

    #pprint.pprint(Counter(
    #    t[1].split(':', 1)[0] for members in str_concepts
    #                          for t in members))

    # Compute initial concept centroids
    #
    # We keep PROPN-tagged concepts distinct, because string similarity is
    # generally useful there.
    general_concepts = []
    propn_concepts = []

    for members in str_concepts:
        tags = Counter(ngram.split(':', 1)[0] for _, ngram, _ in members)
        main_tag, tag_count = tags.most_common(1)[0]
        verse_member_positions = defaultdict(lambda: defaultdict(list))
        #members_have_verse = Counter()
        for i, (_, ngram, positions) in enumerate(members):
            tag = ngram.split(':', 1)[0]
            # Easy way to filter out some noise, although not 100% accurate
            # since e.g. demonyms may be ADJ/PROPN across languages
            if main_tag == 'propn' and tag != 'propn': continue
            #for verse_i in {verse_i for verse_i, _, _ in positions}:
            #    members_have_verse[verse_i] += 1
            for verse_i, _, rel_pos in positions:
                verse_member_positions[verse_i][i].append(rel_pos)
        dist = []
        # There may be different number of instances for a this concept in a
        # given verse, between different translations.
        # We now need to reconcile this, and estimate mean positions of each
        # instance.
        for verse_i, member_positions in verse_member_positions.items():
            # Maximum number of instances
            n = max(map(len, member_positions.values()))
            # The approach is to let the translations with the largest
            # number of instances go first in the inner loop, and so decide
            # preliminary centroids of the instances. Later translations with
            # fewer instances need to assign them to the closest available
            # centroids.
            final_positions = [[] for _ in range(n)]
            #print(n, member_positions.values())
            for positions in sorted(member_positions.values(), key=len,
                                    reverse=True):
                if len(positions) == n:
                    for i, rel_pos in enumerate(positions):
                        final_positions[i].append(rel_pos)
                else:
                    j = 0 # position to copy to
                    for i, rel_pos in enumerate(positions):
                        while (j-i) < (n-len(positions)):
                            this_mean = statistics.mean(final_positions[j])
                            next_mean = statistics.mean(final_positions[j+1])
                            if abs(rel_pos-this_mean) > abs(rel_pos-next_mean):
                                j += 1
                            else:
                                break
                        final_positions[j].append(rel_pos)
                        j += 1
            for positions in final_positions:
                dist.append((verse_i, statistics.mean(positions),
                             len(positions)/len(members)))
                assert len(positions) <= len(members), (positions, members)
        if main_tag == 'propn': propn_concepts.append(dist)
        else: general_concepts.append(dist)

    print('General concepts:', len(general_concepts))
    print('Name concepts:', len(propn_concepts))

    align_raw_dir = os.path.join(align_dir, 'raw')
    align_lemma_dir = os.path.join(align_dir, 'lemma')

    os.makedirs(align_raw_dir, exist_ok=True)
    os.makedirs(align_lemma_dir, exist_ok=True)

    # XXX
    #raw_filenames = random.sample(raw_filenames, 4)
    #lemma_filenames = random.sample(lemma_filenames, 4)
    #lemma_filenames = [filename for filename in lemma_filenames
    #        if os.path.basename(filename)[:3] in 'swe dan nob eng nno deu'.split()]
    #raw_filenames = [filename for filename in raw_filenames
    #        if os.path.basename(filename)[:3] in 'fao kal nan'.split()]

    t0 = time.time()
    with Pool() as p:
        tasks = []
        for lemmatized in (True, False):
            filenames = lemma_filenames if lemmatized else raw_filenames
            out_dir = align_lemma_dir if lemmatized else align_raw_dir
            string_similarity = [False]*len(general_concepts) + \
                                [True]*len(propn_concepts)
            concepts = general_concepts + propn_concepts
            tasks.extend([
                (out_dir, filename, verse_ids, lemmatized, concepts,
                    string_similarity)
                for filename in filenames])
        print('Aligning ({} tasks)'.format(len(tasks)))
        p.starmap(find_cluster_candidates, tasks, 1)
    print('Done in', round(time.time()-t0, 2), 's')


if __name__ == '__main__': main()

