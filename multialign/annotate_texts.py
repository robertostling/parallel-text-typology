import gzip
import pprint
import pickle
import sys
import os.path
from collections import defaultdict
from operator import itemgetter
import glob

from mpfile import MPFile

def iter_ngrams(word):
    s = '#' + word.casefold() + '#'
    for i in range(len(s)-1):
        for j in range(i+2, len(s)+1):
            yield i, s[i:j]

def main():
    lemma_dir = sys.argv[1]
    raw_dir = sys.argv[2]
    text_dir = sys.argv[3]

    with open('data/sentences.txt') as f:
        sent_ids = f.read().split()

    lemma_filenames = glob.glob(os.path.join(lemma_dir, '*.align.gz'))
    raw_filenames = glob.glob(os.path.join(raw_dir, '*.align.gz'))

    cluster_name_glosses = defaultdict(lambda: defaultdict(list))
    for filename in lemma_filenames:
        name = os.path.basename(filename).split('.', 1)[0]
        with gzip.open(filename, 'rb') as f:
            cluster_ngrams = pickle.load(f)
            for cluster_i, ngrams in enumerate(cluster_ngrams):
                ngrams.sort(key=itemgetter(2), reverse=True)
                for ngram, bayes_score, pos_score in ngrams:
                    cluster_name_glosses[cluster_i][name].append(
                            (ngram, bayes_score, pos_score))

    for filename in raw_filenames:
        name = os.path.basename(filename).split('.', 1)[0]
        ngram_best = {}
        ngram_glosses = {}
        with gzip.open(filename, 'rb') as f:
            cluster_ngrams = pickle.load(f)
            for cluster_i, ngrams in enumerate(cluster_ngrams):
                ngrams.sort(key=itemgetter(2), reverse=True)
                for ngram, bayes_score, pos_score in ngrams:
                    if ngram not in ngram_best or pos_score > ngram_best[ngram]:
                        ngram_best[ngram] = pos_score
                        ngram_glosses[ngram] = cluster_name_glosses[cluster_i]
                        break


        ngram_summary = defaultdict(list)
        for ngram, name_glosses in sorted(
                ngram_glosses.items(), key=itemgetter(0)):
            #print(ngram)
            for gloss_name, glosses in sorted(
                    name_glosses.items(), key=itemgetter(0)):
                if glosses[0][1] > 100.0:
                    ngram_summary[ngram].append(glosses[0][0])
                #print('   ', gloss_name, '--', ' '.join(t[0] for t in glosses))

        mpf_filename = os.path.join(text_dir, name + '.txt.gz')
        mpf = MPFile(mpf_filename, sent_ids=sent_ids)

        for verse_id, tokens in mpf.sentences.items():
            ann_tokens = []
            for token in tokens:
                summary = set()
                matches = []
                char_ngram = [None]*(len(token)+2)
                for i, ngram in sorted(iter_ngrams(token), key=lambda t: -len(t[1])):
                    j = i + len(ngram)
                    if ngram in ngram_summary:
                        if all(char_ngram[k] is None for k in range(i, j)):
                            for k in range(i, j):
                                char_ngram[k] = ngram
                for ngram in set(char_ngram)-{None}:
                    summary |= set(ngram_summary[ngram])
                    matches.append(ngram)
                if matches:
                    ann_tokens.append(token+'{'+'+'.join(matches)+'}'+'['+'/'.join(summary)+']')
                    if len(matches) > 1: ann_tokens.append('<--')
                else:
                    ann_tokens.append(token)

            print(verse_id, ' '.join(ann_tokens))

    # TODO: some kind of scoring of how well-matched n-grams are, for when
    #   multiple interpretations are possible
    # TODO: how to match to text?

if __name__ == '__main__': main()

