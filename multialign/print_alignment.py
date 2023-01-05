import gzip
import pprint
import pickle
import sys
import os.path
from collections import defaultdict
from operator import itemgetter

def main():
    cluster_all = defaultdict(list)
    for filename in sys.argv[1:]:
        name = os.path.basename(filename).split('.', 1)[0]
        with gzip.open(filename, 'rb') as f:
            cluster_ngrams = pickle.load(f)
            for cluster_i, ngrams in enumerate(cluster_ngrams):
                ngrams.sort(key=itemgetter(2), reverse=True)
                for ngram, bayes_score, pos_score in ngrams[:1]:
                    cluster_all[cluster_i].append(
                            (name, ngram, bayes_score, pos_score))

    for ngrams in cluster_all.values():
        print('CLUSTER')
        ngrams.sort(key=itemgetter(0))
        for name, ngram, bayes_score, pos_score in ngrams:
            print('   ', name, ngram,
                    round(bayes_score, 2), round(pos_score, 3))

if __name__ == '__main__': main()

