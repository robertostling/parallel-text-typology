import sys
from multiprocessing import Pool
from operator import itemgetter

from empfile import EncodedMPF, betabinomial_similarity

def load_lemma_empf(filename, verse_ids):
    empf = EncodedMPF(filename, verse_ids)
    empf.make_dummy_ngrams()
    empf.count_ngrams()
    #empf.make_ngram_positions(np.arange(len(empf.ngram_list)))
    return empf


def load_raw_empf(filename, verse_ids):
    empf = EncodedMPF(filename, verse_ids)
    empf.make_ngrams()
    empf.count_ngrams()
    return empf

def read_verse_ids(filename):
    with open(filename) as f: return list(map(str.strip, f))


def main():
    verse_ids = read_verse_ids('data/sentences.txt')
    #verse_idx = {verse: i for i, verse in enumerate(verse_ids)}

    target_filename = sys.argv[1]
    source_filenames = sys.argv[2:]

    with Pool() as p:
        source_empfs = p.starmap(
                load_raw_empf,
                [(filename, verse_ids) for filename in source_filenames],
                1)

    target_empf = load_lemma_empf(target_filename, verse_ids)

    while True:
        query = input('Input query (pos:lemma) -- ')
        try:
            query_i = target_empf.ngram_list.index(query)
        except ValueError:
            print('Not found!')
            continue
        target_empf.make_ngram_positions({query_i})
        target_verses = {
                verse_i
                for verse_i, _, _ in target_empf.ngram_positions[query_i]}

        for empf in source_empfs:
            real_target_verses, (result, count) = \
                    empf.find_ngrams_from_verses(target_verses)
            table = []
            for (ngram_i, both) in zip(result, count):
                k = empf.ngram_verse_count[ngram_i]
                l = len(real_target_verses)
                n_items = len(empf.ngram_list)
                total = empf.n_verses
                score = betabinomial_similarity(total, both, k, l, n_items)
                token_score = \
                        betabinomial_similarity(total, both, k, l, 1) / both
                table.append((empf.ngram_list[ngram_i], score, token_score,
                    both, k, l))
            table.sort(key=itemgetter(2))
            for ngram, score, token_score, both, k, l in table:
                print(ngram, round(score, 2), round(token_score, 2), both, k,
                        l)

if __name__ == '__main__': main()


