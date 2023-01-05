import sys
import gzip
import pickle
import glob
from multiprocessing import Pool
from empfile import EncodedMPF
from collections import Counter, defaultdict
import os.path


def token_alignments(lemma_empf, plain_empf, target_empf, lemma_candidates):
    assert lemma_empf.verse_ids == plain_empf.verse_ids
    assert lemma_empf.verse_ids == target_empf.verse_ids

    token_translations = defaultdict(Counter)

    for verse_i, verse in enumerate(lemma_empf.verse_ids):
        if lemma_empf.verse_tokens[verse_i] is None: continue
        if target_empf.verse_tokens[verse_i] is None: continue

        # target n-gram mapped to list of indexes in target verse
        ngram_positions = defaultdict(list)

        target_tokens = [target_empf.word_list[token_i]
                         for token_i in target_empf.verse_tokens[verse_i]]
        lemma_tokens = [lemma_empf.word_list[token_i]
                        for token_i in lemma_empf.verse_tokens[verse_i]]
        plain_tokens = [plain_empf.word_list[token_i]
                        for token_i in plain_empf.verse_tokens[verse_i]]

        assert len(lemma_tokens) == len(plain_tokens)

        target_links = [None]*len(target_tokens)

        for token_idx, token in enumerate(target_tokens):
            # token plus boundary markers
            s = '#' + token + '#'
            for i in range(len(s)-1):
                for j in range(i+2, len(s)+1):
                    ngram = s[i:j]
                    ngram_positions[ngram].append(token_idx)

        for lemma_i, (lemma, plain) in enumerate(
                zip(lemma_tokens, plain_tokens)):
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
                # this form seems acceptable, 
                for idx in ngram_positions[form]:
                    link = target_links[idx]
                    if link is None or saved_per_token > link[1]:
                        target_links[idx] = (plain, saved_per_token)
                        break

        for target, link in zip(target_tokens, target_links):
            if link is not None:
                plain, _ = link
                token_translations[target][plain] += 1

    return dict(token_translations)


def project_embeddings(
        target_filename,
        align_filename,
        source_lemma_filenames,
        source_plain_filenames,
        embeddings_filenames,
        verse_ids):

    target_empf = EncodedMPF(target_filename, verse_ids)
    with gzip.open(align_filename, 'rb') as f:
        filenames_list = pickle.load(f)
        candidates_list = pickle.load(f)
        source_candidates = dict(zip(filenames_list, candidates_list))


    target_vectors = defaultdict(lambda: np.zeros(300, np.float))
    target_n_aligned = Counter()

    for source_i, (lemma_filename, plain_filename) in enumerate(zip(
        source_lemma_filenames,
        source_plain_filenames)):

        lemma_empf = EncodedMPF(lemma_filename, verse_ids)
        plain_empf = EncodedMPF(plain_filename, verse_ids)
        lemma_candidates = source_candidates[os.path.basename(lemma_filename)]
        alignments = \
            token_alignments(
                lemma_empf, plain_empf, target_empf, lemma_candidates)

        print(os.path.basename(lemma_filename).split('.', 1)[0],
                'yields', len(alignments),
                'word forms aligned, of', len(target_empf.word_list))

        source_vocabulary = sorted(
                {word for counts in alignments.values()
                      for word in counts.keys()})
        source_e = Embeddings(embeddings_filenames[embedding_language],
                              source_vocabulary)

        for target_token, source_counts in alignments.items():
            for source_token, n in source_counts:
                v = source_e[source_token]
                if v is not None:
                    target_n_aligned[target_token] += n
                    target_vectors[target_token] += v*n

#def get_form_lemmas(lemma_filename, plain_filename, verse_ids):
#    lemma_empf = EncodedMPF(lemma_filename, verse_ids)
#    plain_empf = EncodedMPF(plain_filename, verse_ids)
#    form_lemmas = {}
#    for verse_i, verse in enumerate(verse_ids):
#        plain_tokens = plain_empf.verse_tokens[verse_i]
#        lemma_tokens = lemma_empf.verse_tokens[verse_i]
#        assert (plain_tokens is None) == (lemma_tokens is None)
#        if plain_tokens is None or lemma_tokens is None: continue
#        assert len(plain_tokens) == len(lemma_tokens)
#        for plain_i, lemma_i in zip(plain_tokens, lemma_tokens):
#            plain_str = plain_empf.word_list[plain_i]
#            lemma_str = lemma_empf.word_list[lemma_i]
#            form_lemmas.setdefault(plain_str, Counter())[lemma_str] += 1
#    return form_lemmas
#
#
#def project_embeddings(source_form_lemmas, align_filename):
#    with gzip.open(align_filename, 'rb') as f:
#        align_filenames = pickle.load(f)
#        align_candidates = pickle.load(f)
#
#    for name, candidates in zip(align_filenames, align_candidates):
#        print(name)
#        if name not in source_form_lemmas:
#            print('    SKIPPING')
#            continue
#        form_lemmas = source_form_lemmas[name]
#        lemma_forms = defaultdict(Counter)
#        for form, lemmas in list(form_lemmas.items())[:10]:
#            #print('   ', form, lemmas)
#            min_n = lemmas.most_common(1)[0][1]
#            top_lemmas = [
#                    (lemma, n) for lemma, n in lemmas.items() if n >= min_n]
#            for lemma, n in top_lemmas:
#                lemma_forms[lemma][form] += 1
#            #sum_n = sum(n for lemma, n in top_lemmas)
#
#        # TODO: average lemma vector according to distribution in forms, then
#        # add that to the types of the best-matching candidate in the target
#        # vocabulary -- for this we should probably load the EncodedMPF of the
#        # target text?
#        for lemma, forms in list(lemma_forms.items())[:10]:
#            print(lemma, forms)
#            print(candidates[lemma])
#            print()

def main():
    with open('data/sentences.txt') as f:
        verse_ids = f.read().split()

    # encode_texts.py plain ...
    source_plain_dir = 'encoded-conllu-plain'
    # encode_texts.py lemma ...
    source_lemma_dir = 'encoded-conllu'
    target_align_dir = 'aligned'

    smith_pattern = '/hd1/multilingual_resources/multilingual_embeddings/'
                    'wiki.%s.original.aligned.vec'


    source_lemma_filenames = glob.glob(
            os.path.join(source_lemma_dir, '*.pickle.gz'))
    # XXX
    #source_lemma_filenames = source_lemma_filenames[:3]
    # XXX
    source_plain_filenames = [
        os.path.join(source_plain_dir, os.path.basename(filename))
        for filename in source_lemma_filenames]

    for plain_filename in source_plain_filenames:
        assert os.path.exists(plain_filename), plain_filename

    #source_form_lemmas = {}

    #for source_i, (lemma_filename, plain_filename) in enumerate(zip(
    #    source_lemma_filenames,
    #    source_plain_filenames)):
    #    form_lemmas = get_form_lemmas(
    #        lemma_filename, plain_filename, verse_ids)
    #    source_name = os.path.basename(lemma_filename)
    #    source_form_lemmas[source_name] = form_lemmas
    #    print('Loaded', os.path.basename(lemma_filename))

    #align_filename = sys.argv[1]
    #project_embeddings(source_form_lemmas, align_filename)


    target_filename = sys.argv[1]
    align_filename = os.path.join(
            target_align_dir,
            os.path.basename(target_filename).split('.', 1)[0] + '.align.gz')

    project_embeddings(
        target_filename,
        align_filename,
        source_lemma_filenames,
        source_plain_filenames,
        verse_ids)

    #target_empf = EncodedMPF(target_filename, verse_ids)
    #with gzip.open(align_filename, 'rb') as f:
    #    align_filenames = pickle.load(f)
    #    align_candidates = pickle.load(f)

    #source_candidates = dict(zip(align_filenames, align_candidates))

    #for source_i, (lemma_filename, plain_filename) in enumerate(zip(
    #    source_lemma_filenames,
    #    source_plain_filenames)):

    #    lemma_empf = EncodedMPF(lemma_filename, verse_ids)
    #    plain_empf = EncodedMPF(plain_filename, verse_ids)
    #    lemma_candidates = source_candidates[os.path.basename(lemma_filename)]
    #    alignments = \
    #        token_alignments(
    #            lemma_empf, plain_empf, target_empf, lemma_candidates)

    #    for source_word, target_word_freqs in alignments.items():
    #        print(source_word, target_word_freqs)

    #    break

if __name__ == '__main__': main()

