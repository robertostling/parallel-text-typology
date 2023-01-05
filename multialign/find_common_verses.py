import sys
import gzip
from collections import Counter

from conllu import read_conllu

def get_conllu_sentences(filename):
    sentences = set()
    for _, sentence in read_conllu(filename):
        for fields in sentence:
            misc = dict(kv.split('=') for kv in fields[9].split('|'))
            verse = misc['ref']
            sentences.add(verse)
    return sentences

def get_mpf_sentences(filename):
    sentences = set()
    is_gzip = filename.endswith('.gz')
    with (gzip.open(filename, 'rt', encoding='utf-8') if is_gzip
            else open(filename, 'r', encoding='utf-8')) as f:
        for line in f:
            if '\t' in line and not line.startswith('#'):
                verse = line.split('\t', 1)[0]
                if verse.isnumeric():
                    sentences.add(verse)
    return sentences

def main():
    filenames = sys.argv[1:]
    conllu_files = [s for s in filenames if s.endswith('.conllu')]
    mpf_files = [s for s in filenames
                 if s.endswith('.txt') or s.endswith('.txt.gz')]

    print('Reading', len(conllu_files), 'CoNLL-U files')
    conllu_sentences = list(map(get_conllu_sentences, conllu_files))
    print('Reading', len(mpf_files), 'paralleltext files')
    mpf_sentences = list(map(get_mpf_sentences, mpf_files))

    print('Computing sentence set')

    sentence_count = Counter()
    for sentences in mpf_sentences:
        sentence_count.update(sentences)

    threshold = 0.8 * len(mpf_files)
    common_sentences = {sentence for sentence, n in sentence_count.items()
                                 if n > threshold}

    if conllu_sentences:
        required_sentences = conllu_sentences[0]
        for sentences in conllu_sentences[1:]:
            required_sentences &= sentences

        final_sentences = required_sentences & common_sentences

        print(len(required_sentences),
                'required sentences (from CoNLL-U files)')
    else:
        final_sentences = common_sentences

    print(len(common_sentences), 'sentences in >80% of translations')
    print(len(final_sentences), 'sentences in final output')

    with open('sentences.txt', 'w') as f:
        for sentence in sorted(final_sentences):
            print(sentence, file=f)


if __name__ == '__main__': main()

