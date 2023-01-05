import re
import os.path
import gzip
from collections import OrderedDict
import pickle

import numpy as np

from mulres.conllu import iterate_conllu


def is_word(s):
    """Return True iff s contains any alphanumeric character"""
    for c in s:
        if c.isalnum(): return True
    return False


class MPFile:
    """A file from the paralleltext repository

    Metadata is stored in the `metadata` attribute as an OrderedDict, with all
    keys converted to lower-case.

    The raw text (original tokenization) is stored as
    {verse_id: list of tokens} in the `sentences` attribute.

    The `annotations` dict contains annotations of the format
    { name: { verse_id: list of annotations } }
    where the annotation list has the same length as the corresponding
    sentence entry, sentences[verse_id].
    """

    def __init__(self, filename=None, **kwargs):
        self.metadata = OrderedDict()
        self.sentences = OrderedDict()
        self.annotations = {}
        if filename is not None:
            if filename.endswith('.conllu') or filename.endswith('.conllu.gz'):
                self.read_conllu(filename, **kwargs)
            else:
                self.read(filename, **kwargs)

    def write_numpy(self, path, sent_ids):
        def make_indexed_list(sent_items, transform=(lambda x: x)):
            vocab = sorted({transform(x) for sent_id in sent_ids
                                         if sent_id in sent_items
                                         for x in sent_items[sent_id]})
            if vocab and isinstance(vocab[0], int):
                data = [np.array(sent_items[sent_id], dtype=np.int32) \
                            if sent_id in sent_items else None
                        for sent_id in sent_ids]
                return None, None, data

            index = {x:i for i,x in enumerate(vocab)}
            data = [np.array([index[transform(x)] for x in sent_items[sent_id]],
                             dtype=np.int32) if sent_id in sent_items \
                                     else None
                    for sent_id in sent_ids]
            return vocab, index, data

        with gzip.open(path, 'wb') as f:
            pickle.dump(self.name, f)
            pickle.dump(sent_ids, f)

            token_list, _, verse_tokens = make_indexed_list(
                self.sentences, str.casefold)
            pickle.dump(('sentences', token_list, verse_tokens), f)

            for name, annotations in self.annotations.items():
                item_list, _, verse_items = make_indexed_list(annotations)
                pickle.dump((name, item_list, verse_items), f)


    def write(self, path, file_format, write_metadata=True):
        """Write the contents of this object to a file"""
        filename = os.path.join(path, '.'.join([self.name, file_format]))
        if file_format == 'txt.gz':
            with gzip.open(filename, 'wt', encoding='utf-8') as f:
                if write_metadata:
                    for k, v in self.metadata.items():
                        print('# {}: {}'.format(k, v), file=f)
                for k, v in self.sentences.items():
                    print(k + '\t' + ' '.join(v), file=f)
        elif file_format == 'turku':
            with open(filename, 'w', encoding='utf-8') as f:
                for k, v in self.sentences.items():
                    print('###C: verse = {}'.format(k), file=f)
                    print(' '.join(v), file=f)
        else: raise NotImplemented('Unknown file format: ' + file_format)


    def write_bitext(self, other, filename, index_filename, punctuation=False):
        re_word = re.compile('\w')

        common = sorted(set(self.sentences.keys()) &
                        set(other.sentences.keys()))
        with open(filename, 'w', encoding='utf-8') as outf, \
             open(index_filename, 'w') as indexf:
            for k in common:
                v1 = self.sentences[k]
                v2 = other.sentences[k]
                if not punctuation:
                    v1 = [token for token in v1 if re_word.search(token)]
                    v2 = [token for token in v2 if re_word.search(token)]
                if v1 and v2:
                    assert all(' ' not in token for token in v1)
                    assert all(' ' not in token for token in v2)
                    print(k, file=indexf)
                    print(' '.join(v1) + ' ||| ' + ' '.join(v2), file=outf)


    def read_conllu(self, filename, sent_ids=None, token_filter=None,
                    include_annotations=['lemma','pos','head','dep']):
        """Fill this object with data from a CoNLL-U file

        This could be either one of the manually annotated PROIEL-derived
        files from the paralleltext repo, or a Turku Neural Parsing Pipeline
        parsed file with 'verse = xxx' metadata.
        """
        verse = None
        self.name = os.path.basename(filename).split('.', 1)[0]
        head_map = {} if ('head' in include_annotations) else None
        head_reloc = [] if ('head' in include_annotations) else None

        for sent_i, (metadata, tokens) in enumerate(iterate_conllu(filename)):
            verse = metadata.get('verse', verse)
            for fields in tokens:
                misc = dict(kv.split('=') for kv in fields[9].split('|')
                            if '=' in kv)
                verse = misc.get('ref', verse)
                if verse is None:
                    raise ValueError('CoNLL-U file without verse identifiers')
                if sent_ids is None or verse in sent_ids:
                    if (token_filter is None) or token_filter(fields[1]):
                        sentence = self.sentences.setdefault(verse, [])
                        head = int(fields[6])
                        available_annotations = dict(
                            pos = fields[3],
                            head = head,
                            dep = fields[7],
                            lemma = fields[2])
                        token = fields[1]
                        if head_map is not None:
                            # any references to this token should be
                            # redirected to here:
                            head_map[(sent_i, int(fields[0]))] = \
                                    (verse, len(sentence))
                            # the head of this token needs to be redirected
                            if head:
                                head_reloc.append(
                                        (verse, len(sentence),
                                            sent_i, head))
                            else:
                                available_annotations['head'] = -1
                        sentence.append(token)
                        for name in include_annotations:
                            value = available_annotations.get(name)
                            if value is None: continue
                            if name not in self.annotations:
                                self.annotations[name] = OrderedDict()
                            annotations = self.annotations[name]
                            annotations.setdefault(verse, []).append(value)

        if head_reloc is not None:
            head = self.annotations['head']
            for verse, i, sent_i, idx in head_reloc:
                reloc = head_map.get((sent_i, idx))
                if reloc is None:
                    # token not present, e.g. non-word
                    new_i = -2
                else:
                    new_verse, new_i = reloc
                    assert verse == new_verse
                head[verse][i] = new_i


    def read(self, filename, only_metadata=False, sent_ids=None,
            token_filter=None):
        """Fill this object with data from a paralleltext format file.

        This modifies the `metadata` and `sentences` attributes.
        """
        if sent_ids is not None and not isinstance(sent_ids, set):
            sent_ids = set(sent_ids)
        self.name = os.path.basename(filename).split('.', 1)[0]
        re_metadata = re.compile(r'#\s*([^:]+):\s*(.*)$')
        with gzip.open(filename, 'rt', encoding='utf-8') \
                if filename.endswith('.gz') else \
                open(filename, 'r', encoding='utf-8') as f:
            for i,line in enumerate(f):
                line = line.strip()
                if line.startswith('#'):
                    m = re_metadata.match(line)
                    if m:
                        self.metadata[m.group(1).lower()] = m.group(2)
                else:
                    if only_metadata: return
                    fields = line.split('\t')
                    if len(fields) == 2:
                        sent_id, sent = fields
                        if sent_ids is None or sent_id in sent_ids:
                            tokens = sent.split()
                            if token_filter is None:
                                self.sentences[sent_id] = tokens
                            else:
                                self.sentences[sent_id] = [
                                        token for token in tokens
                                        if token_filter(token)]
                    elif len(fields) < 2:
                        pass
                    else:
                        raise ValueError(
                        'Expected comment or two-column line at %s:%d' % (
                            filename, i+1))
        if (not self.sentences) and (not only_metadata):
            raise ValueError('No valid sentences in file: ' + filename)


def main():
    import sys
    mpf = MPFile(sys.argv[1], token_filter=is_word)
    for verse, sentence in mpf.sentences.items():
        for i, token in enumerate(sentence):
            fields = [token]
            for name, annotation in mpf.annotations.items():
                fields.append(str(annotation[verse][i]))
            print('\t'.join(fields))
        print()

if __name__ == '__main__': main()

