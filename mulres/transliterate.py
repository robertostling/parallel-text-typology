"""Transliteration and phonological abstraction"""

import re
from unidecode import unidecode

NORMALIZED_ALPHABET = 'abcdefghijklmnopqrstuvwxyzNSX '
NORMALIZED_ALPHABET_RE = re.compile('['+NORMALIZED_ALPHABET+']+')

RE_Y_VOWEL = re.compile(r'y([aeiou])')
RE_DOUBLE_VOWEL = re.compile(r'(aa|ee|ii|oo|uu)')
RE_NONALPHA = re.compile(r'[^a-zA-Z ]+')

TR_MISC = str.maketrans('wS', 'us')
TR_NASALS = str.maketrans('mN', 'nn')
TR_DEVOICE = str.maketrans('bdgzv', 'ptksf')

# Convert UTF-8 string in any orthography to a (very rough approximation of)
# pseudo-phonetic transcription.
def normalize(s):
    s = unidecode(s).lower()
    s = RE_NONALPHA.sub('', s)
    s = RE_DOUBLE_VOWEL.sub(lambda m: m.group(1)[0], s)
    s = s.replace('sch', 'S')
    s = s.replace('sh', 'S')
    s = s.replace('ng', 'N')
    s = s.replace('ch', 'X')
    s = s.replace('kj', 'X')
    s = s.replace('ky', 'X')
    s = s.replace('gj', 'X')
    s = s.replace('gy', 'X')
    s = s.replace('c', 'X')
    s = s.replace('h', '') # ok?
    s = RE_Y_VOWEL.sub(lambda m: 'i'+m.group(1), s)
    return ''.join(NORMALIZED_ALPHABET_RE.findall(s))

def remove_distinctions(s, distinctions=(TR_NASALS, TR_DEVOICE, TR_MISC)):
    for d in distinctions:
        s = s.translate(d)
    return s


if __name__ == '__main__':
    import sys
    s = ' '.join(sys.argv[1:])
    print('Normalized 1:', normalize(s))
    print('Normalized 2:', remove_distinctions(normalize(s)))

