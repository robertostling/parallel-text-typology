"""Convenience script for ordering languages by family.

The input file should contain a list of ISO codes.
Output is one family per line, with the name of the family and the languages
it contains (from Glottolog).
"""

import sys
from collections import Counter, defaultdict

from langinfo.glottolog import Glottolog

def main():
    with open(sys.argv[1]) as f:
        isos = f.read().split()

    family_languages = defaultdict(list)
    uncategorized = []

    for iso in isos:
        try:
            l = Glottolog[iso]
            family = l.family_id
            if family is None:
                family_languages[l.name].append(l.name)
            else:
                family_languages[Glottolog[family].name].append(l.name)
        except KeyError:
            uncategorized.append(iso)

    for family, languages in sorted(family_languages.items()):
        print(family, '--', ', '.join(languages))

    print('Uncategorized:', ' '.join(uncategorized))

if __name__ == '__main__': main()

