"""Managing Intercontinental Dictionary Series (IDS) data"""

import csv
import os.path
from collections import defaultdict

class IDS:
    def __init__(self, data_path):
        language_code_iso = {}
        with open(os.path.join(data_path, 'languages.csv'), 'r',
                  encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                iso = row['ISO639P3code'].strip()
                code = row['ID'].strip()
                if len(iso) != 3: continue
                language_code_iso[code] = iso

        concept_code_concepticon = {}
        with open(os.path.join(data_path, 'parameters.csv'), 'r',
                  encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row['ID'].strip()
                concepticon = row['Concepticon_ID'].strip()
                int(concepticon) # assert it should be an integer
                concept_code_concepticon[code] = concepticon

        iso_form_concepts = defaultdict(lambda: defaultdict(set))
        iso_form_concepticon = defaultdict(lambda: defaultdict(set))
        with open(os.path.join(data_path, 'forms.csv'), 'r',
                  encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                language_code = row['Language_ID'].strip()
                concept_code = row['Parameter_ID'].strip()
                form = row['Form'].strip()
                iso = language_code_iso.get(language_code)
                if not iso: continue
                concepticon = concept_code_concepticon.get(concept_code)
                if not concepticon: continue
                iso_form_concepts[iso][form].add(concept_code)
                iso_form_concepticon[iso][form].add(concepticon)

        self._iso_form_concepts = {
                iso: {
                    form: sorted(concepts)
                    for form, concepts in form_concepts.items()}
                for iso, form_concepts in iso_form_concepts.items()}

        self._iso_form_concepticon = {
                iso: {
                    form: sorted(concepts)
                    for form, concepts in form_concepts.items()}
                for iso, form_concepts in iso_form_concepticon.items()}

        self.isos = sorted(iso_form_concepts.keys())

        # TODO: extend lexicon by removing dashes (e.g. Swedish compounds)
        #       but is this common?
        #       except final dashes, e.g. "hun-"
        #       remove question marks as well


    def __getitem__(self, iso):
        return self._iso_form_concepts[iso]


if __name__ == '__main__':
    import sys
    ids = IDS(sys.argv[1])
    #print(ids['swe']['vi'])
    from pprint import pprint
    pprint(ids['dan'])

