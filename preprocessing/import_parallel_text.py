import sys
import glob
import os
import json

from langinfo.glottolog import Glottolog

from mulres.mpfile import MPFile
from mulres.ids import IDS
from mulres.config import (corpus_path, toparse_path, json_path, ids_path,
                           data_path)

preferred_texts = set('''
spa-x-bible-newworld
nob-x-bible-newworld
dan-x-bible-newworld
ell-x-bible-newworld
eus-x-bible-batua
zsm-x-bible-goodnews
nld-x-bible-newworld
pol-x-bible-newworld
eng-x-bible-newworld2013
ind-x-bible-newworld
deu-x-bible-newworld
nno-x-bible-1978
rus-x-bible-newworld
fin-x-bible-newworld
slk-x-bible-newworld
ita-x-bible-newworld
fra-x-bible-newworld
por-x-bible-newworld1996
swe-x-bible-newworld
epo-x-bible
ekk-x-bible-1997
bel-x-bible-bokun
mkd-x-bible-2004
ukr-x-bible-2009
lit-x-bible-tikejimozodis
hrv-x-bible-newworld
ces-x-bible-newworld
slv-x-bible-newworld
tur-x-bible-newworld
sme-x-bible
afr-x-bible-newworld
cat-x-bible-inter
hin-x-bible-bsi
ron-x-bible-newworld
lav-x-bible
kor-x-bible-newworld2014
vie-x-bible-newworld
kat-x-bible-revised
hye-x-bible-newworld
hun-x-bible-newworld
bul-x-bible-newworld
gle-x-bible
sqi-x-bible-interconfessional
'''.split())

# uig-x-bible-arabic    poor segmentation (doesn't even separate punctuation)
# arb-x-bible-newworld  poor tagging? X tags everywhere
# pes-x-bible-newworld  special characters not separated
# heb-x-bible-newworld  seems messed up
# srp-x-bible           poor tagging? ADP tags everywhere
# (minor?) sentence segmentation issues, but probably OK: por, slk, nld


# Map of ISO 639-3 codes to Glottocodes where these are not available in
# Glottolog.
iso_glottocode_map = {
        # Enxet (Southern Lengua) -- missing ISO code in Glottolog,
        # alternative would be to use the common ISO code for both
        # Lengua varieties (leg)
        'enx': 'sout2989'
}

# Languages with at least 40% p@1 in the evaluations of Smith et al.
# Used for selecting translations with access to decent multilingual word
# embeddings.
smith_map = {
        ('eng', 'latn'): 'en',
        ('fra', 'latn'): 'fr',
        ('por', 'latn'): 'pt',
        ('spa', 'latn'): 'es',
        ('ita', 'latn'): 'it',
        ('nld', 'latn'): 'nl',
        ('nob', 'latn'): 'no',
        ('dan', 'latn'): 'da',
        ('cat', 'latn'): 'ca',
        ('swe', 'latn'): 'sv',
        ('ces', 'latn'): 'cs',
        ('ron', 'latn'): 'ro',
        ('deu', 'latn'): 'de',
        ('pol', 'latn'): 'pl',
        ('hun', 'latn'): 'hu',
        ('fin', 'latn'): 'fi',
        ('epo', 'latn'): 'eo',
        ('rus', 'cyrl'): 'ru',
        ('glg', 'latn'): 'gl',
        ('mkd', 'cyrl'): 'mk',
        ('ind', 'latn'): 'id',
        ('bul', 'cyrl'): 'bg',
        ('zsm', 'latn'): 'ms',
        ('ukr', 'cyrl'): 'uk',
        ('hrv', 'latn'): 'hr',
        ('tur', 'latn'): 'tr',
        ('slv', 'latn'): 'sl',
        ('ell', 'grek'): 'el',
        ('slk', 'latn'): 'sk',
        ('ekk', 'latn'): 'et',
        ('srp', 'cyrl'): 'sr',
        ('afr', 'latn'): 'af',
        ('lit', 'latn'): 'lt',
        ('arb', 'arab'): 'ar',
        ('bos', 'latn'): 'bs',
        ('lav', 'latn'): 'lv',
        ('eus', 'latn'): 'eu',
        ('pes', 'arab'): 'fa',
        ('hye', 'armn'): 'hy',
        ('sqi', 'latn'): 'sq',
        ('bel', 'cyrl'): 'be',
        ('kat', 'geor'): 'ka',
        ('kor', 'hang'): 'ko',
        ('vie', 'latn'): 'vi',
        ('hin', 'deva'): 'hi',
}

# Blacklist of languages to exclude because of known issues (usually
# poor or lacking segmentation/tokenization)
#
# TurkuNLP tool has poor performance on Chinese and Japanese, better avoid
# Also avoid Classical Chinese due to segmentation issues.
#
# lif-x-bible-2009 is written with Limbu script, which unidecode does not
# support. For the Limbu language we still have lif-x-bible-devanagari.
#
iso_blacklist = set('cmn jpn lzh'.split())
text_blacklist = set(('khm-x-bible-newworld khm-x-bible-standard2005 '
                      'ksw-x-bible mya-x-bible-common lif-x-bible-2009'
                      ).split())

iso_map = {
        'ara': 'arb', # Unknown status of Arabic translations, using Modern
                      # Standard Arabic (arb) for now
        'zho': 'cmn', # Least bad match for standard written Chinese
        'oji': 'ciw',
        'ebk': 'bkb', # Glottolog does not follow the split from ISO 639-3:
                      # bkb -> ebk (Eastern) + obk (Southern) Bontok
        'msa': 'zsm', # Standard Malay
        'est': 'ekk', # Standard Estonian
        'iku': 'ike', # Specific code for Eastern Canadian Inuktitut
        'aze': 'azb', # Specific code for South Azerbaijani
        # hva (Huastec San Luis Potosí) uses hus (general Huastec code) in
        # metadata, which should be OK
        'aym': 'ayr', # Metadata contains general Aymara code, filename
                      # uses Central Aymara (ayr) which is preferable assuming
                      # that is correct
        # acr (Achi' de Cubulco) uses acc (general Achi code) in metadata
        # tzo-x-bible-chamula.txt uses tzc in metadata (specific code for
        # Chamula dialect)
        'nwx': 'new', # Newar, unclear which variety. nxw and new both used,
                      # with Glottolog preferring the latter
}

# Map from (ISO 639-3, ISO 15924) code to preferred UD Treebank
ud_models = {
        ('afr', 'latn'): 'af_afribooms',
        ('arb', 'arab'): 'ar_padt',
        # br_keb: low LAS
        # bxr_bdt: low LAS
        #('bel', 'cyrl'): '',
        ('bul', 'cyrl'): 'bg_btb',
        ('cat', 'latn'): 'ca_ancora',
        #('cop', 'copt'): 'cop',
        ('ces', 'latn'): 'cs_pdt',
        ('chu', 'cyrl'): 'cu_proiel',
        ('dan', 'latn'): 'da_ddt',
        ('deu', 'latn'): 'de_gsd',
        ('ell', 'grek'): 'el_gdt',
        ('eng', 'latn'): 'en_ewt',
        ('spa', 'latn'): 'es_ancora',
        ('ekk', 'latn'): 'et_edt',
        ('eus', 'latn'): 'eu_bdt',
        ('pes', 'arab'): 'fa_seraji',
        ('fin', 'latn'): 'fi_tdt',
        # fo_oft: low LAS
        ('fra', 'latn'): 'fr_gsd',
        # fro_srcmf: any suitable bibles for Old French (ca 800-1400)?
        ('gle', 'latn'): 'ga_idt',
        ('glg', 'latn'): 'gl_ctg',
        # Don't need PROIEL versions
        #('got', 'latn'): 'got_proiel',
        #('grc', 'grek'): 'grc_proiel',
        ('heb', 'hebr'): 'he_htb',
        ('hin', 'deva'): 'hi_hdtb',
        ('hrv', 'latn'): 'hr_set',
        # hsb_ufal: low LAS
        ('hun', 'latn'): 'hu_szeged',
        ('ind', 'latn'): 'id_gsd',
        ('ita', 'latn'): 'it_isdt',
        ('jpn', 'jpan'): 'ja_gsd',
        # kk_ktb: low LAS
        # kmr_mg: low LAS
        ('kor', 'hang'): 'ko_kaist',
        # Don't need PROIEL versions
        #('lat', 'latn'): 'la_proiel',
        #('lit', 'latn'): 'lt',
        ('lav', 'latn'): 'lv_lvtb',
        ('nld', 'latn'): 'nl_alpino',
        ('nob', 'latn'): 'no_bokmaal',
        ('nno', 'latn'): 'no_nynorsk',
        ('pol', 'latn'): 'pl_lfg',
        ('por', 'latn'): 'pt_bosque',
        ('ron', 'latn'): 'ro_rrt',
        ('rus', 'cyrl'): 'ru_syntagrus',
        #('san', 'deva'): 'sa',
        ('slk', 'latn'): 'sk_snk',
        ('slv', 'latn'): 'sl_ssj',
        ('sme', 'latn'): 'sme_giella',
        ('swe', 'latn'): 'sv_talbanken',
        ('srp', 'cyrl'): 'sr_set',
        # th_pud: too small, no models work
        #('tam', 'taml'): 'ta',
        ('tur', 'latn'): 'tr_imst',
        ('ukr', 'cyrl'): 'uk_iu',
        ('uig', 'arab'): 'ur_udtb',
        ('vie', 'latn'): 'vi_vtb',
        ('cmn', 'hant'): 'zh_gsd',
        }

def main():
    paralleltext_path = sys.argv[1]

    ids = IDS(ids_path)

    os.makedirs(corpus_path, exist_ok=True)
    os.makedirs(toparse_path, exist_ok=True)

    text_table = {}

    raw_files = glob.glob(os.path.join(
        paralleltext_path, 'bibles', 'corpus', '*x-bible*.txt'))

    proiel_files = [
            os.path.join(paralleltext_path, 'bibles', 'tagged',
                iso+'-x-bible-proiel.conllu')
            for iso in ('grc', 'lat')]

    for filename in raw_files:
        mpf = MPFile(filename, only_metadata=False)
        file_iso = mpf.name[:3]
        meta_iso = mpf.metadata['closest_iso_639-3']
        iso = iso_map.get(meta_iso, meta_iso)

        if (iso in iso_blacklist) or (mpf.name in text_blacklist):
            print('BLACKLISTED', iso, mpf.name)
            continue

        try:
            languoid = Glottolog[iso]
            glottocode = languoid.id
        except KeyError:
            languoid = None
            glottocode = iso_glottocode_map[iso]
            languoid = Glottolog[glottocode]

        if file_iso != iso or meta_iso != iso:
            print(os.path.basename(filename), '-->', iso, glottocode)

        try:
            script = mpf.metadata['iso_15924']
        except KeyError:
            print('***', os.path.basename(filename), 'lacking iso_15924')
            script = '???'

        mpf.name = iso + mpf.name[3:]

        tokens = [token.lower() for sentence in mpf.sentences.values()
                                for token in sentence]
        types = set(tokens)

        script = mpf.metadata['iso_15924']
        ud_model = ud_models.get((iso, script.lower()))
        smith_model = smith_map.get((iso, script.lower()))

        n_ot = sum(verse[0] in '0123' for verse in mpf.sentences.keys())
        n_nt = sum(verse[0] in '456' for verse in mpf.sentences.keys())

        preferred_source = 'yes' if mpf.name in preferred_texts else 'no'

        info = dict(
                name=mpf.name,
                iso=iso,
                glottocode=languoid.id,
                script=script,
                verses=len(mpf.sentences),
                tokens=len(tokens),
                types=len(types),
                nt_verses=n_nt,
                ot_verses=n_ot,
                preferred_source=preferred_source
                )

        if ud_model: info['ud'] = ud_model
        if smith_model: info['smith'] = smith_model

        # If we have a lemmatizer *and* IDS lexicon:
        if (iso in ids.isos) and ud_model:
            lexicon = ids[iso]
            n_types_in_ids = sum(s in lexicon for s in types)
            print(iso, ':', n_types_in_ids, 'of', len(types), 'in IDS')
            # Quick filtering, need lemmatized data for proper result
            if n_types_in_ids > 100:
                info['ids'] = iso

        text_table[info['name']] = info

        if ud_model and preferred_source == 'yes':
            # Write plain text file for processing with the Turku neural
            # parser pipeline
            mpf.write(toparse_path, file_format='turku')
            #mpf.write(toparse_path, file_format='txt.gz', write_metdata=False)
        else:
            # Write plain text file in paralleltext format
            # Do not write metadata, since this may contradict what is in
            # resources.json
            mpf.write(corpus_path, file_format='txt.gz',
                      write_metadata=False)


    with open(json_path, 'w') as f:
        json.dump(text_table, f, sort_keys=True, indent=4)


    # The (old) code below assumes a list of dicts
    text_table = list(text_table.values())

    ud_languages = {info['iso'] for info in text_table if 'ud' in info}
    ud_names = {info['ud'] for info in text_table if 'ud' in info}
    smith_languages = {info['iso'] for info in text_table if 'smith' in info}
    ids_languages = {info['iso'] for info in text_table if 'ids' in info}

    print(len(text_table), 'texts')
    print(len({info['glottocode'] for info in text_table}), 'languages')
    print(len({info['script'] for info in text_table}), 'different scripts')
    print(sum(int('ud' in info) for info in text_table),
            'translations with UD models from', len(ud_languages),
            'different languages')
    print(sum(int('smith' in info) for info in text_table), 'translations in',
            len(smith_languages), 'languages with Smith et al. embeddings')
    print(sum(int('ids' in info) for info in text_table), 'translations in',
            len(ids_languages), 'languages with (possible) IDS lexicon')

    #print(' '.join(sorted(ud_languages)))
    #print(' '.join(sorted(ud_names)))

if __name__ == '__main__': main()

