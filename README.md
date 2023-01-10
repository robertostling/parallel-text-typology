# parallel-text-typology

Code for the paper *Neural models can sometimes discover typological
generalizations*, used for deriving resources from massively multilingual
parallel texts.

See below under Evaluation for instructions on how to check
which typological features are encoded by language representations. The first
(and main) part of this document describes how to generate the various
resources needed to reproduce our experiments, but this is a time-consuming
process and we have published the generated data that we think may be of
interest:

* Östling, Robert & Kurfalı, Murathan. (2023). Parallel text typology dataset (1.0.0) [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.7506220](https://doi.org/10.5281/zenodo.7506220)
* Östling, Robert & Kurfalı, Murathan. (2018). Projected word embeddings
  dataset (1.0.0) [Data set]. [Copy at Stockholm University](http://dali.ling.su.se/projected-word-embeddings))

# Components

## Dependencies and preparations

First, install the package `mulres` in develop/editable mode:

    pip3 install --user --editable .

A number of programs should be installed in the parent directory of
`multilingual-resources`.

### external dependencies

The experiments in our paper use the following versions, but more recent
versions seem to work as well:

    pip3 install --user python-numba==0.48.0
    pip3 install --user python-Levenshtein==0.12.0
    pip3 install --user umap-learn==0.3.10
    pip3 install --user scikit-learn=1.0.1

### langinfo

Library to access Glottolog data. Note that the git repository contains a
cached Glottolog database, which may or may not be up to date. For
reproducibility, use the cached version (i.e. do not run `download.sh` and
`cache.py`).

    git clone http://github.com/robertostling/langinfo.git
    cd langinfo
    cd langinfo/data
    ./download.sh
    cd ../..
    python3 cache.py
    python3 setup.py install --user

## Import a parallel text collection

* `preprocessing/import_parallel_text.py`
* `preprocessing/find_common_verses.py`
* `data/resources/parallel-text` (output directory)
* `data/resources/parallel-text-toparse` (output directory)
* `data/resources.json` (output file)
* `data/resources/verses.txt` (output file)

Commands (ca 12 minutes):

    python3 preprocessing/import_parallel_text.py ../paralleltext

Given a source directory with multi-parallel texts in Cysouw's format, this
should create a table (as a list containing dictionaries, in a json file)
containing a list of parallel texts and the resources available for them. It
should also copy (or symlink) the files into `data/resource/parallel-text`,
renaming them to ensure that the first 3 letters always contains the ISO 639-3
code.

Files to be processed by the Turku neural parser pipeline will be written to
`data/resources/parallel-text-toparse`. Metadata is encoded so the pipeline
will pass it along.

The above two directories are mutually exclusive, to avoid possible confusion
downstream. The (re-)tokenization provided by the Turku pipeline is the
reference tokenization for those texts, while the original tokenization is
kept for the remaining texts (in `data/resources/parallel-text`).

The fields defined in `data/resources.json` dictionaries are:

* `name`: Name of text (e.g. `zul-x-bible`). This refers to the files in
  `data/resource/parallel-text`.
* `glottocode`: Glottocode of language.
* `iso`: ISO 639-9 of language.
* `script`: ISO-15924 code of script.
* `verses`: Integer number of verses in text.
* `nt_verses`: Integer number of verses in New Testament part.
* `ot_verses`: Integer number of verses in Old Testament part.
* `tokens`: Integer number of tokens in text.
* `types`: Integer number of types (case-normalized) in text.
* `ud`: (optional) Name of UD treebank to use for pre-processing
  (lemmatization, PoS tagging, parsing). Example `ar_padt`.
* `annotated` (optional): If this text is a manually annotated CoNLL-U file,
  specify `yes` here.
* `ids` (optional): IDS language identifier, to be used for adding IDS
  concepts (assuming also that a lemmatizer model is available).
* `smith` (optional): if there are Smith et al. (2017) aligned word embeddings
  available for this language/script, specify the language code (ISO 639-1)
  used in their repository.

Parallel texts in languages/orthographies without proper word segmentation
should not be included at all.

Then, create the file `data/resources/verses.txt` (ca 1 minute):

    python3 preprocessing/find_common_verses.py


## Add UD annotations (tokenization, tagging, parsing)

* `preprocessing/generate_parsing_script.py` 
* `data/resources/parallel-text-toparse` (input directory)
* `data/resources.json` (input file)
* `processed/ud` (output directory)

Run old Turku neural parser pipeline with models from UD.

All texts in `parallel-text-toparse` with `ud` given in `data/resources.json`
should be processed with the model specified there, and written as CoNLL-U
files into `processed/ud`.

The script `preprocessing/generate_parsing_script.py` generates a shellscript,
which can be copied to the working directory of the Turku neural parser
pipeline and run from there (within whatever Python environment it needs).

    python3 preprocessing/generate_parsing_script.py >/path/to/turkunlp/run.sh
    cd /path/to/turkunlp/
    [set up virtualenv, docker or other magic]
    bash run.sh

This is fast enough to run overnight, even without a GPU.

## Encode texts

* `preprocessing/encode_texts.py`
* `data/resources.json` (input file)
* `data/resources/parallel-text` (input directory)
* `processed/ud` (input directory)
* `processed/encoded` (output directory)

In order to speed up later steps, encode the available data into an efficient
format of mostly NumPy arrays (pickled). This takes about 7 minutes on a
16-core system (90 CPU minutes):

    python3 preprocessing/encode_texts.py

These files are loaded with the EncodedMPF class of `mulres/empfile.py`.


## Type-level alignments

* `processing/multialign.py`
* `data/resources.json` (input file)
* `processed/encoded` (input directory)
* `processed/aligned` (output directory)

Align all texts with UD models and preferred_source = "yes", with all texts
with preferred_source = "no". The result is saved in `processed/aligned` as
pickled lists of lemma-to-ngram dictionaries.

    python3 processing/multialign.py

This takes about 12 hours on a 16-core system (aligning 1671 targets with 35
sources).

## Transliterating texts

Transliterate all texts and remove some distinctions (ca 3 minutes on a
16-core system):

    python3 processing/transliterate_texts.py

This is used by the character-based language model, in order to remove
as many irrelevant orthographic distinctions as possible.

## TODO: downloading and aligning word embeddings

@murathan: add information on where scripts for this are located in the repo.
The scripts could download the data, or (if small enough, e.g. MUSE
dictionaries) add directly to this repo.

## Projecting word embeddings

Edit `mulres/config.py` to provide a path to the multilingual embeddings.
Then run (ca 30 minutes on a 16-core system):

    python3 preprocessing/cache_embeddings.py

This cuts out the relevant parts of the embeddings (i.e. Bible vocabulary) and
saves them in `data/processed/cached-embeddings`.

Then run (ca 300 minutes on a 16-core system):

    python3 processing/project_embeddings.py

Multilingual word embeddings will be projected to all Bible texts, and placed
in `data/processed/embeddings`.

## Projecting word order statistics

Project word order statistics (ca 330 minutes on a 16-core system):

    python3 processing/project_wordorder.py

The wordorder statistics for each file will be put in
`data/processed/wordorder/`. An evaluation with respect to URIEL data can be
performed using the `evaluate_projection.py` script:

    python3 evaluation/evaluate_pojection.py


## Generate paradigms

Guess nominal and verbal paradigms for each text (ca 300 minutes on a 16-core
system):

    python3 processing/guess_paradigms.py

The outputs are *not* transliterated or normalized, this is handled by the
paradigm prediction model.

The paradigms are also used for estimating affixation type (ratio of suffixing
to prefixing, or no affixation). This is done by the following script, and
takes less than a minute:

    python3 processing/guess_affixation.py

The results are saved in `data/processed/affixation.json`.

## Creating language representations from ASJP

To create baseline language representations using ASJP lexical similarity,
run the following (ca 20 minutes on a 16-core system):

    preprocessing/download-asjp.sh
    python3 processing/create_asjp_vectors.py

This first creates `data/processed/asjp/full.pickle` which contains a full
pairwise language similarity matrix, using mean normalized Levenshtein
distance from the 40-item ASJP vocabulary list (slow, ca 5 core hours).  Then,
it uses UMAP to reduce this matrix into 100-dimensional language
representations, which are saved in `data/processed/asjp/asjp.vec` (fast, ca
15 seconds). The `full.pickle` file is cached, in case the second step needs
to be run multiple times.

## Creating lexical language representations

    python3 processing/create_lexical_vectors.py

This creates `data/processed/lexical/lexical_umap.vec` and
`data/processed/lexical/lexical_svd.vec`, containing representations derived
as described above for ASJP vectors, but using word lists derived from aligned
Bible texts rather than ASJP. Expected output and resource use for a full run
below, but note that intermediate steps (`forms.csv` and `matrix.pickle`) are
cached.

    $ time python3 processing/create_lexical_vectors.py
    Generating data/processed/lexical/forms.csv
    1664 texts to compute lexical similarity between
    Generating data/processed/lexical/matrix.pickle
    1383616 pairs of languages
    86476 pairs per core, computing pairwise distances...
    Reducing 1664x1664 distance matrix with UMAP
    Reducing 1664x1664 distance matrix with SVD

    real    380m23.328s
    user    4268m34.988s
    sys     2m41.576s


# Evaluation

To perform these evaluations you need to have unpacked our data repository.
Then, ensure that the `data` subdirectory links to the `evaluation-data`
subdirectory of the data repository:

    ln -s /path/to/data/repository/evaluation-data data

After this, you sholud be able to reproduce the following.

## Evaluate language embeddings

The main results in the paper were generated with the following command:

    time python3 evaluation/evaluate_language_vectors.py \
        language-vectors/iso-bibles.txt \
        log-balanced-l21e-3.tsv \
        language-vectors/word_level.vec \
        language-vectors/reinflect_verb.vec \
        language-vectors/ostling2017_all.lang \
        language-vectors/malaviya_mtvec.lang \
        language-vectors/malaviya_mtcell.lang \
        language-vectors/asjp_svd.lang \
        language-vectors/character_level.vec \
        language-vectors/lexical_svd.vec \
        language-vectors/nmt_from_eng.vec \
        language-vectors/nmt_to_eng.vec \
        language-vectors/reinflect_all.vec \
        language-vectors/reinflect_noun.vec \
        language-vectors/word_encoding.vec

Using the following parameters:

    balance = True
    cross_validate = False
    fixed_regularization = 1e-3
    naive_loocv = False

The file `log-balanced-l21e-3.tsv` is very large, but can be summarized into a
number of more readable statistics using the scripts `analyze_predictions.py`
and `analyze_simple.py`:

    python3 evaluation/analyze_simple.py log-balanced-l21e-3.tsv

The above command creates `log-balanced-l21e-3.tsv.predictions` and
`log-balanced-l21e-3.tsv.summary`, which can be used to more quickly summarize
the results and plot figures:

    mkdir -p data/figures/barplots
    python3 evaluation/analyze_predictions.py log-balanced-l21e-3.tsv

Figures will be saved as PDF files in `data/figures`. Currently the plotting
code is hard-coded to include the language representations investigated in our
paper and will crash if some of those representations are missing, but you can
comment out the call to `plot_figures` at the end of the script to analyze
other sets of language representations.

This script outputs a large amount of information. Search for lines starting
with two dashes `--` to find the start of an evaluation. For instance, in our
evaluation, the first line should be:

    --word_level S_RELATIVE_AFTER_NOUN uriel 64.4% (n = 545) F1 = 0.641

This indicates that the mean family-weighted F-score is a rather low 0.641,
when using the representations in `word_level.vec` to evaluate the
`S_RELATIVE_AFTER_NOUN` feature using URIEL training data (`proj` instead of
`uriel` indicates that the classifier was trained on projected data).

Following this is a family-weighted confusion matrix, then:

    PROJECTION BASELINE (n = 545): F1 = 0.851  Acc = 90.4%

This means that this parameter can be estimated with a F1 of 0.851 (with
respect to URIEL) using annotation projection. Since this evaluation does not
necessarily use exactly the same set of languages as the full evaluation
above, we also show a directly comparable number using the same data subset:

    COMPARABLE F1 = 0.679  Acc = 69.2

Since 0.679 is much lower than 0.851, this indicates that the classifier has a
limited ability (probably only due to correlations with other features) to
predict this feature, and thit it is unlikely to be encoded clearly in the
`word_level` embeddings.

Following this is a 3-way confusion matrix (see paper for discussion), and
lists of Bible translations for each element of this matrix. This can be used
for a manual analysis. A quick typological classification using URIEL data is
printed for each language, along with its family according to Glottolog.

