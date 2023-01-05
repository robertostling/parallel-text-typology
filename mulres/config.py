import os.path

# Data files included in the repository
contact_path = os.path.join('resources', 'neighbour_graph.list')

# Base path for all data
data_path = 'data'

# Download with preprocessing/download-ids.sh
ids_path = os.path.join(data_path, 'resources', 'ids')

# Download with preprocessing/download-asjp.sh
asjp_path = os.path.join(data_path, 'resources', 'asjp', 'dataset.tab')

# Texts imported from the paralleltext repository
corpus_path = os.path.join(data_path, 'resources', 'parallel-text')

# Texts to be fed to the Turku neural parser pipeline
toparse_path = os.path.join(data_path, 'resources', 'parallel-text-toparse')

# Metadata for each translation (see README.md)
json_path = os.path.join(data_path, 'resources.json')

# List of IDs of verses that occur in most translations in the Bible corpus
verses_path = os.path.join(data_path, 'resources', 'common-verses.txt')

# Directory containing UD-annotated texts
# If changed, this must be synchronized with
#   preprocessing/generate_parsing_script.py (which uses absolute paths)
ud_path = os.path.join(data_path, 'processed', 'ud')

# Path where binary encoded texts are put
encoded_path = os.path.join(data_path, 'processed', 'encoded')

# Path where type-level alignments are saved
aligned_path = os.path.join(data_path, 'processed', 'aligned')

# Path where projected word alignments are saved
embeddings_path = os.path.join(data_path, 'processed', 'embeddings')

# Path where ASJP-derived embeddings are saved
asjp_embeddings_path = os.path.join(data_path, 'processed', 'asjp')

# Path where embeddings from alignment-based lexical similarity are saved
lexical_path = os.path.join(data_path, 'processed', 'lexical')

# Path where projected word order statistics are saved
wordorder_path = os.path.join(data_path, 'processed', 'wordorder')

# Path where transliterated texts are saved.
transliterated_path =  os.path.join(
        data_path, 'processed', 'transliterated')

# Path where guessed paradigms are saved.
paradigms_path = os.path.join(data_path, 'processed', 'paradigms')

# Guesses for inferred affixation data
affixation_path = os.path.join(data_path, 'processed', 'affixation.json')


# Path to pretrained Turku Neural Parser Pipeline models
turku_model_path = '/hd1/turkunlp/models'

# Path to aligned Smith et al. (2017) word embeddings (or in general any other
# aligned vector space, with the %s being replaced by Wikipedia language
# codes, i.e. typically ISO 639-1)
smith_path_pattern = ('/hd1/multilingual_resources/multilingual_embeddings/'
                      'wiki.%s.original.aligned.vec')

# Path to cache of Smith et al. embeddings, with only Bible vocabulary kept
embeddings_cache_path = os.path.join(
        data_path, 'processed', 'cached-embeddings')

# Path to cache of evaluation results
evaluations_cache_path = os.path.join(
        data_path, 'processed', 'cached-evaluations')

typology_db_path = os.path.join(
        data_path, 'processed', 'typology_db.pickle')

## MURATHAN
# Path where the training data for LMs are saved.
lm_train_path = os.path.join(data_path, 'lm_input')
language_embeddings_path = os.path.join(data_path, "output", 'language_embeddings')

paradigms_input_path = os.path.join(lm_train_path, 'paradigms_input') # paradigm models
translation_input_path = os.path.join(lm_train_path, 'translation_input') # nmt models

word_lm_input_path = os.path.join(lm_train_path, 'word_lm_input')  # word-lm models
word_lm_input_path_binary = os.path.join(word_lm_input_path, "np")
word_lm_input_path_lookup = os.path.join(word_lm_input_path, "lookup")

bible_list_file = os.path.join(data_path, 'bible_list.txt')
