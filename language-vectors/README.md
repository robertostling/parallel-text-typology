# Collection of language vectors from different sources

The naming convention here is that `.lang` indicates that vector keys are
3-letter ISO 639-3 codes, while `.vec` files may contain multiple vectors per
language, e.g. `swe-x-bible-1917`.

## Baselines

* `malaviya_mtcell.lang` Malavia et al. (2017), NMT state derived vectors
* `malaviya_mtvec.lang` Malavia et al. (2017), NMT language embeddings
* `malaviya_mtboth.lang` Malavia et al. (2017), combination of both above
* `ostling2017_l1.lang` Östling and Tiedemann (2017), layer 1
* `ostling2017_l2.lang` Östling and Tiedemann (2017), layer 2
* `ostling2017_l3.lang` Östling and Tiedemann (2017), layer 3
* `ostling2017_all.lang` Östling and Tiedemann (2017), concatenation of above

The Östling and Tiedemann (2017) vectors are from the original files `1024state-64lang-72h*.lang`

The Malavia et al. (2017) vectors are extracted from the lang2vec Python
library.

## Our new models

* `word_level.vec` our word-level LSTM LM (`LATEST_word_level_language_embedding_lstm_512_ep_0.vec`)
* `nmt_to_eng.vec` our OpenNMT everything-to-English model (`malaviya_style_language_embeddings.vec`), should be equivalent to Malavia et al.
* `nmt_from_eng.vec` our OpenNMT English-to-everything model (`eng_to_others_nmt_language_embeddings.vec`)
* `character_level.vec` our character-level LSMT LM (`char_level_language_embedding_lstm_512_ep_0.vec`)
* `reinflect_noun.vec` our reinflection model, nouns only (`morph_level_NOUN_language_embeddings.vec`)
* `reinflect_verb.vec` our reinflection model, verbs only (`morph_level_VERB_language_embeddings.vec`)
* `reinflect_all.vec` our reinflection model, nouns+verbs (`morph_level_mixed_language_embeddings.vec`)
* `word_encoding.vec` our model for predicting projected word embeddings from character sequences (`embedding_prediction_language_embedding_lstm_512_ep_2.vec`)

## Our lexical baselines

* `asjp_umap.lang` uses UMAP on a mean normalized Levenshtein distance pairwise distance matrix from ASJP
* `asjp_svd.vec` uses truncated SVD on a mean normalized Levenshtein distance pairwise distance matrix from word alignments
* `lexical_umap.vec` uses UMAP on a mean normalized Levenshtein distance pairwise distance matrix from word alignments
* `lexical_svd.vec` uses truncated SVD on a mean normalized Levenshtein distance pairwise distance matrix from word alignments

