#!/usr/bin/env bash

## To train the word and char-based LM with the default parameters.

model_type=$1
if [[ $model_type == "word" ]]; then
  echo "Preparing the input for the word-based LM model."
  echo "Please note that this process will take approximately 1.5 hours and the resulting training data will be approximately ~260 GBs."
  python train-lm/data_processors/prepare_word_lm_data.py
  python train-lm/word_level/train_word_lm.py
elif [[ $model_type == "char" ]]; then
  python train-lm/char_level/train_char_lm.py
else
      echo "Invalid LM type. Avaliable options are 'char' and 'word'"
fi

