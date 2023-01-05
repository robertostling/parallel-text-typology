#!/usr/bin/env bash

model_type=$1 ## add check
if [[ $model_type == "paradigm" ]]; then
    paradigm_pos=$2 ## add check
else
    direction=$2 ## add check
fi

lstm_size=$3
word_vec_size=$4
save_checkpoint_steps=50000
train_steps=100000

if [[ "$#" -ne 4 ]]; then
word_vec_size=100
fi
if [[ "$#" -ne 3 ]]; then
lstm_size=1024
fi
if [[ "$#" -ne 2 ]]; then
    echo -e "Less than 2 arguments."
    echo -e "You need to specify which model you want to train. Available options are: translation eng_to_others ; translation others_to_eng ; paradigm noun ; paradigm verb ; paradigm mixed"
exit 2
fi
echo "Input preparation...."
if [[ $model_type == "paradigm" ]]; then
paradigms_input_path=$(python - <<EOF
from mulres import config
print(config.paradigms_input_path)
EOF
)
    data_dir="${paradigms_input_path}_${paradigm_pos}"
    model_name="${paradigm_pos}_paradigm"
    python train-lm/data_processors/prepare_paradigm_data.py
else
translation_input_path=$(python - <<EOF
from mulres import config
print(config.translation_input_path)
EOF
)
    python train-lm/data_processors/prepare_nmt_data.py
    data_dir="${translation_input_path}/${direction}"
    model_name="${direction}_translation"
fi

output_path==$(python - <<EOF
from mulres import config
print(config.language_embeddings_path)
EOF
)

echo "ONMT preprocess..."
onmt_preprocess -train_src "${data_dir}/src-train.txt" -train_tgt "${data_dir}/tgt-train.txt" \
            -valid_src "${data_dir}/src-val.txt" -valid_tgt "${data_dir}/tgt-val.txt" -save_data "${data_dir}/${model_name}" -overwrite
sleep 2
echo "ONMT train..."
onmt_train -data "${data_dir}/${model_name}" -save_model "${data_dir}/${model_name}-model" -world_size 1 -gpu_ranks 0 --word_vec_size $word_vec_size \
                --rnn_size $lstm_size  -save_checkpoint_steps $save_checkpoint_steps -train_steps $train_steps
echo "Saving the language embeddings..."
python data_processors/extract_lm_onmt.py --model "${data_dir}/${model_name}-model_step_${train_steps}.pt" --output "${output_path}/${model_name}"
echo "done."
