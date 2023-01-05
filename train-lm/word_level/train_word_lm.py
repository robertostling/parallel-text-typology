import argparse
import os

import numpy as np
from tqdm import tqdm
from word_lm import word_lm as language_model
from word_lm_dataset import word_lm_dataset  as data_reader
from mulres import config


def save_embeddings(model, dataset, output_dir, hidden_size, epoch):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    weights = np.array(model.layers[2].get_weights())
    weights = np.squeeze(weights)
    language_embeddings = {v: weights[k] for k, v in dataset.id_trans_map.items() if "<pad_lang>" != v}
    keys = sorted(list(language_embeddings.keys()))
    model.save_weights(os.path.join(output_dir, "word_level_model_ep_{}_lstm_{}.h5".format(hidden_size, epoch)))

    with open(os.path.join(output_dir, "word_level_language_embedding_lstm_{}_ep_{}.vec".format(hidden_size, epoch)), "w", encoding="utf8") as f:
        f.write("{}\t{}\n".format(len(language_embeddings), list(language_embeddings.values())[0].shape[0]))
        for k in keys:
            s = np.array2string(language_embeddings[k]).replace("\n", "")
            s = ' '.join(s[1:-1].split())
            f.write(k + " " + s + "\n")
    f.close()
    print("Epoch {} finished! Embeddings are saved!".format(epoch))


def train(args):
    binary_file_dir = args.binary_dir
    look_up_dir = args.lookup_dir
    output_dir = args.output

    hidden_size = args.lstm_size
    batch_size = args.batch
    word_vector_dim = args.we_size
    lang_vector_dim = args.lv_size
    no_of_epoch = args.epoch
    limit = args.limit
    dataset = data_reader(look_up_dir, binary_file_dir, batch_size=batch_size, limit=limit)
    number_of_lang = dataset.number_of_languages
    number_of_sentence = dataset.total_number_of_sentences

    lang_model = language_model(number_of_lang, word_vector_dim, lang_vector_dim, hidden_size)
    ## TRAINING
    steps_per_epoch = int(number_of_sentence / batch_size) + 1
    for ep in range(no_of_epoch):
        loss = []
        data_generator = dataset.batch_generator()
        epoch_iterator = tqdm(data_generator, total=steps_per_epoch, desc="Iteration")

        for i, (x, y) in enumerate(epoch_iterator):
            l = lang_model.model.train_on_batch(x, y)
            loss.append(l)
            epoch_iterator.set_description("loss:" + str(round(np.average(loss), 4)))
        print()
        save_embeddings(lang_model.model, dataset, output_dir, hidden_size, ep)
    print("done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Word-level LM to learn language embedddings')
    parser.add_argument('--binary_dir', '-bin', type=str, default=config.word_lm_input_path_binary)
    parser.add_argument('--lookup_dir', '-lookup', type=str, default=config.word_lm_input_path_lookup)
    parser.add_argument('--output', '-o', type=str, default=config.language_embeddings_path)

    parser.add_argument('--lv_size', type=int, default=100)
    parser.add_argument('--we_size', type=int, default=300)
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--lstm_size', type=int, default=512)
    parser.add_argument('--limit', type=int, default=2000, help="Limit the number of languages (used in testing)")

    args = parser.parse_args()
    train(args)
