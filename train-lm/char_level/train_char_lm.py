import argparse
import os
import numpy as np
from char_lm import char_language_model
from char_lm_dataset import char_lm_dataset, char_lm_dataset_word
from tqdm import tqdm
from mulres import config


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def save_embeddings(model, dataset, output_dir, hidden_size, epoch):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    weights = np.array(model.layers[3].get_weights())
    weights = np.squeeze(weights)
    language_embeddings = {v: weights[k] for k, v in dataset.id_trans_map.items() if "<pad_lang>" != v}
    keys = sorted(list(language_embeddings.keys()))
    model.save_weights(os.path.join(output_dir, "char_level_lm_weights_lstm_{}_lstm_{}.h5".format(hidden_size, epoch)))

    with open(os.path.join(output_dir, "char_level_language_embedding_lstm_{}_ep_{}.vec".format(hidden_size, epoch)), "w", encoding="utf8") as f:
        f.write("{}\t{}\n".format(len(language_embeddings), list(language_embeddings.values())[0].shape[0]))
        for k in keys:
            s = np.array2string(language_embeddings[k]).replace("\n", "")
            s = ' '.join(s[1:-1].split())
            f.write(k + " " + s + "\n")
    f.close()
    print("Epoch {} finished! Embeddings are saved!".format(epoch))


def train(args):
    output_dir = args.output
    limit = args.limit
    config_file = args.config
    input_file = args.input
    hidden_size = args.lstm_size
    batch_size = args.batch
    valid_size = args.valid
    word_vector_dim = args.ce_size
    lang_vector_dim = args.le_size
    no_of_epoch = args.epoch
    level = args.level

    if level == "word":
        dataset = char_lm_dataset_word(input_file, batch_size, valid_size)
    else:
        dataset = char_lm_dataset(input_file, config_file, batch_size=batch_size, limit=limit)
    validation_x = dataset.validation_data[0]
    validation_labels = dataset.validation_data[1]
    number_of_lang = dataset.num_of_lang
    vocab_size = dataset.char_size

    lang_model = char_language_model(vocab_size, number_of_lang, word_vector_dim, lang_vector_dim, hidden_size)
    for ep in range(no_of_epoch):
        print("--------- Epoch {} ---------".format(str(ep)))
        loss = []
        data_generator = dataset.batch_generator()
        steps_per_epoch = int(dataset.corpus_size / batch_size) + 1
        epoch_iterator = tqdm(data_generator, total=steps_per_epoch, desc="Iteration")

        for i, (x, y) in enumerate(epoch_iterator):
            l = lang_model.model.train_on_batch(x, y)
            loss.append(l)
            epoch_iterator.set_description("loss:" + str(round(np.average(loss), 4)))
        lang_model.model.evaluate(validation_x, validation_labels, batch_size=batch_size)
        save_embeddings(lang_model.model, dataset, output_dir, hidden_size, ep)
    print("done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Char-level LM to learn language embeddings')

    parser.add_argument('--input', '-i', type=str, default=config.transliterated_path)
    parser.add_argument('--output', '-o', type=str, default=config.language_embeddings_path)
    parser.add_argument('--config', type=str, default=config.json_path)

    parser.add_argument('--level', type=str, default="sentence")
    parser.add_argument('--ce_size', type=int, default=100)
    parser.add_argument('--le_size', type=int, default=100)
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--valid', type=int, default=100)
    parser.add_argument('-limit', type=int, default=2000)
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--lstm_size', type=int, default=512)

    args = parser.parse_args()
    train(args)
