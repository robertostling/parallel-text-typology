import sys
from prepare_data.embeddings import Embeddings
from collections import defaultdict
import pickle


def prepare_data_we_prediction(paradigm_dir, embedding_dir, common_translation_dir):
    all_data = defaultdict(dict)
    common_translations = defaultdict(list)
    paradigm_dir += "/%s.paradigms"
    for line in open(common_translation_dir, "r", encoding="utf8").readlines():
        common_translations[line.split("-")[0]].append(line.replace("\n",""))
    for lang, translations in common_translations.items():
        try:
            embed = Embeddings(embedding_dir % lang).embeddings
        except:
            print("no embeddings for ", lang)
            continue
        for translation in translations:
            paradigms_raw = [line.replace("\n", "") for line in open(paradigm_dir % translation, "r", encoding="utf8").readlines()]
            training_data = []
            for line in paradigms_raw:
                pos_tag = line.split(" ")[0]
                if pos_tag not in ["NOUN", "VERB"]:
                    continue
                words = [( pair.split("/")[-1], embed[pair.split("/")[0]]) for pair in line.split(" ")[3:] if len(pair.split("/")[1]) > 1 and pair.split("/")[0] in embed.keys()]
                if len(words) >= 1:
                    training_data.extend(words)
            if len(training_data) > 0:
                all_data[translation] = training_data
                print(translation, len(training_data))
    pickle.dump(all_data, open("pickle/we_prediction_data_small.pickle", "wb"))


if __name__ == "__main__":
    prepare_data_we_prediction(sys.argv[1], sys.argv[2], sys.argv[3])
