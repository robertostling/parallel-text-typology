#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import os

import numpy as np
import onmt.opts as opts
from mulres import config
from onmt.translate.translator import build_translator
from onmt.utils.parse import ArgumentParser


def extract_language_embeds(embeddings, wordlist, output):
    embed_dict = dict()
    for id, w in enumerate(wordlist):
        if "<" in w and "-" in w:
            trans = w[1:-1]
            embed_dict[trans] = embeddings[id]

    output_dir = os.path.join(config.language_embeddings_path, "{}.vec".format(output))
    # save translation embeddings
    with open(output_dir, "w", encoding="utf8") as f:
        f.write(str(len(embed_dict)) + " " + str(len(embed_dict[list(embed_dict.keys())[0]])) + "\n")
        translations = sorted(list(embed_dict.keys()))
        for translation in translations:
            le = embed_dict[translation].cpu().detach().numpy()
            embed = " ".join(np.array_str(le).replace("\n", "")[2:-1].split())
            f.write(translation + " " + embed + "\n")
    f.close()
    print("Embeddings are saved to {}".format(output_dir))


def main(opt):
    ArgumentParser.validate_translate_opts(opt)
    translator = build_translator(opt, report_score=True)
    if "eng_to_others" in opt.models[0]:
        embeddings = translator.model.decoder.embeddings.word_lut.weight
        extract_language_embeds(embeddings, translator.fields["tgt"].base_field.vocab.itos, opt.output)
    else:
        embeddings = translator.model.encoder.embeddings.word_lut.weight
        extract_language_embeds(embeddings, translator.fields["src"].base_field.vocab.itos, opt.output)


def _get_parser():
    parser = ArgumentParser(description='translate.py')
    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()
    opt = parser.parse_args()
    main(opt)
