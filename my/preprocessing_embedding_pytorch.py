#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
import six
import sys
import numpy as np
import argparse
import torch
from tqdm import tqdm

# TRANSLATE = {  # use the same setting like learning to ask baseline
#     "-lsb-" : "[",
#     "-rsb-" : "]",
#     "-lrb-" : "(",
#     "-rrb-" : ")",
#     "-lcb-" : "{",
#     "-rcb-" : "}",
#     "-LSB-" : "[",
#     "-RSB-" : "]",
#     "-LRB-" : "(",
#     "-RRB-" : ")",
#     "-LCB-" : "{",
#     "-RCB-" : "}",
# }

# # TRANSLATE = {  # use the same setting like learning to ask baseline
# #     "[" : "-lsb-",
# #     "]" : "-rsb-",
# #     "(" : "-lrb-",
# #     ")" : "-rrb-",
# #     "{" : "-lcb-",
# #     "}" : "-rcb-",
# }

parser = argparse.ArgumentParser(description='embeddings_to_torch.py')
parser.add_argument('-emb_file', required=True,
                    help="Embeddings from this file")
parser.add_argument('-output_file', required=True,
                    help="Output file for the prepared data")
parser.add_argument('-dict_file', required=True,
                    help="Dictionary file")
parser.add_argument('-emb_dim', type=int, default=300,
                    help="Dictionary file")
parser.add_argument('-verbose', action="store_true", default=False)
opt = parser.parse_args()


def get_vocabs(dict_file):
    vocabs = torch.load(dict_file)
    # enc_vocab, dec_vocab = [vocab[1] for vocab in vocabs]
    for vocab in vocabs:
        if vocab[0] == 'src':
            enc_vocab = vocab[1]
        elif vocab[0] == 'tgt':
            dec_vocab = vocab[1]

    print("From: %s" % dict_file)
    print("\t* source vocab: %d words" % len(enc_vocab))
    print("\t* target vocab: %d words" % len(dec_vocab))

    return enc_vocab, dec_vocab


# def get_embeddings(file):
#     embs = dict()
#     for l in open(file, 'rb').readlines():
#         l_split = l.decode('utf8').strip().split(sep=" ")  # modified a little bit
#         if len(l_split) == 2:
#             continue
#         embs[l_split[0]] = [float(em) for em in l_split[1:]]
#     print("Got {} embeddings from {}".format(len(embs), file))
#
#     return embs


def match_embeddings(vocab, opt):
    dim = opt.emb_dim
    # filtered_embeddings = np.zeros((len(vocab), dim))
    filtered_embeddings = np.random.uniform(low=-1.0 / 3, high=1.0 / 3, size=(len(vocab), dim))  # modified
    filtered_embeddings = np.asarray(filtered_embeddings, dtype=np.float32)
    count = {"match": 0, "miss": 0}
    # match_set = set()
    done_flag = [0] * len(vocab)
    with open(opt.emb_file, 'rb') as emb_file:
        for line in emb_file:
            line_split = line.decode('utf-8').strip().split(sep=' ')
            w = line_split[0]
            # print(line)
            try:
                vec = [float(em) for em in line_split[1:]]
                assert len(vec) == opt.emb_dim
            except:
                if AssertionError:
                    print("len(vec)={} != opt.emb_dim={}".format(len(vec), opt.emb_dim))
                else:
                    print("Can not convert to float!")
                print(line)
                continue

            # if w in TRANSLATE:
            #     w = TRANSLATE[w]

            if w in vocab.stoi:
                w_id = vocab.stoi[w]
                filtered_embeddings[w_id] = vec
                # match_set.add(w_id)
                done_flag[w_id] = 1
                continue

            if w.lower() in vocab.stoi:
                w_id = vocab.stoi[w.lower()]
                if done_flag[w_id] == 0:
                    filtered_embeddings[w_id] = vec
                    # match_set.add(w_id)
                    done_flag[w_id] = 1
                continue

            if w.upper() in vocab.stoi:
                w_id = vocab.stoi[w.upper()]
                if done_flag[w_id] == 0:
                    filtered_embeddings[w_id] = vec
                    # match_set.add(w_id)
                    done_flag[w_id] = 1

    count['match'] = sum(done_flag)
    count['miss'] = len(vocab) - count['match']
    #
    # for w, w_id in vocab.stoi.items():
    #     if w in TRANSLATE:
    #         w = TRANSLATE[w]
    #         print("{} translated".format(w))
    #     done = False
    #     for word in (w, w.upper(), w.lower()):
    #         if word in emb:
    #             filtered_embeddings[w_id] = emb[word]
    #             count['match'] += 1
    #             done = True
    #             break
    #     if not done:
    #         if opt.verbose:
    #             print(u"not found:\t{}".format(word), file=sys.stderr)
    #         count['miss'] += 1

    return torch.Tensor(filtered_embeddings), count


def main():
    enc_vocab, dec_vocab = get_vocabs(opt.dict_file)
    # embeddings = get_embeddings(opt.emb_file)
    np.random.seed(19941023)
    filtered_enc_embeddings, enc_count = match_embeddings(enc_vocab, opt)
    filtered_dec_embeddings, dec_count = match_embeddings(dec_vocab, opt)

    print("\nMatching: ")
    match_percent = [_['match'] / (_['match'] + _['miss']) * 100
                     for _ in [enc_count, dec_count]]
    print("\t* enc: %d match, %d missing, (%.2f%%)" % (enc_count['match'],
                                                       enc_count['miss'],
                                                       match_percent[0]))
    print("\t* dec: %d match, %d missing, (%.2f%%)" % (dec_count['match'],
                                                       dec_count['miss'],
                                                       match_percent[1]))

    print("\nFiltered embeddings:")
    print("\t* enc: ", filtered_enc_embeddings.size())
    print("\t* dec: ", filtered_dec_embeddings.size())

    enc_output_file = opt.output_file + ".enc.pt"
    dec_output_file = opt.output_file + ".dec.pt"
    print("\nSaving embedding as:\n\t* enc: %s\n\t* dec: %s"
          % (enc_output_file, dec_output_file))
    torch.save(filtered_enc_embeddings, enc_output_file)
    torch.save(filtered_dec_embeddings, dec_output_file)
    print("\nDone.")


if __name__ == "__main__":
    main()
