# get the pre-trained src and tgt word embedding
nohup python -u preprocessing_embedding_pytorch.sh \
-emb_file ../data/glove/glove.840B.300d.txt \
-output_file ../data/WAns_corenlp_f_yifan_embeded \
-dict_file ../data/WAns_corenlp_f_nqg.vocab.pt \
> get_emb_log.out &