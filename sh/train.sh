# you need go to onmt/ModelConstructor.py line 76 select a suitable encoder
# encoders are defined in onmt/Models.py
nohup python -u train.py \
-src_word_vec_size 300 \
-tgt_word_vec_size 300 \
-model_type text \
-encoder_type brnn \
-decoder_type rnn \
-enc_layers 2 \
-dec_layers 2 \
-rnn_size 600 \
-rnn_type LSTM \
-brnn_merge concat \
-global_attention general \
-save_model saved_models/py.sent.WAns.1gate7.1DecL.HalfDecoderHiddenSize.840B.300d.600rnn \
-seed 3435 \
-data data/WAns_corenlp_f_nqg \
-pre_word_vecs_enc data/WAns_corenlp_f_yifan_embeded.enc.pt \
-pre_word_vecs_dec data/WAns_corenlp_f_yifan_embeded.dec.pt \
-fix_word_vecs_enc
-fix_word_vecs_dec
-batch_size 64 \
-epochs 15 \
-optim sgd \
-max_grad_norm 5 \
-dropout 0.3 \
-learning_rate 1.0 \
-start_decay_at 8 \
-gpuid 0 \
> train_log.out &