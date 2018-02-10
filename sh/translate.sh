nohup python -u translate.py \
-model saved_models/py.sent.WAns.1gate7.1DecL.HalfDecoderHiddenSize.840B.300d.600rnn_acc_48.81_ppl_23.76_e15.pt \
-output log/py_sent_test_WAns_1gate7_1DecL_epoch15.txt \
-src data/corenlp_data/rich_feature/src-test.txt \
-ans data/corenlp_data/rich_feature/src-ans-test.txt \
-tgt data/corenlp_data/rich_feature/tgt-test.txt \
-beam_size 3 \
-replace_unk \
-batch_size 64 \
-gpu 0 \
> translate_log.out &