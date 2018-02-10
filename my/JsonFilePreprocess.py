import json
import argparse
import os
from tqdm import tqdm


def PtStyleDataGenerator(opt, data_type):
    assert data_type in ['train', 'dev', 'test']
    print("\nProcessing {} set...".format(data_type))
    # ======== read data ==========
    # *x = *p = *cx: *xi save [article_idx, paragraph_idx (context)] index pair for each question. [question_idx]
    # ids: store queestion hash id for each question. [question_idx]
    # q_lower: store lower-cased questions. [question_idx, word_idx]
    # cq_lower: store the lower-cased and charaterized questions. [question_idx, word_idx, char_idx]
    # answerss_lower: store lower-cased answerss for each question. [question_idx, ans_idx]
    # y: (question_idx, ans_idx) get the answer span: [[StartSentIdx, StartWordIdx], [EndSentIdx, EndWordIdx]}
    # na: (no answer) bool list for each question, True if no answer, False if answer exists. [question_idx]
    data = json.load(open(os.path.join(opt.data_dir, "rawJsonData\\data_{}.json".format(data_type)), encoding="utf-8"))
    questions = data['q_lower']
    # answerss = data['answerss_lower']
    ans_spanss = data['y']
    context_idxs = data['*x']

    # ======== read shared data ==========
    # char_counter: the char counter
    # lower_word_counter: the word counter
    # lower_word2vec: the lowered word2vec dict
    # p_lower: original lowered paragraphs, (article_idx, parag_idx)
    # x_lower: tokenized lowered sent-split paragraphs, (article_idx, parag_idx, sent_idx, wrod_idx)
    # cx_lower: char context, (article_idx, parag_idx, sent_idx, wrod_idx, char_idx)
    # ner: ner tag for each word in paragraphs. (article_idx, parag_idx, sent_idx, word_idx)
    # pos: pos tag for each word in paragraphs. (article_idx, parag_idx, sent_idx, word_idx)
    shared = json.load(open(os.path.join(opt.data_dir, "rawJsonData\\shared_{}.json".format(data_type)), encoding="utf-8"))
    x_lower = shared['x_lower']
    x_ner = shared['ner']
    x_pos = shared['pos']

    # read stop words
    stop_words = json.load(open(opt.stop_words_file, encoding='utf-8'))

    # ======== generate pytorch-style data file ==========
    if opt.rich_feature:
        saved_path = os.path.join(opt.data_dir, 'rich_feature')
        if not os.path.exists(saved_path):
            os.mkdir(saved_path)
        para_src_file = open(os.path.join(saved_path, 'para-{}.txt'.format(data_type)), 'w', encoding='utf-8')
        para_posi_file = open(os.path.join(saved_path, 'para-posi-{}.txt'.format(data_type)), 'w', encoding='utf-8')
        sents_src_file = open(os.path.join(saved_path, 'src-{}.txt'.format(data_type)), 'w', encoding='utf-8')
        sents_posi_file = open(os.path.join(saved_path, 'src-posi-{}.txt'.format(data_type)), 'w', encoding='utf-8')
        tgt_file = open(os.path.join(saved_path, 'tgt-{}.txt'.format(data_type)), 'w', encoding='utf-8')
    else:
        saved_path = os.path.join(opt.data_dir, 'non_rich_feature')
        if not os.path.exists(saved_path):
            os.mkdir(saved_path)
        para_src_file = open(os.path.join(saved_path, 'nf-para-{}.txt'.format(data_type)), 'w', encoding='utf-8')
        para_posi_file = open(os.path.join(saved_path, 'nf-para-posi-{}.txt'.format(data_type)), 'w',
                              encoding='utf-8')
        sents_src_file = open(os.path.join(saved_path, 'nf-src-{}.txt'.format(data_type)), 'w', encoding='utf-8')
        sents_posi_file = open(os.path.join(saved_path, 'nf-src-posi-{}.txt'.format(data_type)), 'w',
                               encoding='utf-8')
        tgt_file = open(os.path.join(saved_path, 'nf-tgt-{}.txt'.format(data_type)), 'w', encoding='utf-8')

    # count the filted data for training set
    filted_cnt = 0
    total_num = len(questions)
    for q_idx in tqdm(range(total_num)):
        q = questions[q_idx]
        ans_spans = ans_spanss[q_idx]
        context_idx = context_idxs[q_idx]
        # get paragraph source
        context = x_lower[context_idx[0]][context_idx[1]]   # [sent_idx, word_idx]
        context_ner = x_ner[context_idx[0]][context_idx[1]]   # [sent_idx, word_idx]
        context_pos = x_pos[context_idx[0]][context_idx[1]]   # [sent_idx, word_idx]
        context_lens = [len(sent_i) for sent_i in context]
        ans_span = ans_spans[0]

        # flatten the paragraph
        flat_context = []
        for sent_i in context:
            flat_context.extend(sent_i)
        # check whether exits non-stop-words overlap when generating training data
        if data_type == 'train':
            ans_sents_start = sum(context_lens[:ans_span[0][0]])
            ans_sents_end = sum(context_lens[:ans_span[1][0] + 1])
            flat_sents = flat_context[ans_sents_start:ans_sents_end]
            if not check_src_tgt(flat_sents, q, stop_words):
                filted_cnt += 1
                continue

        flat_context_ner = []
        for sent_ner_i in context_ner:
            flat_context_ner.extend(sent_ner_i)
        flat_context_pos = []
        for sent_pos_i in context_pos:
            flat_context_pos.extend(sent_pos_i)
        flat_ans_span = [sum(context_lens[:ans_span[0][0]]) + ans_span[0][1],
                         sum(context_lens[:ans_span[1][0]]) + ans_span[1][1]]
        assert context[ans_span[0][0]][ans_span[0][1]] == flat_context[flat_ans_span[0]]
        assert context[ans_span[1][0]][ans_span[1][1] - 1] == flat_context[flat_ans_span[1] - 1]
        assert len(flat_context) == len(flat_context_ner)
        assert len(flat_context) == len(flat_context_pos)
        # get the feature-rich paragraph
        feature_rich_flat_parag = []
        for i in range(len(flat_context)):
            # word_i = flat_context[i].replace(' ', '').replace('\xa0', '').replace('\u202f', '')
            word_i = ''.join(flat_context[i].split())
            ner_i = flat_context_ner[i]
            pos_i = flat_context_pos[i]
            is_ans = 'I' if i in range(flat_ans_span[0], flat_ans_span[1]) else 'O'
            if opt.rich_feature:
                feature_rich_word = word_i + "￨" + ner_i + "￨" + pos_i + "￨" + is_ans
                # check non-breaking space
                # if '½' in word_i:
                #     print(flat_context[i-1], flat_context_ner[i-1], flat_context_pos[i-1])
                #     print(word_i)
                #     print([ord(char) for char in word_i])
            else:
                feature_rich_word = word_i
            feature_rich_flat_parag.append(feature_rich_word)

        # get sentence source and sentence answer span
        ans_sents_start = sum(context_lens[:ans_span[0][0]])
        ans_sents_end = sum(context_lens[:ans_span[1][0] + 1])
        feature_rich_flat_sents = feature_rich_flat_parag[ans_sents_start:ans_sents_end]
        flat_sents_ans_span = [flat_ans_span[0] - ans_sents_start,
                               flat_ans_span[1] - ans_sents_start]

        # write these data into file
        line = ' '.join(feature_rich_flat_parag) + '\n'
        para_src_file.write(line)
        line = ' '.join([str(tmp) for tmp in flat_ans_span]) + '\n'
        para_posi_file.write(line)
        line = ' '.join(feature_rich_flat_sents) + '\n'
        sents_src_file.write(line)
        line = ' '.join([str(tmp) for tmp in flat_sents_ans_span]) + '\n'
        sents_posi_file.write(line)
        line = ' '.join(q) + '\n'
        tgt_file.write(line)

    para_src_file.close()
    para_posi_file.close()
    sents_src_file.close()
    sents_posi_file.close()
    tgt_file.close()
    if data_type == 'train':
        print("Total qs: {}, filted: {}, percentage: {:.2f}".format(total_num, filted_cnt, filted_cnt/total_num))
    else:
        print("Total qs: {}".format(total_num))


def check_src_tgt(src, tgt, stopwords):
    overlap = set(src) & set(tgt)
    for word in overlap:
        if word not in stopwords:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="JsonFilePreprocess")
    parser.add_argument('-data_dir', type=str, default='..\\data\\corenlp_data')
    parser.add_argument('-stop_words_file', type=str,
                        default='..\\data\\corenlp_data\\stfd_stopwords\\corenlp_stopwords.json')
    parser.add_argument('-rich_feature', type=bool, default=True)
    opt = parser.parse_args()
    print("========Generate rich-feature data...========")
    PtStyleDataGenerator(opt, data_type='dev')
    PtStyleDataGenerator(opt, data_type='test')
    PtStyleDataGenerator(opt, data_type='train')

    opt.rich_feature = False
    print("\n\n========Generate non-rich-feature data...========")
    PtStyleDataGenerator(opt, data_type='dev')
    PtStyleDataGenerator(opt, data_type='test')
    PtStyleDataGenerator(opt, data_type='train')


if __name__ == '__main__':
    main()