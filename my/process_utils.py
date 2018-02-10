import json


def mask_ans_words(words_list):
    masked_words = []
    for word in words_list:
        if word[-1] == 'I':
            word = word.split('￨')
            word[0] = '<unk>'
            word = '￨'.join(word)
        masked_words.append(word)
    line = ' '.join(masked_words) + '\n'
    return line


def filter_stop_words(words_list):
    masked_words = []
    stop_words_file = open("..\\data\\corenlp_data\\stfd_stopwords\\corenlp_stopwords.json")
    stop_words = json.load(stop_words_file)
    stop_words = set(stop_words)
    for word in words_list:
        word = word.split("￨")
        if word[0] in stop_words:
            continue
        word = "￨".join(word)
        masked_words.append(word)
    line = ' '.join(masked_words) + '\n'
    return line


def extract_ans(words_list):
    ans_words = []
    for word in words_list:
        if word[-1] == 'I':
            ans_words.append(word)
    line = ' '.join(ans_words) + '\n'
    return line


def process(data_type, process_fc):
    para_file = open("..\\data\\corenlp_data\\rich_feature\\para-{}.txt".format(data_type), encoding='utf-8')
    src_file = open("..\\data\\corenlp_data\\rich_feature\\src-{}.txt".format(data_type), encoding='utf-8')
    para_lines = para_file.readlines()
    src_lines = src_file.readlines()
    processed_para = []
    processed_src = []
    for i in range(len(src_lines)):
        para_line = para_lines[i].strip().split()
        processed_para.append(process_fc(para_line))
        src_line = src_lines[i].strip().split()
        processed_src.append(process_fc(src_line))
    masked_para_file = open("..\\data\\corenlp_data\\rich_feature\\para-ans-{}.txt".format(data_type),
                            'w', encoding='utf-8')
    masked_src_file = open("..\\data\\corenlp_data\\rich_feature\\src-ans-{}.txt".format(data_type),
                           'w', encoding='utf-8')
    masked_para_file.writelines(processed_para)
    masked_src_file.writelines(processed_src)
    return processed_src


if __name__ == '__main__':
    # # mask answer words
    # process('dev', mask_ans_words)
    # process('test', mask_ans_words)
    # process('train', mask_ans_words)
    # # stop words filtering
    # process('dev', filter_stop_words)
    # process('test', filter_stop_words)
    # process('train', filter_stop_words)
    # answer words extracting
    dev_ans = process('dev', extract_ans)
    test_ans = process('test', extract_ans)
    train_ans = process('train', extract_ans)
    total_ans = train_ans + dev_ans
    total_cnt = [0]*50
    total_percent = [0.]*50
    for line in total_ans:
        word_list = line.strip().split()
        ans_len = len(word_list)
        total_cnt[ans_len - 1] += 1
    print(["{}: {}".format(idx + 1, cnt) for idx, cnt in enumerate(total_cnt)])
    total_num = sum(total_cnt)
    for i in range(50):
        total_percent[i] = sum(total_cnt[:i+1])/total_num
    print(["{}: {:.5f}".format(idx + 1, p) for idx, p in enumerate(total_percent)])


