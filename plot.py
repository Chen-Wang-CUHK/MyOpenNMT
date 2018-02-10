
# # check the first question word
# file1 = open('data\\corenlp_data\\rich_feature\\tgt-test.txt', encoding='utf-8')
# file2 = open('log\\LuaStyle_Corenlp_f_Baseline_log\\no_gate\\sent\\test_nongated_use_lua_brnn_epoch15.txt', encoding='utf-8')
# file3 = open('log\\LuaStyle_Corenlp_f_Baseline_log\\gated\\sent\\sigmoid(W[hi,s]_b)_600\\test_sigmoid(W[hi,s]_b)_600_use_lua_brnn_epoch15.txt', encoding='utf-8')
#
# lines1 = file1.readlines()
# lines2 = file2.readlines()
# lines3 = file3.readlines()
#
# cnt1 = 0
# cnt2 = 0
# for i in range(len(lines1)):
#     word1 = lines1[i].split()[0]
#     word2 = lines2[i].split()[0]
#     if word2 == word1:
#         cnt1 += 1
#     word3 = lines3[i].split()[0]
#     if word3 == word1:
#         cnt2 += 1
#     print(word1, word2, word3)
#
# print("nogate:{}, gate:{}".format(cnt1/len(lines1), cnt2/len(lines1)))
baseline_file = open('log\\PytorchStyle_Corenlp_f_log\\nogate\\py_sent_train_nogate_f_log.txt')
gated_file1 = open('log\\PytorchStyle_Corenlp_f_log\\gated\\2gate5\\py_sent_train_2gate5_corenlp_f_log.txt')
gated_file2 = open('log\\sent_corenlp_f_log.txt')

bl_lines = baseline_file.readlines()
lines1 = gated_file1.readlines()
lines2 = gated_file2.readlines()

def get_dev_ppl(lines):
    dev_ppls = []
    train_ppls = []
    for i in range(len(lines)):
        line = lines[i]
        if 'Validation perplexity' in line:
            dev_ppls.append(float(line.split()[-1]))
        if 'Train perplexity' in line:
            train_ppls.append(float(line.split()[-1]))
    return dev_ppls, train_ppls

bl_dev_ppls, bl_train_ppls = get_dev_ppl(bl_lines)
dev_ppls1, train_ppls1 = get_dev_ppl(lines1)
dev_ppls2, train_ppls2 = get_dev_ppl(lines2)

import matplotlib.pyplot as plt

x = [i for i in range(1, 16)]
plt.plot(x, bl_dev_ppls, 'r', #x, bl_train_ppls, 'r-',
         x, dev_ppls1, 'b', #x, train_ppls1, 'b-',
         x, dev_ppls2, 'g')#, x, dev_ppls2, 'g-')
plt.show()