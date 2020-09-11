"""
BiGRU-CRF模型对中间文件的属性词-情感词匹配方法
"""
import csv
from config import FEATURE_SENTS_TRAIN, FEATURE_SENTS_TEST, FOR_MODEL, TASK1_ANS

idx_test = []
X_test = []
y_pred = []

with open('results/temp_out.txt', encoding='utf-8') as f:
    for line in f:
        lst = line.split(',', 2)
        idx_test.append(int(lst[0]))
        X_test.append(lst[1])
        y_pred.append(eval(lst[2]))

fo_result = []

for i, sent in enumerate(X_test):
    j = 0
    while j < len(sent):
        fos = []
        tmp = []
        flag = 'OFF'
        while j < len(sent):
            token = sent[j]
            netag = y_pred[i][j]
            if token in '，。？！“”《》；：‘’【】、¥（）——～~':
                j += 1
                break
            if netag == 'B-T':
                if tmp:
                    s = ''.join(tmp)
                    fos.append((s, flag))
                    tmp = []
                tmp.append(token)
                flag = 'T'
            elif netag == 'B-O':
                if tmp:
                    s = ''.join(tmp)
                    fos.append((s, flag))
                    tmp = []
                tmp.append(token)
                flag = 'O'
            elif netag == 'I-T':
                if not tmp:
                    tmp = []
                    tmp.append(token)
                    flag = 'T'
                else:
                    tmp.append(token)
            elif netag == 'I-O':
                if not tmp:
                    tmp = []
                    tmp.append(token)
                    flag = 'O'
                else:
                    tmp.append(token)
            elif netag == 0:
                if tmp:
                    s = ''.join(tmp)
                    fos.append((s, flag))
                    tmp = []
                flag = 'OFF'
            j += 1
        if tmp:
            s = ''.join(tmp)
            fos.append((s, flag))
            tmp = []
        if not fos:
            continue
        idx = idx_test[i]
        # print(str(idx) + '\t', end='')
        # for itr in fos:
        #     print(itr[1], end='')
        # # print('\t'+''.join(itr['word'] for itr in sent))
        # print('\t'+'\t'.join("%s:%s" % (token, netag) for (token, netag) in fos))

        sta = []
        for itr in fos:
            if not sta:
                sta.append(itr)
            elif itr[1] == 'T':
                if sta[-1][1] == 'O':
                    fo_result.append((idx, itr[0], sta[-1][0]))
                    del sta[-1]
                else:
                    sta.append(itr)
            elif itr[1] == 'O':
                if sta[-1][1] == 'T':
                    fo_result.append((idx, sta[-1][0], itr[0]))
                    del sta[-1]
                else:
                    sta.append(itr)
        for itr in sta:
            if itr[1] == 'T':
                fo_result.append((idx, itr[0], '_'))
            else:
                fo_result.append((idx, '_', itr[0]))

fo_result = sorted(list(set(fo_result)))
with open(TASK1_ANS, 'w', encoding='utf-8') as f:
    for itr in fo_result:
        f.write("%d,%s,%s\n" % (itr[0], itr[1], itr[2]))