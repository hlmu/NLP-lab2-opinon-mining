from config import TRAIN_REVIEW, TEST_REVIEW, TRAIN_LABEL, TRAIN_REVIEW_CLEANED, TRAIN_LABEL_CLEANED, TEST_REVIEW_CLEANED
import csv

# 对文本进行序列标注
def netagger(sent, sent_id):
    ne = [('O', -1)] * len(sent)
    tags = sent_tags[sent_id]
    for tag in tags:
        start_pos = tag[0]
        end_pos = tag[1]
        category = tag[2]
        label_id = tag[3]
        first = False
        for idx in range(len(sent)):
            if idx >= start_pos and idx < end_pos:
                if not first:
                    ne[idx] = ('B-T', label_id) if category == 'feature' else ('B-O', label_id)
                    first = True
                else:
                    ne[idx] = ('I-T', label_id) if category == 'feature' else ('I-O', label_id)
    return ne

def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
        return True
    else:
        return False

# 输入sent为(sent_id, [(char, (tag, label_id)), (char, (tag, label_id)), (char, (tag, label_id))...])
def wash_one_sent(sent):
    sent_id = sent[0]
    # 将语句转换为unicode格式
    # sent_tmp = [(char.decode('utf-8'), tag) for char,tag in sent[1]]
    sent_tmp = sent[1]

    # 去除末尾所有非中文的部分
    idx = len(sent_tmp) - 1
    while idx >= 0 and not is_chinese(sent_tmp[idx][0]):
        sent_tmp = sent_tmp[:-1]
        idx -= 1
    # 处理非法字符情况
    idx = 0
    while idx < len(sent_tmp):
        if sent_tmp[idx][0] not in ['，','。','：','！','？','*']\
             and not is_chinese(sent_tmp[idx][0])\
                 and sent_tmp[idx][1][0] not in ['B-T', 'B-O', 'I-T', 'I-O']:
            sent_tmp = sent_tmp[0:idx] + sent_tmp[idx+1:]
        else:
            idx += 1
    # 处理连续逗号情况
    idx = 0
    while idx < len(sent_tmp) - 1:
        if sent_tmp[idx][0] == '，' and sent_tmp[idx + 1][0] == '，':
            sent_tmp = sent_tmp[0:idx] + sent_tmp[idx+1:]
        else:
            idx += 1
    return (sent_id, sent_tmp)


sent_tags = {}
labels_raw = {}
# 5,_, , ,很是划算,3,7,价格,正面
# 6,香味,14,16,淡淡的,11,14,气味,正面
# sent_tags {idx:(start_pos, end_pos, 'type', label_id}
with open(TRAIN_LABEL, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    label_id = 1
    for row in reader:
        idx = int(row[0])
        if idx not in sent_tags:
            sent_tags[idx] = []
        if row[1] != '_':
            sent_tags[idx].append((int(row[2]), int(row[3]), 'feature', label_id))
        if row[4] != '_':
            sent_tags[idx].append((int(row[5]), int(row[6]), 'opinion', label_id))
        labels_raw[label_id] = (row[7], row[8])
        label_id += 1

sents_train = []
with open(TRAIN_REVIEW, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        # 处理空格
        row[1] = row[1].replace(' ', '，')
        sent_id = int(row[0])
        sent = row[1]
        netags = netagger(sent, sent_id)
        sents_train.append((sent_id, [(char, tag) for char, tag in zip(sent, netags)]))

sent_cleaned = []
for sent in sents_train:
    sent_cleaned.append(wash_one_sent(sent))

with open(TRAIN_LABEL_CLEANED, 'w+', encoding='utf-8') as f:
    f.write('id,Reviews\n')
    for (sent_id, sent_tmp) in sent_cleaned:
        sent_ = ''.join([tmp[0] for tmp in sent_tmp])
        f.write(str(sent_id) + ',' + sent_ + '\n')

labels = {}
for sent_id, sent_tmp in sent_cleaned:
    i = 0
    while i < len(sent_tmp):
        word, (tag, label_id) = sent_tmp[i]
        if tag == 'O':
            i += 1
            continue
        start_pos = i
        i += 1
        while i < len(sent_tmp) and (sent_tmp[i][1][0] == 'I-O' or sent_tmp[i][1][0] == 'I-T'):
            i += 1
        end_pos = i
        word = ''.join([tmp[0] for tmp in sent_tmp[start_pos:end_pos]])    

        if sent_id not in labels:
            labels[sent_id] = {}
        
        if label_id not in labels[sent_id]:
                labels[sent_id][label_id] = []
        
        if tag == 'B-T':
            labels[sent_id][label_id].append((1, word, start_pos, end_pos))
        else:
            labels[sent_id][label_id].append((2, word, start_pos, end_pos))

with open(TRAIN_LABEL_CLEANED, 'w+', encoding='utf-8') as f:
    f.write('id,AspectTerms,A_start,A_end,OpinionTerms,O_start,O_end,Categories,Polarities\n')
    for sent_id, tmp_labels in sorted(labels.items(), key=lambda x: x[0]):
        for label_id, fo_lst in sorted(tmp_labels.items(), key=lambda x: x[0]):
            if len(fo_lst) == 0:
                print('为啥是0',sent_id)
            elif len(fo_lst) == 1:
                flag, word, start_pos, end_pos = fo_lst[0]
                if flag == 1:
                    f.write(str(sent_id) + ',' + word + ',' + str(start_pos) + ',' +\
                        str(end_pos) + ',_, , ,' + labels_raw[label_id][0] + ',' + labels_raw[label_id][1] + '\n')
                else:
                    f.write(str(sent_id) + ',_, , ,' + word + ',' + str(start_pos) +',' +\
                    str(end_pos) + ',' + labels_raw[label_id][0] + ',' + labels_raw[label_id][1] + '\n')
            elif len(fo_lst) == 2:
                [(flag1, feature, start_pos1, end_pos1), (flag2, opinion, start_pos2, end_pos2)] = \
                    sorted(fo_lst, key=lambda x: x[0])
                f.write(str(sent_id) + ',' + feature + ',' + str(start_pos1) + ',' + str(end_pos1) + ',' +\
                    opinion + ',' + str(start_pos2) +',' + str(end_pos2) + ',' + labels_raw[label_id][0] + ',' + labels_raw[label_id][1] + '\n')
            else:
                print('超啦啊啊啊',sent_id)

sents_test = []
with open(TEST_REVIEW, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        # 处理空格
        row[1] = row[1].replace(' ', '，')
        sent_id = int(row[0])
        sent = row[1]
        sents_test.append((sent_id, sent))

# 输入格式是(sent_id, sent)
def wash_one_test_sent(sent):
    sent_id = sent[0]
    sent_tmp = sent[1]

    # 去除末尾所有非中文的部分
    idx = len(sent_tmp) - 1
    while idx >= 0 and not is_chinese(sent_tmp[idx]):
        sent_tmp = sent_tmp[:-1]
        idx -= 1
    
    idx = 0
    while idx < len(sent_tmp):
        if sent_tmp[idx] not in ['，','。','：','！','？','*']\
             and not is_chinese(sent_tmp[idx][0]):
            sent_tmp = sent_tmp[0:idx] + sent_tmp[idx+1:]
        else:
            idx += 1

    # 处理连续逗号情况
    idx = 0
    while idx < len(sent_tmp) - 1:
        if sent_tmp[idx][0] == '，' and sent_tmp[idx + 1][0] == '，':
            sent_tmp = sent_tmp[0:idx] + sent_tmp[idx+1:]
        else:
            idx += 1
    return (sent_id, sent_tmp)

with open(TEST_REVIEW_CLEANED, 'w+', encoding='utf-8') as f:
    f.write('id,Review\n')
    for sent in sents_test:
        sent = wash_one_test_sent(sent)
        f.write(str(sent[0]) + ',' + sent[1] + '\n')