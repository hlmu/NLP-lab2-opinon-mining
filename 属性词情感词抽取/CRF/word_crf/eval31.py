# -*- coding: utf-8 -*-
'''
任务3.1的评价函数
输入：（正确标注的标注的文件，待测试标注文件）
正确标注文件第一行有id,AspectTerms,A_start,A_end,OpinionTerms,O_start,O_end,Categories,Polarities
'''

import csv
from config import TASK1_ANS, TRAIN_LABEL

# 三元组评价函数
def evaluate(correct_file, my_file):
    # 5,_, , ,很是划算,3,7,价格,正面
    # 6,香味,14,16,淡淡的,11,14,气味,正面
    correct_tags = {}
    with open(correct_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            idx = int(row[0])
            if idx not in correct_tags:
                correct_tags[idx] = []
            correct_tags[idx].append((row[1],row[4]))
    
    # 1,_,是个不错
    # 1,_,正品
    # 2,_,喜欢的
    my_tags = {}
    with open(my_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            idx = int(row[0])
            if idx not in my_tags:
                my_tags[idx] = []
            my_tags[idx].append((row[1],row[2]))
    
    all_correct = sum([len(val) for key,val in correct_tags.items()])
    all_my = sum([len(val) for key,val in my_tags.items()])

    correct_count = 0
    max_num = max(correct_tags.keys())
    for i in range(max_num):
        if i in my_tags and i in correct_tags:
            cor = correct_tags[i]
            my = my_tags[i]
            for my_tmp in my:
                if my_tmp in cor:
                    correct_count += 1
                    cor.remove(my_tmp)
    
    precise = correct_count / all_correct
    recall = correct_count / all_my
    f1 = (2 * precise * recall) / (precise + recall)
    return precise, recall, f1

if __name__ == '__main__':
    print(evaluate(TRAIN_LABEL, TASK1_ANS))