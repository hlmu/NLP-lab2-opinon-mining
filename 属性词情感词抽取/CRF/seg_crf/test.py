# import csv
# from config import LTP_DATA_DIR
# from config import TRAIN_REVIEW, TRAIN_LABEL, FEATURE_SENTS

# sents = {}
# with open(TRAIN_REVIEW, newline='') as csvfile:
#         reader = csv.reader(csvfile)
#         next(reader)
#         for row in reader:
#             sents[int(row[0])] = row[1]

# # 5,_, , ,很是划算,3,7,价格,正面
# # 6,香味,14,16,淡淡的,11,14,气味,正面
# # sum = 0
# # cnt = 0
# # with open(TRAIN_LABEL, newline='') as csvfile:
# #         reader = csv.reader(csvfile)
# #         next(reader)
# #         for row in reader:
# #             sum += 1
# #             idx = int(row[0])
# #             if row[1] != '_' and row[4] != '_':
# #                 t_s = int(row[2])
# #                 t_e = int(row[3])
# #                 o_s = int(row[5])
# #                 o_e = int(row[6])
# #                 s = min(t_s, o_s)
# #                 t = max(t_e, o_e)
# #                 sent = sents[int(row[0])]
# #                 for i in range(s, t):
# #                     if sent[i] in '，。！？ ':
# #                         print(sent)
# #                         print(row)
# #                         cnt += 1
# #                         continue
# #         print(cnt)
# #         print(sum)
# #         print(cnt / sum)

# # 5,_, , ,很是划算,3,7,价格,正面
# # 6,香味,14,16,淡淡的,11,14,气味,正面
# sum = 0
# cnt = 0
# with open(TRAIN_LABEL, newline='') as csvfile:
#         reader = csv.reader(csvfile)
#         next(reader)
#         for row in reader:
#             sum += 1
#             idx = int(row[0])
#             if row[1] != '_' and row[4] == '_':
#                 cnt += 1
#                 sent = sents[int(row[0])]
#                 print(sent)
#                 print(row)
#                 continue
#         print(cnt)
#         print(sum)
#         print(cnt / sum)
import pkuseg

seg = pkuseg.pkuseg(model_name='web', postag=True)  # 开启词性标注功能
text = seg.cut('我爱北京天安门')    # 进行分词和词性标注
print(text)