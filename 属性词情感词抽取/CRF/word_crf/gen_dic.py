import csv
from config import TRAIN_LABEL, TRAIN_REVIEW, OPINION_DIC, FEATURE_DIC

opinion_dic = set()
feature_dic = set()
with open(TRAIN_LABEL, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            opinion_dic.add(row[4])
            feature_dic.add(row[1])

opinion_dic.remove('_')
with open(OPINION_DIC, 'w') as f:
    for opinion in sorted(list(opinion_dic)):
        f.write(opinion + '\n')

feature_dic.remove('_')
with open(FEATURE_DIC, 'w') as f:
    for feature in sorted(list(feature_dic)):
        f.write(feature + '\n')
