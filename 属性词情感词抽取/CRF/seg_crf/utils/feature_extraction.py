def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word': word,
        'postag': postag,
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word': word1,
            '-1:postag': postag1,
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word': word1,
            '+1:postag': postag1,
        })
    else:
        features['EOS'] = True

    arc = sent[i][3]
    if arc[0] == 0:
        features['HED'] = True
        # features.update({
        #     'fa:posi': arc[0],
        #     'fa:rely': arc[1]
        # })
    else:
        father_idx = arc[0]-1
        word1 = sent[father_idx][0]
        postag1 = sent[father_idx][1]
        features.update({
            'fa:word': word1,
            'fa:postag': postag1,
            'fa:posi': father_idx,
            'fa:relate': arc[1]
        })

    return features


def sent2features(sent):
    return [word2features(sent[1], i) for i in range(len(sent[1]))]

def sent2labels(sent):
    return [item[4] for item in sent[1]]

def sent2tokens(sent):
    return [item[0] for item in sent[1]]

def sent2index(sent):
    return sent[0]