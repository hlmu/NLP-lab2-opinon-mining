def word2features(sent, i):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'w': word,
        'p': sent[i][1][0]
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            'w-1': word1,
            'w-1:w': word1+word,
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            'w+1': word1,
            'w:w+1': word+word1,
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent[1], i) for i in range(len(sent[1]))]

def sent2labels(sent):
    return [item[2] for item in sent[1]]

def sent2tokens(sent):
    return [item[0] for item in sent[1]]

def sent2index(sent):
    return sent[0]