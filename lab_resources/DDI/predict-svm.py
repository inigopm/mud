#!/usr/bin/env python3

import sys
from joblib import load
from sklearn.feature_extraction import DictVectorizer

def instances(fi):
    xseq = []
    toks = []
    for line in fi:
        line = line.strip()
        if not line:
            yield xseq, toks
            xseq = []
            toks = []
            continue
        fields = line.split('\t')
        item = fields[5:]        
        xseq.append(item)
        toks.append([fields[0], fields[1], fields[2], fields[3], fields[4]])
    if xseq:
        yield xseq, toks

def prepare_instances(xseq):
    features = []
    for token in xseq:
        token_dict = {feat.split('=')[0]: feat.split('=')[1] for feat in token}
        features.append(token_dict)
    return features

def main():
    model_file = sys.argv[1]
    vectorizer_file = sys.argv[2]

    model = load(model_file)
    vectorizer = load(vectorizer_file)

    for xseq, toks in instances(sys.stdin):
        xseq = prepare_instances(xseq)
        X = vectorizer.transform(xseq)
        predictions = model.predict(X)

        for tok, label in zip(toks, predictions):
            sid, form, start, end, _ = tok
            print(sid, form, start, end, label, sep='\t')

if __name__ == "__main__":
    main()
