#!/usr/bin/env python3

import sys
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from joblib import dump, load

def fix_format(token):
    if 'BoS' in token:
        token = token.replace('BoS', 'formPrev=BoS\tsuf3Prev=BoS')
    if 'EoS' in token:
        token = token.replace('EoS', 'formNext=EoS\tsuf3Next=EoS')
    return token

def load_data(data):
    features = []
    labels = []
    for token in data:
        token = token.strip()
        token = fix_format(token).split('\t')
        token_dict = {feat.split('=')[0]: feat.split('=')[1] for feat in token[1:]}
        features.append(token_dict)
        labels.append(token[0])
    return features, labels

if __name__ == '__main__':

    model_file = sys.argv[1]
    vectorizer_file = sys.argv[2]

    train_features, y_train = load_data(sys.stdin)
    y_train = np.asarray(y_train)

    v = DictVectorizer(sparse=True)
    X_train = v.fit_transform(train_features)

    # Using a Linear SVM
    svm = LinearSVC(penalty='l2', dual=False, tol=0.0001, C=1.0)
    svm.fit(X_train, y_train)

    # Save classifier and DictVectorizer
    dump(svm, model_file)
    dump(v, vectorizer_file)
