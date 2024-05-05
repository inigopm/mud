#! /bin/bash

BASEDIR=.

# ./corenlp-server.sh -quiet true -port 9000 -timeout 15000  &
# sleep 1

# extract features
echo "Extracting features"
python extract-features.py $BASEDIR/data/devel/ > devel.cod &
python extract-features.py $BASEDIR/data/train/ | tee train.cod | cut -f4- > train.cod.cl

# kill `cat /tmp/corenlp-server.running`

# train model
echo "Training model"
python train-sklearn.py model.joblib vectorizer.joblib < train.cod.cl
# run model
echo "Running model..."
python predict-sklearn.py model.joblib vectorizer.joblib < devel.cod > devel.out
# evaluate results 
echo "Evaluating results..."
python evaluator.py DDI $BASEDIR/data/devel/ devel.out > devel.stats


# # train Naive Bayes model
# echo "Training Naive Bayes model..."
# python train-sklearn.py model_nb.joblib vectorizer_nb.joblib < train.cod.cl
# # run Naive Bayes model
# echo "Running Naive Bayes model..."
# python predict-sklearn.py model_nb.joblib vectorizer_nb.joblib < devel.cod > devel-NB.out
# # evaluate Naive Bayes results 
# echo "Evaluating Naive Bayes results..."
# python evaluator.py NER $BASEDIR/data/devel devel-NB.out > devel-NB.stats

# # SVM
# echo "Training SVM model..."
# python train-svm.py model_svm.joblib vectorizer_svm.joblib < train.cod.cl
# echo "Running SVM model..."
# python predict-svm.py model_svm.joblib vectorizer_svm.joblib < devel.cod > devel-SVM.out
# echo "Evaluating SVM results..."
# python evaluator.py NER $BASEDIR/data/devel devel-SVM.out > devel-SVM.stats
