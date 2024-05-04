#! /bin/bash

BASEDIR=.

# convert datasets to feature vectors
echo "Extracting features..."
python extract-features.py $BASEDIR/data/train/ > train.feat
python extract-features.py $BASEDIR/data/devel/ > devel.feat
python extract-features.py $BASEDIR/data/test/ > test.feat

# # train CRF model
# echo "Training CRF model..."
# python train-crf.py model.crf < train.feat
# # run CRF model
# echo "Running CRF model..."
# python predict.py model.crf < devel.feat > devel-CRF.out
# # evaluate CRF results
# echo "Evaluating CRF results..."
# python evaluator.py NER $BASEDIR/data/devel devel-CRF.out > devel-CRF.stats


#Extract Classification Features
cat train.feat | cut -f5- | grep -v ^$ > train.clf.feat


# train Naive Bayes model
echo "Training Naive Bayes model..."
python train-sklearn.py model_nb.joblib vectorizer_nb.joblib < train.clf.feat
# run Naive Bayes model
echo "Running Naive Bayes model..."
python predict-sklearn.py model_nb.joblib vectorizer_nb.joblib < devel.feat > devel-NB.out
# evaluate Naive Bayes results 
echo "Evaluating Naive Bayes results..."
python evaluator.py NER $BASEDIR/data/devel devel-NB.out > devel-NB.stats

# SVM
echo "Training SVM model..."
python train-svm.py model_svm.joblib vectorizer_svm.joblib < train.clf.feat
echo "Running SVM model..."
python predict-sklearn.py model_svm.joblib vectorizer_svm.joblib < devel.feat > devel-SVM.out
echo "Evaluating SVM results..."
python evaluator.py NER $BASEDIR/data/devel devel-SVM.out > devel-SVM.stats

# Test
# echo "Testing Models..."
python predict.py model.crf < test.feat > test-CRF.out
python evaluator.py NER $BASEDIR/data/test test-CRF.out > test-CRF.stats

python predict-sklearn.py model_nb.joblib vectorizer_nb.joblib < test.feat > test-NB.out
python evaluator.py NER $BASEDIR/data/test test-NB.out > test-NB.stats

python predict-sklearn.py model_svm.joblib vectorizer_svm.joblib < test.feat > test-SVM.out
python evaluator.py NER $BASEDIR/data/test test-SVM.out > test-SVM.stats


# remove auxiliary files.
rm train.clf.feat
