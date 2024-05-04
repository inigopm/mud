#! /bin/bash

BASEDIR=.

./corenlp-server.sh -quiet true -port 9000 -timeout 15000  &
sleep 1

# extract features
echo "Extracting features"
python extract-features.py $BASEDIR/data/devel/ > devel.cod &
python extract-features.py $BASEDIR/data/train/ | tee train.cod | cut -f4- > train.cod.cl

kill `cat /tmp/corenlp-server.running`

# train model
echo "Training model"
python train-sklearn.py model.joblib vectorizer.joblib < train.cod.cl
# run model
echo "Running model..."
python predict-sklearn.py model.joblib vectorizer.joblib < devel.cod > devel.out
# evaluate results 
echo "Evaluating results..."
python evaluator.py DDI $BASEDIR/data/devel/ devel.out > devel.stats

