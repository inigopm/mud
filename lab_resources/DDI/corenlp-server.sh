#!/usr/bin/env bash
#
# Runs Stanford CoreNLP server

# set this path to the directory where you decompressed StanfordCore
# STANFORDDIR=r'C:\Users\Silvia\OneDrive - Universitat Politècnica de Catalunya\Escritorio\UPC\MASTER DS\1B\MUD\stanford-corenlp-4.5.6\stanford-corenlp-4.5.6'
# ../stanford-corenlp-4.5.4
STANFORDDIR=C:\\Users\\inigo\\Documents\\UPC\\Cuatri2\\MUD\\stanford-corenlp-4.5.6

if [ -f /tmp/corenlp-server.running ]; then
    echo "server already running"
else
    echo java -mx5g -cp \"$STANFORDDIR\*\" edu.stanford.nlp.pipeline.StanfordCoreNLPServer $*
    java -mx5g -cp "$STANFORDDIR/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer $* &
    echo $! > /tmp/corenlp-server.running
    wait
    rm /tmp/corenlp-server.running
fi
