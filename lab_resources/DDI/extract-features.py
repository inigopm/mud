#! /usr/bin/python3

import sys
import re
from os import listdir

from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize

import spacy

# Load a SpaCy model suitable for your language; 'en_core_web_sm' for English
nlp = spacy.load("en_core_web_sm")
   
## --------- tokenize sentence ----------- 
## -- Tokenize sentence, returning tokens and span offsets

# def tokenize(txt):
#     tokens = []
#     offset = 0
#     doc = nlp(txt.strip())  # Strip to remove leading/trailing whitespace including newlines
#     for token in doc:
#         if token.text in ['\r\n', '\n', '\r']:  # Skip newline/carriage return tokens
#             continue
#         current_position = txt.find(token.text, offset)
#         if current_position != -1:
#             offset = current_position
#         tokens.append((token.text, offset, offset + len(token.text) - 1))
#         offset += len(token.text)
#     return tokens


# def tokenize(txt):
#     # Using SpaCy's tokenizer
#     doc = nlp(txt)
#     print("doc", doc)
#     tokens = []
#     for token in doc:
#         # Normalizing all tokens to lowercase
#         normalized_text = token.text.lower()
#         # Append token text, start and end character offsets to tokens list
#         tokens.append((normalized_text, token.idx, token.idx + len(token.text) - 1))
#     return tokens


def tokenize(txt):
    offset = 0
    tks = []
    ## word_tokenize splits words, taking into account punctuations, numbers, etc.
    for t in word_tokenize(txt):
        ## keep track of the position where each token should appear, and
        ## store that information with the token
        offset = txt.find(t, offset)
        tks.append((t, offset, offset+len(t)-1))
        offset += len(t)

    ## tks is a list of triples (word,start,end)
    return tks


## --------- get tag ----------- 
##  Find out whether given token is marked as part of an entity in the XML

def get_tag(token, spans) :
   (form,start,end) = token
   for (spanS,spanE,spanT) in spans :
      if start==spanS and end<=spanE : return "B-"+spanT
      elif start>=spanS and end<=spanE : return "I-"+spanT

   return "O"
 
## --------- Feature extractor ----------- 
## -- Extract features for each token in given sentence

def extract_features(tokens, pos_tags=None):
    result = []
    for k in range(len(tokens)):
        tokenFeatures = []
        t = tokens[k][0]
        tokenFeatures.append("form=" + t)
        tokenFeatures.append("suf3=" + t[-3:])
        tokenFeatures.append("pre2=" + t[:2])
        tokenFeatures.append("token_len=" + str(len(t)))
        tokenFeatures.append("contains_digit=" + ("1" if any(char.isdigit() for char in t) else "0"))
        tokenFeatures.append("is_numeric=" + ("1" if t.isdigit() else "0"))
        tokenFeatures.append("is_upper=" + ("1" if t.isupper() else "0"))
        tokenFeatures.append("is_lower=" + ("1" if t.islower() else "0"))
        tokenFeatures.append("is_title=" + ("1" if t.istitle() else "0"))
        tokenFeatures.append("contains_punct=" + ("1" if any(char in ',.!?;:' for char in t) else "0"))

        if pos_tags:
            tokenFeatures.append("pos=" + pos_tags[k])

        if k > 0:
            tPrev = tokens[k-1][0]
            tokenFeatures.append("formPrev=" + tPrev)
        else:
            tokenFeatures.append("BoS")

        if k < len(tokens)-1:
            tNext = tokens[k+1][0]
            tokenFeatures.append("formNext=" + tNext)
        else:
            tokenFeatures.append("EoS")

        result.append(tokenFeatures)
    return result



# def extract_features(tokens) :

#    # for each token, generate list of features and add it to the result
#    result = []
#    for k in range(0,len(tokens)):
#       tokenFeatures = [];
#       t = tokens[k][0]

#       tokenFeatures.append("form="+t)
#       tokenFeatures.append("suf3="+t[-3:])

#       if k>0 :
#          tPrev = tokens[k-1][0]
#          tokenFeatures.append("formPrev="+tPrev)
#          tokenFeatures.append("suf3Prev="+tPrev[-3:])
#       else :
#          tokenFeatures.append("BoS")

#       if k<len(tokens)-1 :
#          tNext = tokens[k+1][0]
#          tokenFeatures.append("formNext="+tNext)
#          tokenFeatures.append("suf3Next="+tNext[-3:])
#       else:
#          tokenFeatures.append("EoS")
    
#       result.append(tokenFeatures)
    
#    return result


## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  baseline-NER.py target-dir
## --
## -- Extracts Drug NE from all XML files in target-dir, and writes
## -- them in the output format requested by the evalution programs.
## --


# directory with files to process
datadir = sys.argv[1]

# process each file in directory
for f in listdir(datadir) :
   
   # parse XML file, obtaining a DOM tree
   tree = parse(datadir+"/"+f)
   
   # process each sentence in the file
   sentences = tree.getElementsByTagName("sentence")
   for s in sentences :
      sid = s.attributes["id"].value   # get sentence id
      spans = []
      stext = s.attributes["text"].value   # get sentence text
      entities = s.getElementsByTagName("entity")
      for e in entities :
         # for discontinuous entities, we only get the first span
         # (will not work, but there are few of them)
         (start,end) = e.attributes["charOffset"].value.split(";")[0].split("-")
         typ =  e.attributes["type"].value
         spans.append((int(start),int(end),typ))
         

      # convert the sentence to a list of tokens
      tokens = tokenize(stext)
      # print("TOKENS", tokens)
      # sys.exit()
      # extract sentence features
      features = extract_features(tokens)

      # print features in format expected by crfsuite trainer
      for i in range (0,len(tokens)) :
         # see if the token is part of an entity
         tag = get_tag(tokens[i], spans) 
         print (sid, tokens[i][0], tokens[i][1], tokens[i][2], tag, "\t".join(features[i]), sep='\t')

      # blank line to separate sentences
      print()
