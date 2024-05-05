#! /usr/bin/python3

import sys
from os import listdir

from xml.dom.minidom import parse

from deptree import *
#import patterns


## ------------------- 
## -- Convert a pair of drugs and their context in a feature vector
def add_clue_lemma_features(tree, clue_lemmas, feats, start_index, end_index, position_label):
    for tk in range(start_index, end_index):
        lemma = tree.get_lemma(tk).lower()
        if lemma in clue_lemmas:
            feats.add(f"{lemma}_{position_label}=True")

def format_path_info(tree, path, include_lcs=False, lcs=None):
    nodes = []
    edges = []
    previous_node = None
    for node in path:
        nodes.append(tree.get_word(node))
        nodes.append(tree.get_lemma(node))
        nodes.append(tree.get_tag(node))
        
        if node != lcs:
            if previous_node is not None:
                edges.append(tree.get_rel(node))
                parent = tree.get_parent(node)
                if parent and node:
                    direction = '>' if parent == node + 1 else '<'
                    edges.append(direction)
                    edges.append('dir' if parent == node + 1 else 'indir')
            previous_node = node

    if include_lcs and lcs:
        nodes.append(tree.get_word(lcs))
        nodes.append(tree.get_lemma(lcs))
        nodes.append(tree.get_tag(lcs))
    
    return ("<".join(nodes), "<".join(edges))

def add_token_features(tree, feats, start_index, end_index, position_label):
    for c, tk in enumerate(range(start_index, end_index)):
        try:
            if tree.is_stopword(tk):
                continue
            word = tree.get_word(tk).lower()
            lemma = tree.get_lemma(tk).lower()
            tag = tree.get_tag(tk)

            feats.add(f"wb_{position_label}_{c}=" + word)
            feats.add(f"lb_{position_label}_{c}=" + lemma)
            if position_label == 'between':  # Agregamos también el tag si es entre las entidades
                feats.add(f"tb_{position_label}_{c}=" + tag)
        except:
            return set()
        
def extract_features(tree, entities, e1, e2, clue_lemmas) :
    feats = set()

    # get head token for each gold entity
    tkE1 = tree.get_fragment_head(entities[e1]['start'],entities[e1]['end'])
    tkE2 = tree.get_fragment_head(entities[e2]['start'],entities[e2]['end'])

    if tkE1 is not None and tkE2 is not None:
        # features for tokens in between E1 and E2
        #for tk in range(tkE1+1, tkE2) :
        tk=tkE1+1
        try:
            while (tree.is_stopword(tk)):
                tk += 1
        except:
            return set()
        word  = tree.get_word(tk)
        lemma = tree.get_lemma(tk).lower()
        tag = tree.get_tag(tk)
        feats.add("lib=" + lemma)
        feats.add("wib=" + word)
        feats.add("lpib=" + lemma + "_" + tag)

        tokens_between = []
        for tk in range(tkE1 + 1, tkE2):
            try:
                if not tree.is_stopword(tk):
                    word = tree.get_word(tk)
                    lemma = tree.get_lemma(tk).lower()
                    tokens_between.append((lemma, word))
            except:
                return set()

        # n-grams
        for i in range(len(tokens_between) - 2 + 1):
            ngram = " ".join([tokens_between[j][0] for j in range(i, i + 2)])
            feats.add("ngram=" + ngram)
    
        # CLUE LEMMAS
        add_clue_lemma_features(tree, clue_lemmas, feats, 0, tkE1, "before")
        add_clue_lemma_features(tree, clue_lemmas, feats, tkE1 + 1, tkE2, "between")
        total_tokens = tree.get_n_nodes()
        add_clue_lemma_features(tree, clue_lemmas, feats, tkE2 + 1, total_tokens, "after")

        eib = False
        for tk in range(tkE1 + 1, tkE2):
            if tree.is_entity(tk, entities):
                eib = True

        # indicate the presence of a THIRD ENTITY in between E1 and E2
        feats.add('eib=' + str(eib))

        # features about PATHS in the tree
        lcs = tree.get_LCS(tkE1, tkE2)

        path1 = tree.get_up_path(tkE1, lcs)
        path1 = "<".join([tree.get_lemma(x) + "_" + tree.get_rel(x) for x in path1])
        feats.add("path1=" + path1)

        path2 = tree.get_down_path(lcs, tkE2)
        path2 = ">".join([tree.get_lemma(x) + "_" + tree.get_rel(x) for x in path2])
        feats.add("path2=" + path2)

        path = path1 + "<" + tree.get_lemma(lcs) + "_" + tree.get_rel(lcs) + ">" + path2
        feats.add("path=" + path)

        # L1 distance between Entities
        distance = abs(tkE2 - tkE1)
        feats.add("distance=" + str(distance))

        # Neighbors feature
        neighbor1 = tree.get_word(tree.get_parent(tkE1)) if tree.get_parent(tkE1) is not None else 'None'
        neighbor2 = tree.get_word(tree.get_parent(tkE2)) if tree.get_parent(tkE2) is not None else 'None'
        feats.add("neighbor1=" + neighbor1)
        feats.add("neighbor2=" + neighbor2)

        # Prepositions and verbs based on PoS tag
        prepositions = [tree.get_word(tk).lower() for tk in range(tkE1 + 1, tkE2) if tree.get_tag(tk).startswith('IN')]
        verbs = [tree.get_word(tk).lower() for tk in range(tkE1 + 1, tkE2) if tree.get_tag(tk).startswith('V')]
        
        feats.add("prepositions=" + "_".join(prepositions))
        feats.add("verbs=" + "_".join(verbs))

        # Entity types
        e1_type = entities[e1].get('type', '<none>')
        e2_type = entities[e2].get('type', '<none>')
        feats.add("e1_type=" + e1_type)
        feats.add("e2_type=" + e2_type)

        # Next entity types for tkE1 and tkE2
        for i, entity in enumerate([tkE1, tkE2]):
            next_entity_types = []
            for tk in range(entity + 1, min(tree.get_n_nodes(), entity + 3)):  # up to two nodes ahead
                if tk in entities and tree.is_entity(tk, entities):
                    next_entity_type = entities[tk].get('type', '<none>')
                    next_entity_types.append(next_entity_type)
            feats.add("next_entity{}_types=".format(i + 1) + "_".join(next_entity_types))

        for tk in range(tkE1+1, tkE2):
            lemma = tree.get_lemma(tk).lower()
            tag = tree.get_tag(tk)
            if lemma in clue_lemmas and tag.startswith('V'):
                preps = [tree.get_lemma(tp) for tp in range(tk-2, tk+3) if tree.get_tag(tp).startswith('IN')]
                feats.add("clue_verb_context=" + lemma + "_" + "_".join(preps))

        # PATHS EXTENSION
        path1 = tree.get_up_path(tkE1, lcs)
        path1_nodes, path1_edges = format_path_info(tree, path1, include_lcs=False)
        feats.add("path1_nodes=" + path1_nodes)
        feats.add("path1_edges=" + path1_edges)

        path2 = tree.get_up_path(tkE2, lcs)
        path2_nodes, path2_edges = format_path_info(tree, path2, include_lcs=True, lcs=lcs)
        feats.add("path2_nodes=" + path2_nodes)
        feats.add("path2_edges=" + path2_edges)

        path = path1 + [lcs] + path2
        complete_path_nodes, complete_path_edges = format_path_info(tree, path, include_lcs=False)
        feats.add("path_nodes=" + complete_path_nodes)
        feats.add("path_edges=" + complete_path_edges)

        add_token_features(tree, feats, 0, tkE1, "before")
        add_token_features(tree, feats, tkE1 + 1, tkE2, "between")
        total_tokens = tree.get_n_nodes()
        add_token_features(tree, feats, tkE2 + 1, total_tokens, "after")

    return feats

## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  extract_features targetdir
## --
## -- Extracts feature vectors for DD interaction pairs from all XML files in target-dir
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
        stext = s.attributes["text"].value   # get sentence text
        # load sentence entities
        entities = {}
        ents = s.getElementsByTagName("entity")
        for e in ents :
           id = e.attributes["id"].value
           offs = e.attributes["charOffset"].value.split("-")           
           entities[id] = {'start': int(offs[0]), 'end': int(offs[-1])}

        # there are no entity pairs, skip sentence
        if len(entities) <= 1 : continue

        # analyze sentence
        analysis = deptree(stext)

        # for each pair in the sentence, decide whether it is DDI and its type
        pairs = s.getElementsByTagName("pair")
        for p in pairs:
            # ground truth
            ddi = p.attributes["ddi"].value
            if (ddi=="true") : dditype = p.attributes["type"].value
            else : dditype = "null"
            # target entities
            id_e1 = p.attributes["e1"].value
            id_e2 = p.attributes["e2"].value
            # feature extraction
            # clue_lemmas = ["affect", "effect","diminish","produce","increase","result","decrease", "induce", "enhance", "lower", "cause", "interact", "interaction", "shall", "caution", "advise", "reduce", "prolong", "not"]
            clue_lemmas = [
                "affect", "effect", "diminish", "produce", "increase", "result", "decrease",
                "induce", "enhance", "lower", "cause", "interact", "interaction", "shall",
                "caution", "advise", "reduce", "prolong", "not", "improve", "worsen",
                "prevent", "treat", "impair", "aggravate", "mitigate", "ameliorate", "alleviate",
                "exacerbate", "suppress", "stimulate", "modify", "alter", "counteract",
                "expose", "risk", "prevent", "contribute", "lead", "associate", "correlate",
                "potentiate", "inhibit", "block", "activate", "compete", "disrupt",
                "interfere", "facilitate", "implicate", "attribute", "suggest", "recommend",
                "advocate", "challenge", "confirm", "deny", "dispute", "substantiate", "validate"
            ]

            feats = extract_features(analysis,entities,id_e1,id_e2,clue_lemmas) 
            # resulting vector
            if len(feats) != 0:
              print(sid, id_e1, id_e2, dditype, "\t".join(feats), sep="\t")

