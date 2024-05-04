#! /usr/bin/python3

import sys
from os import listdir

from xml.dom.minidom import parse

from deptree import *
#import patterns


## ------------------- 
## -- Convert a pair of drugs and their context in a feature vector

def extract_features(tree, entities, e1, e2) :
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
      
      eib = False
      for tk in range(tkE1+1, tkE2) :
         if tree.is_entity(tk, entities):
            eib = True 
      
	  # feature indicating the presence of an entity in between E1 and E2
      feats.add('eib='+ str(eib))

      # features about paths in the tree
      lcs = tree.get_LCS(tkE1,tkE2)
      
      path1 = tree.get_up_path(tkE1,lcs)
      path1 = "<".join([tree.get_lemma(x)+"_"+tree.get_rel(x) for x in path1])
      feats.add("path1="+path1)

      path2 = tree.get_down_path(lcs,tkE2)
      path2 = ">".join([tree.get_lemma(x)+"_"+tree.get_rel(x) for x in path2])
      feats.add("path2="+path2)

      path = path1+"<"+tree.get_lemma(lcs)+"_"+tree.get_rel(lcs)+">"+path2      
      feats.add("path="+path)
      
   return feats

def extract_features1(tree, entities, e1, e2, clue_lemmas) :
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

        if tkE1 is not None and tkE2 is not None:
        # tokens between E1 and E2
            tokens_between = []
            for tk in range(tkE1 + 1, tkE2):
                try:
                    if not tree.is_stopword(tk):
                        # word = tree.get_word(tk)
                        # lemma = tree.get_lemma(tk).lower()
                        tokens_between.append((lemma, word))
                except:
                    return set()

        # n-grams extraction
        for i in range(len(tokens_between) - 2 + 1):
            ngram = " ".join([tokens_between[j][0] for j in range(i, i + 2)])
            feats.add("ngram=" + ngram)
        
        # CLUE LEMMAS
      
        for tk in range(0, tkE1):
            lemma = tree.get_lemma(tk).lower()
            if lemma in clue_lemmas:
                feats.add(lemma + f"_before=True")

        for tk in range(tkE1+1, tkE2):
            lemma = tree.get_lemma(tk).lower()
            if lemma in clue_lemmas:
                feats.add(lemma + f"_between=True")

        total_tokens = tree.get_n_nodes()
        for tk in range(tkE2+1, total_tokens):
            lemma = tree.get_lemma(tk).lower()
            if lemma in clue_lemmas:
                feats.add(lemma + f"_after=True")


        eib = False
        for tk in range(tkE1 + 1, tkE2):
            if tree.is_entity(tk, entities):
                eib = True

        # feature indicating the presence of a THIRD ENTITY in between E1 and E2
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

        # Distance between Entities feature
        distance = abs(tkE2 - tkE1)
        feats.add("distance=" + str(distance))

        # Entity Neighbors feature
        neighbor1 = tree.get_word(tree.get_parent(tkE1)) if tree.get_parent(tkE1) is not None else 'None'
        neighbor2 = tree.get_word(tree.get_parent(tkE2)) if tree.get_parent(tkE2) is not None else 'None'
        feats.add("neighbor1=" + neighbor1)
        feats.add("neighbor2=" + neighbor2)

        # TYPES OF ENTITIES
        # feats.add(word + "is_person=")
        # 'is_person'
        # 'is_organization'
        # 'is_diagnose'
        # 'is_drug'
        # 'is_crime'
        # 'is_penalty'




        # PATHS EXTENSION
        # path1 = tree.get_up_path(tkE1, lcs)
        # path1_nodes = []
        # path1_edges = []
        # for node in path1:
        #     path1_nodes.append(tree.get_word(node))
        #     path1_nodes.append(tree.get_lemma(node))
        #     path1_nodes.append(tree.get_tag(node))
        #     if node != lcs:
        #         parent = tree.get_parent(node)
        #         path1_edges.append(tree.get_rel(node))
        #         if parent and node:
        #             path1_edges.append('>' if parent == node + 1 else '<')
        #             path1_edges.append('dir' if parent == node + 1 else 'indir')
        # feats.add("path1_nodes=" + "<".join(path1_nodes))
        # feats.add("path1_edges=" + "<".join(path1_edges))
        # # path 2
        
        # path2_nodes = []
        # path2_edges = []
        # for node in path2:
        #     path2_nodes.append(tree.get_word(node))
        #     path2_nodes.append(tree.get_lemma(node))
        #     path2_nodes.append(tree.get_tag(node))
        #     if node != lcs:
        #         parent = tree.get_parent(node)
        #         path2_edges.append(tree.get_rel(node))
        #         if parent and node:
        #             path2_edges.append('>' if parent == node + 1 else '<')
        #             path2_edges.append('dir' if parent == node + 1 else 'indir')
        # feats.add("path2_nodes=" + "<".join(path2_nodes))
        # feats.add("path2_edges=" + "<".join(path2_edges))
        # full path

        # path = path1 + [lcs] + path2
        # path_nodes = []
        # path_edges = []
        # for node in path:
        #     path_nodes.append(tree.get_word(node))
        #     path_nodes.append(tree.get_lemma(node))
        #     path_nodes.append(tree.get_tag(node))
        #     if node != lcs:
        #         parent = tree.get_parent(node)
        #         path_edges.append(tree.get_rel(node))
        #         if parent and node:
        #             path_edges.append('>' if parent == node + 1 else '<')
        #             path_edges.append('dir' if parent == node + 1 else 'indir')
        # feats.add("path_nodes=" + "<".join(path_nodes))
        # feats.add("path_edges=" + "<".join(path_edges))


        # # loop before tkE1
        # for c, tk in enumerate(range(0, tkE1)):
        #     # features for tokens before E1
        #     try:
        #         if tree.is_stopword(tk):
        #             continue
        #         word = tree.get_word(tk).lower()
        #         lemma = tree.get_lemma(tk).lower()
        #         tag = tree.get_tag(tk)

        #         feats.add(f"wb_before_{tkE1 - c}=" + word)
        #         feats.add(f"lb_before_{tkE1 - c}=" + lemma)
        #         feats.add(f"tb_before_{tkE1 - c}=" + tag)
        #         feats.add(f"wlb_before_{tkE1 - c}=" + word + "_" + lemma)
        #         feats.add(f"wtb_before_{tkE1 - c}=" + word + "_" + tag)
        #         feats.add(f"ltb_before_{tkE1 - c}=" + lemma + "_" + tag)
        #         feats.add(f"wltb_before_{tkE1 - c}=" + word + "_" + lemma + "_" + tag)

        #         # Check for presence of clue verbs before tkE1
        #         for label, verb_list in zip(['moa', 'effect', 'advice', 'int'],
        #                                     [moa_clue_verbs, effect_clue_verbs, advice_clue_verbs,
        #                                      int_clue_verbs]):
        #             if word in verb_list:
        #                 feats.add(f'clue_verb_{label}_before={word}')
        #                 break
        #     except:
        #         return set()

        # # loop between tkE1 an tkE2
        # for c, tk in enumerate(range(tkE1 + 1, tkE2)):
        #     # features for tokens in between E1 and E2
        #     try:
        #         if tree.is_stopword(tk):
        #             continue
        #         word = tree.get_word(tk).lower()

        #         # Remove any non-alphanumeric characters from the word
        #         # clean_word = ''.join(c for c in word if c.isalnum())

        #         lemma = tree.get_lemma(tk).lower()
        #         tag = tree.get_tag(tk)

        #         feats.add(f"wb_between_{c}=" + word)
        #         feats.add(f"lb_between_{c}=" + lemma)
        #         feats.add(f"tb_between_{c}=" + tag)
        #         feats.add(f"wlb_between_{c}=" + word + "_" + lemma)
        #         feats.add(f"wtb_between_{c}=" + word + "_" + tag)
        #         feats.add(f"ltb_between_{c}=" + lemma + "_" + tag)
        #         feats.add(f"wltb_between_{c}=" + word + "_" + lemma + "_" + tag)

        #         # Check for presence of clue verbs between entity 1 and entity 2
        #         for label, verb_list in zip(['moa', 'effect', 'advice', 'int'],
        #                                     [moa_clue_verbs, effect_clue_verbs, advice_clue_verbs,
        #                                      int_clue_verbs]):
        #             if word in verb_list:
        #                 feats.add(f'clue_verb_{label}_between={word}')
        #                 break
        #     except:
        #         return set()

        # for c, tk in enumerate(range(tkE2 + 1, total_tokens)):
        #     # features for tokens after E2
        #     try:

        #         if tree.is_stopword(tk):
        #             continue
        #         word = tree.get_word(tk).lower()
        #         lemma = tree.get_lemma(tk).lower()
        #         tag = tree.get_tag(tk)

        #         feats.add(f"wb_after_{c}=" + word)
        #         feats.add(f"lb_after_{c}=" + lemma)
        #         feats.add(f"tb_after_{c}=" + tag)
        #         feats.add(f"wlb_after_{c}=" + word + "_" + lemma)
        #         feats.add(f"wtb_after_{c}=" + word + "_" + tag)
        #         feats.add(f"ltb_after_{c}=" + lemma + "_" + tag)
        #         feats.add(f"wltb_after_{c}=" + word + "_" + lemma + "_" + tag)

        #         # Check for presence of clue verbs after entity 2
        #         for label, verb_list in zip(['moa', 'effect', 'advice', 'int'],
        #                                     [moa_clue_verbs, effect_clue_verbs, advice_clue_verbs,
        #                                      int_clue_verbs]):
        #             if word in verb_list:
        #                 feats.add(f'clue_verb_{label}_after={word}')
        #     except:
        #         return set()

    return feats

'''

    '''


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
            clue_lemmas = ["affect", "effect","diminish","produce","increase","result","decrease", "induce", "enhance", "lower", "cause", "interact", "interaction", "shall", "caution", "advise", "reduce", "prolong", "not"]
            advice_clue_verbs = ["avoid", "do not use", "contraindicated", "caution", "not recommended", "use with care", "use alternative", "discontinue", "adjust dose"]

            effect_clue_verbs = ["increase", "decrease", "potentiate", "reduce", "enhance", "weaken", "alleviate", "worsen", "change", "affect"]

            moa_clue_verbs = ["inhibit", "induce", "block", "activate", "alter", "modulate", "stimulate", "suppress", "compete", "bind"]

            int_clue_verbs = ["interact", "co-administered", "concomitant", "combination", "administered together", "use together", "taken with"]


            feats = extract_features1(analysis,entities,id_e1,id_e2,clue_lemmas) 
            # resulting vector
            if len(feats) != 0:
              print(sid, id_e1, id_e2, dditype, "\t".join(feats), sep="\t")

