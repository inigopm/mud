"""
preprocess.py:  In this file, you should add all preprocessing steps before
computing features from the data.  Some suggestions of steps that you can
apply are sentence splitting, tokenization, remove punctuation, lemmatization,
or any ad hoc step of your choice.

Some of the processes may be language-specific. As this information is our
label, we can not use this information in our preprocessing steps. All steps
must be applied equally to all sentences, or based on features you can extract
from the sentences only.

Some steps may affect the number of sentences, In those cases you would
have to implement the required measures to modify the labels accordingly.
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import sys
import pandas as pd
import re
import simplemma
import fasttext
import time
import spacy

model = fasttext.load_model('lid.176.bin')

def detect_language_fasttext(text):
    predictions = model.predict(text, k=1)  # k=1 means obtain the best prediction.
    return predictions[0][0].replace('__label__', '')

def safe_detect_language(sentence, cod='en'):
    try:
        return detect_language_fasttext(sentence)
    except Exception as e:
        print(f"Error detecting language: {e}")
        print(f"sentence{sentence}")
        return cod

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


nlp_models = {
    'en': "en_core_web_sm",     # English
    'id': "xx_ent_wiki_sm",     # Indonesian - Multi-language
    'tr': "xx_ent_wiki_sm",     # Turkish - Multi-language
    'ar': "xx_ent_wiki_sm",     # Arabic - Multi-language
    'ro': "ro_core_news_sm",    # Romanian
    'fa': "xx_ent_wiki_sm",     # Persian - Multi-language
    'es': "es_core_news_sm",    # Spanish
    'sv': "sv_core_news_sm",    # Swedish
    'et': "xx_ent_wiki_sm",     # Estonian - Multi-language
    'ko': "ko_core_news_sm",    # Korean
    'ru': "ru_core_news_sm",    # Russian
    'th': "xx_ent_wiki_sm",     # Thai - Multi-language
    'pt': "pt_core_news_sm",    # Portuguese
    'nl': 'nl_core_news_sm',    # Dutch
    'zh': 'zh_core_web_sm',     # Chinese
    'ta': "xx_ent_wiki_sm",     # Tamil - Multi-language
    'hi': "xx_ent_wiki_sm",     # Hindi - Multi-language
    'ja': 'ja_core_news_sm',    # Japanese
    'fr': 'fr_core_news_sm',    # French
    'ur': "xx_ent_wiki_sm",     # Urdu - Multi-language
    'la': "xx_ent_wiki_sm",     # Latin - Multi-language
    'it': 'it_core_news_sm',
    'de':'de_core_news_sm',
    'uk':'uk_core_news_sm',
    'fi':'fi_core_news_sm',
    'no':'nb_core_news_sm',
    'multi': 'xx_ent_wiki_sm'
}

# unique_models = set(nlp_models.values())
# for model in unique_models:
#     spacy.cli.download(model)

spacy_nlp = {lang: spacy.load(model) for lang, model in nlp_models.items()}
sentence_splitter = spacy.load('xx_sent_ud_sm') # Multilingual sentence splitter.

"""
Function to detect language with rules
"""
def detect_language_simple(text):
    rules = {
        'en': [r'the', r'and', r'or'],
        'es': [r' y ', r' o ', r'que'],
        'fr': [r' et ', r' ou ', r'que'],
        'de': [r' und ', r' oder ', r'dass'],
        'et': [r' ja ', r' või ', r'on'],
        'zh': [r'[\u4e00-\u9fff]'],
        'ja': [r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]'],
        'pt': [r' e ', r' ou ', r'que'],
        'nl': [r' en ', r' of ', r'dat'],
        'hi': [r'[ऀ-ॿ]'],
        'ar': [r'[؀-ۿ]'],
        'ko': [r'[\uac00-\ud7a3]'],
        'id': [r' dan ', r' atau ', r'yang'],
        'ro': [r' și ', r' sau ', r'că'],
        'tr': [r' ve ', r' veya ', r'ki'],
        'sv': [r' och ', r' eller ', r'att'],
        'ta': [r'[஀-௿]'],
        'ps': [r'[ا-ي]'],
        'th': [r'[\u0e00-\u0e7f]'],
        'ur': [r'[؀-ۿ]'],
        'ru': [r' и ', r' или ', r'что'],
        'la': [r' et ', r' aut ', r'qui'],
        'fa': [r'[ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی]'],
    }

    language_scores = {lang: 0 for lang in rules}
    for lang, patterns in rules.items():
        for pattern in patterns:
            if re.search(pattern, text):
                language_scores[lang] += 1

    detected_language = max(language_scores, key=language_scores.get)

    return detected_language if language_scores[detected_language] > 0 else 'en'

def safe_detect_language_simple(sentence, cod='en'):
    try:
        return detect_language_simple(sentence)
    except Exception as e:
        print(f"Error detecting language: {e}")
        print(f"sentence{sentence}")
        return cod
    
def split_into_sentences(text):
    # Using regex to better predict the end of a sentence.
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', text)
    return [sentence.strip() for sentence in sentences if sentence.strip() != '']

def split_into_sentences_spacy(text):
    doc = sentence_splitter(text)
    return [sent.text.strip() for sent in doc.sents]


def lemmatize_sentence(sentence, label, lang_code):
    
    if lang_code in spacy_nlp:
        nlp = spacy_nlp[lang_code]
        try:
            doc = nlp(sentence)
        except Exception as e:

            return "NaN"
        
        lemmatized_sentence = ' '.join([token.lemma_ if token.lemma_ != '' else token.text for token in doc])
        return lemmatized_sentence
    else:
        return sentence
    
def lemmatize_sentence2(sentence):
    
    nlp = spacy_nlp['multi']

    try:
        doc = nlp(sentence)
    except:
        return "NaN"
    
    lemmatized_sentence = ' '.join([token.lemma_ if token.lemma_ != '' else token.text for token in doc])

    return lemmatized_sentence

# Function to try no preprocessing.
def preprocess0(sentences, labels): return sentences, labels

# Tokenizer function. You can add here different preprocesses.
def preprocess(sentences, labels):
    '''
    Task: Given a sentence apply all the required preprocessing steps
    to compute train our classifier, such as sentence splitting, 
    tokenization or sentence splitting.

    Input: Sentence in string format
    Output: Preprocessed sentence either as a list or a string
    '''
    # Crear un DataFrame a partir de las series
    df = pd.DataFrame({'sentences': sentences, 'labels': labels})
    
    # # df['sentences'] = df['sentences'].apply(split_into_sentences)
    # df['sentences'] = df['sentences'].apply(split_into_sentences_spacy)

    # # Explode la columna 'sentences' para que cada frase esté en su propia fila, duplicando la etiqueta correspondiente
    # df_exploded = df.explode('sentences').reset_index(drop=True)

    df['sentences'] = df.apply(lambda row: lemmatize_sentence(row['sentences'], row['labels'], safe_detect_language(row['sentences'])), axis=1)

    new_sentences_series = df['sentences']
    new_labels_series = df['labels']

    return new_sentences_series, new_labels_series

# Tokenizer function. You can add here different preprocesses.
def preprocess2(sentences, labels):
    '''
    Task: Given a sentence apply all the required preprocessing steps
    to compute train our classifier, such as sentence splitting, 
    tokenization or sentence splitting.

    Input: Sentence in string format
    Output: Preprocessed sentence either as a list or a string
    '''
    # Crear un DataFrame a partir de las series
    df = pd.DataFrame({'sentences': sentences, 'labels': labels})
    
    # # df['sentences'] = df['sentences'].apply(split_into_sentences)
    # df['sentences'] = df['sentences'].apply(split_into_sentences_spacy)

    
    # # # Explode la columna 'sentences' para que cada frase esté en su propia fila, duplicando la etiqueta correspondiente
    # df_exploded = df.explode('sentences').reset_index(drop=True)

    df['sentences'] = df.apply(lambda row: lemmatize_sentence(row['sentences'], row['labels'], safe_detect_language_simple(row['sentences'])), axis=1)    
    # df['sentences'] = df.apply(lambda row: lemmatize_sentence2(row['sentences']), axis=1)

    new_sentences_series = df['sentences']
    new_labels_series = df['labels']

    return new_sentences_series, new_labels_series

# Tokenizer function. You can add here different preprocesses.
def preprocess3(sentences, labels):
    '''
    Task: Given a sentence apply all the required preprocessing steps
    to compute train our classifier, such as sentence splitting, 
    tokenization or sentence splitting.

    Input: Sentence in string format
    Output: Preprocessed sentence either as a list or a string
    '''
    # Crear un DataFrame a partir de las series
    df = pd.DataFrame({'sentences': sentences, 'labels': labels})
    
    # # df['sentences'] = df['sentences'].apply(split_into_sentences)
    # df['sentences'] = df['sentences'].apply(split_into_sentences_spacy)

    
    # # # Explode la columna 'sentences' para que cada frase esté en su propia fila, duplicando la etiqueta correspondiente
    # df_exploded = df.explode('sentences').reset_index(drop=True)

    df['sentences'] = df.apply(lambda row: lemmatize_sentence(row['sentences'], row['labels'], safe_detect_language_simple(row['sentences'])), axis=1)    
    # df['sentences'] = df.apply(lambda row: lemmatize_sentence2(row['sentences']), axis=1)

    new_sentences_series = df['sentences']
    new_labels_series = df['labels']

    return new_sentences_series, new_labels_series



