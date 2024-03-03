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

# Carga el modelo de detección de idioma
model = fasttext.load_model('lid.176.bin')

def detect_language_fasttext(text):
    predictions = model.predict(text, k=1)  # k=1 significa obtener la mejor predicción
    return predictions[0][0].replace('__label__', '')  # Limpia la salida para obtener solo el código del idioma

def safe_detect_language(sentence, cod='en'):
    try:
        # Suponiendo que detect_language_fasttext devuelve el código de idioma
        return detect_language_fasttext(sentence)
    except Exception as e:
        print(f"Error detecting language: {e}")
        print(f"sentence{sentence}")
        # Devuelve un código de idioma predeterminado o None si hay un error
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
}

# unique_models = set(nlp_models.values())
# for model in unique_models:
#     spacy.cli.download(model)

spacy_nlp = {lang: spacy.load(model) for lang, model in nlp_models.items()}

# Función para dividir las oraciones en frases
def split_into_sentences(text):
    # Dividir por '.', '?', '!', pero asegurándose de no dividir abreviaturas o números.
    # Puedes ajustar la expresión regular según tus necesidades específicas.
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', text)
    return [sentence.strip() for sentence in sentences if sentence.strip() != '']

def lemmatize_sentence(sentence, label, lang_code):
    
    # Verificar si el idioma y su código están soportados
    if lang_code in spacy_nlp:
        # Cargar el modelo de SpaCy para el idioma especificado
        nlp = spacy_nlp[lang_code]
        # Procesar la oración con el modelo de SpaCy
        try:
            doc = nlp(sentence)
        except:
            return "NaN"
        
        # Lematizar cada token en la oración
        lemmatized_sentence = ' '.join([token.lemma_ for token in doc])
        
        return lemmatized_sentence
    else:
        return sentence


#Tokenizer function. You can add here different preprocesses.
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
    
    # Aplicar la función para dividir cada 'sentence' en una lista de frases más pequeñas
    df['sentences'] = df['sentences'].apply(split_into_sentences)
    
    # Explode la columna 'sentences' para que cada frase esté en su propia fila, duplicando la etiqueta correspondiente
    df_exploded = df.explode('sentences').reset_index(drop=True)
    
    df_exploded['sentences'] = df_exploded.apply(lambda row: lemmatize_sentence(row['sentences'], row['labels'], safe_detect_language(row['sentences'])), axis=1)    
    new_sentences_series = df_exploded['sentences']
    new_labels_series = df_exploded['labels']
    
    print(new_sentences_series.head())
    print(new_labels_series.head())

    return new_sentences_series, new_labels_series

#Tokenizer function. You can add here different preprocesses.
def preprocess0(sentences, labels): return sentences, labels


# ("sent 1. sentx  asdasdasd","sent 2 asdasdasd")
# ("sent 1.") => [sent,1,.]
# " ".join()

# pd.series()



