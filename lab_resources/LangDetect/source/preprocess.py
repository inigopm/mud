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
        'ast': [r'ñ', r'á', r'é', r'í', r'ó', r'ú'],  # Asturiano
        'bg': [r'[\u0430-\u044f]', r'ж', r'ч', r'ш', r'ю', r'я'],  # Búlgaro
        'ca': [r'ç', r'è', r'é', r'à', r'í', r'ó', r'ú'],  # Catalán
        'cs': [r'č', r'š', r'ž', r'ý', r'á', r'í', r'é'],  # Checo
        'cy': [r'ŵ', r'ŷ'],  # Galés
        'da': [r'æ', r'ø', r'å'],  # Danés
        'de': [r'ä', r'ö', r'ü', r'ß'],  # Alemán
        'el': [r'[α-ω]', r'ί', r'ή', r'ώ', r'ά', r'έ', r'ύ'],  # Griego
        'en': [r'the', r'and'],  # Inglés
        'enm': [r'thou', r'thee', r'art', r'hast'],  # Inglés medio
        'es': [r'ñ', r'á', r'é', r'í', r'ó', r'ú'],  # Español
        'et': [r'õ', r'ä', r'ö', r'ü'],  # Estonio
        'fa': [r'[ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی]'],  # Persa
        'fi': [r'ä', r'ö'],  # Finlandés
        'fr': [r'ç', r'é', r'à', r'è', r'ù'],  # Francés
        'ga': [r'á', r'é', r'í', r'ó', r'ú'],  # Irlandés
        'gd': [r'bh', r'dh', r'gh', r'mh', r'th'],  # Gaélico escocés
        'gl': [r'ñ', r'á', r'é', r'í', r'ó', r'ú'],  # Gallego
        'gv': [r'çh', r'gh'],  # Manés
        'hbs': [r'č', r'ć', r'đ', r'š', r'ž'],  # Serbocroata
        'hi': [r'[ऀ-ॿ]'],  # Hindi
        'hu': [r'á', r'é', r'í', r'ó', r'ö', r'ü', r'ű'],  # Húngaro
        'hy': [r'[ա-ֆ]'],  # Armenio
        'id': [r' dan ', r' yang ', r' di '],  # Indonesio
        'is': [r'á', r'é', r'í', r'ó', r'ú', r'ý', r'ð', r'þ'],  # Islandés
        'it': [r'à', r'é', r'ì', r'ò', r'ù'],  # Italiano
        'ka': [r'[ა-ჰ]'],  # Georgiano
        'la': [r' et ', r' est ', r' non '],  # Latín
        'lb': [r'ä', r'ë', r'ï', r'ö', r'ü'],  # Luxemburgués
        'lt': [r'ą', r'č', r'ę', r'ė', r'į', r'š', r'ų', r'ū', r'ž'],  # Lituano
        'lv': [r'ā', r'č', r'ē', r'ģ', r'ī', r'ķ', r'ļ', r'ņ', r'š', r'ū', r'ž'],  # Letón
        'mk': [r'ѓ', r'ж', r'ѕ', r'ќ', r'ч', r'ш', r'џ'],  # Macedonio
        'ms': [r' dan ', r' yang ', r' untuk '],  # Malayo
        'nb': [r'å', r'æ', r'ø'],  # Noruego Bokmål
        'nl': [r' de ', r' het ', r' en '],  # Neerlandés
        'nn': [r'å', r'æ', r'ø'],  # Noruego Nynorsk
        'pl': [r'ą', r'ć', r'ę', r'ł', r'ń', r'ó', r'ś', r'ź', r'ż'],  # Polaco
        'pt': [r'ão', r'ç', r'ê', r'ô'],  # Portugués
        'ro': [r'ă', r'â', r'î', r'ș', r'ț'],  # Rumano
        'ru': [r'ы', r'ь', r'ю', r'я', r'э', r'щ'],  # Ruso
        'se': [r'á', r'č', r'đ', r'ŋ', r'š', r'ž'],  # Sami septentrional
        'sk': [r'á', r'č', r'ď', r'é', r'í', r'ĺ', r'ň', r'ó', r'ŕ', r'š', r'ť', r'ú', r'ý', r'ž'],  # Eslovaco
        'sl': [r'č', r'š', r'ž', r'ć'],  # Esloveno
        'sq': [r'ë', r'ç'],  # Albanés
        'sv': [r'å', r'ä', r'ö'],  # Sueco
        'sw': [r' na ', r' ya '],  # Suajili
        'tl': [r'ng', r' mga '],  # Tagalo
        'tr': [r'ğ', r'ı', r'ö', r'ş', r'ü'],  # Turco
        'uk': [r'г', r'ґ', r'є', r'і', r'ї', r'й'],  # Ucraniano
        'no': [r'å', r'æ', r'ø']  # Noruego
    }

    # Contar coincidencias de cada regla en el texto
    language_scores = {lang: 0 for lang in rules}
    for lang, patterns in rules.items():
        for pattern in patterns:
            if re.search(pattern, text):
                language_scores[lang] += 1

    # Determinar el idioma con el mayor número de coincidencias
    detected_language = max(language_scores, key=language_scores.get)

    # Si no hay coincidencias, devolver 'unknown'
    return detected_language if language_scores[detected_language] > 0 else 'unknown'

# Función para dividir las oraciones en frases
def split_into_sentences(text):
    # Dividir por '.', '?', '!', pero asegurándose de no dividir abreviaturas o números.
    # Puedes ajustar la expresión regular según tus necesidades específicas.
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', text)
    return [sentence.strip() for sentence in sentences if sentence.strip() != '']

# Función para dividir las oraciones en frases usando spaCy
def split_into_sentences_spacy(text):
    doc = sentence_splitter(text)
    return [sent.text.strip() for sent in doc.sents]


def lemmatize_sentence(sentence, label, lang_code):
    
    # Verificar si el idioma y su código están soportados
    if lang_code in spacy_nlp:
        # Cargar el modelo de SpaCy para el idioma especificado
        nlp = spacy_nlp[lang_code]
        # Procesar la oración con el modelo de SpaCy
        try:
            doc = nlp(sentence)
        except Exception as e:

            return "NaN"
        
        # Lematizar cada token en la oración
        lemmatized_sentence = ' '.join([token.lemma_ if token.lemma_ != '' else token.text for token in doc])
        return lemmatized_sentence
    else:
        return sentence
    
def lemmatize_sentence2(sentence):
    
    nlp = spacy_nlp['multi']

    # Procesar la oración con el modelo de SpaCy
    try:
        doc = nlp(sentence)
        # print("doc: ", doc)
        # post_doc = [token.lemma_ if token.lemma_ != '' else token.text for token in doc]
    except:
        return "NaN"
    
    # Lematizar cada token en la oración
    lemmatized_sentence = ' '.join([token.lemma_ if token.lemma_ != '' else token.text for token in doc])

    return lemmatized_sentence

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
    # df['sentences'] = df['sentences'].apply(split_into_sentences)
    df['sentences'] = df['sentences'].apply(split_into_sentences_spacy)

    
    # Explode la columna 'sentences' para que cada frase esté en su propia fila, duplicando la etiqueta correspondiente
    df_exploded = df.explode('sentences').reset_index(drop=True)

    print("aaaaa: ", df_exploded)

    df_exploded['sentences'] = df_exploded.apply(lambda row: lemmatize_sentence(row['sentences'], row['labels'], safe_detect_language(row['sentences'])), axis=1)    
    # df_exploded['sentences'] = df_exploded.apply(lambda row: lemmatize_sentence2(row['sentences']), axis=1)

    print("bbbbb: ", df_exploded)

    new_sentences_series = df_exploded['sentences']
    new_labels_series = df_exploded['labels']

    return new_sentences_series, new_labels_series

#Tokenizer function. You can add here different preprocesses.
def preprocess0(sentences, labels): return sentences, labels


# ("sent 1. sentx  asdasdasd","sent 2 asdasdasd")
# ("sent 1.") => [sent,1,.]
# " ".join()

# pd.series()



