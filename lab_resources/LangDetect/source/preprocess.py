import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import sys
import pandas as pd
import re
import simplemma

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

language_codes = {
    'Dutch': 'nl',
    'Japanese': 'ja',
    'Spanish': 'es',
    'Hindi': 'hi',
    'Pushto': 'ps',
    'Korean': 'ko',
    'Estonian': 'et',
    'Chinese': 'zh',
    'Arabic': 'ar',
    'Latin': 'la',
    'Thai': 'th',
    'Indonesian': 'id',
    'Persian': 'fa',
    'English': 'en',
    'Tamil': 'ta',
    'Urdu': 'ur',
    'Russian': 'ru',
    'Turkish': 'tr',
    'French': 'fr',
    'Portuguese': 'pt',  # Corregido de 'Portugese' a 'Portuguese'
    'Romanian': 'ro',
    'Swedish': 'sv'
}

supported_languages = [
    "ast", "bg", "ca", "cs", "cy", "da", "de", "el", "en", "enm", "es", "et", "fa",
    "fi", "fr", "ga", "gd", "gl", "gv", "hbs", "hi", "hu", "hy", "id", "is", "it",
    "ka", "la", "lb", "lt", "lv", "mk", "ms", "nb", "nl", "nn", "pl", "pt", "ro",
    "ru", "se", "sk", "sl", "sq", "sv", "sw", "tl", "tr", "uk"
]


# Función para dividir las oraciones en frases
def split_into_sentences(text):
    # Dividir por '.', '?', '!', pero asegurándose de no dividir abreviaturas o números.
    # Puedes ajustar la expresión regular según tus necesidades específicas.
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', text)
    return [sentence.strip() for sentence in sentences if sentence.strip() != '']

# Función para lematizar una frase basada en su etiqueta de idioma
def lemmatize_sentence(sentence, label):
    lang_code = language_codes.get(label)
    # Verificar si el idioma y su código están soportados
    if lang_code in supported_languages:
        # Dividir la frase en tokens
        tokens = sentence.split()
        # Lematizar cada token
        return ' '.join([simplemma.lemmatize(token, lang=lang_code) for token in tokens])
    else:
        # Devolver la frase original si el idioma no está soportado
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
    # Place your code here
    # Keep in mind that sentence splitting affectes the number of sentences
    # and therefore, you should replicate labels to match.
    preprocessed_sentences = []
    preprocessed_labels = []    
    
    # lematizador
    lemmatizer = WordNetLemmatizer()

    print(sentences)
    print(labels)
    for sentence, label in zip(sentences, labels):
        
        
        tokens = word_tokenize(sentence)

        # stop_words = set(stopwords.words(label))

        # Convertir a minúsculas y lematización
        tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalpha()]

        # Agregar los tokens preprocesados y el label correspondiente
        preprocessed_sentences.append(tokens)
        preprocessed_labels.append(label)
    return preprocessed_sentences, preprocessed_labels


#Tokenizer function. You can add here different preprocesses.
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
    
    # Aplicar la función para dividir cada 'sentence' en una lista de frases más pequeñas
    df['sentences'] = df['sentences'].apply(split_into_sentences)
    
    # Explode la columna 'sentences' para que cada frase esté en su propia fila, duplicando la etiqueta correspondiente
    df_exploded = df.explode('sentences').reset_index(drop=True)
    
    # SE QUEDA ESTANCAU AQUI
    # Lemmatize sentences according to the language in the label (if  the language is in the lemmatizer)
    df_exploded['sentences'] = df_exploded.apply(lambda row: lemmatize_sentence(row['sentences'], row['labels']), axis=1)

    new_sentences_series = df_exploded['sentences']
    new_labels_series = df_exploded['labels']
    
    print(new_sentences_series.head())
    print(new_labels_series.head())

    return new_sentences_series, new_labels_series

# #Tokenizer function. You can add here different preprocesses.
# def preprocess(sentences, labels):
#     return sentences, labels

# ("sent 1. sentx  asdasdasd","sent 2 asdasdasd")
# ("sent 1.") => [sent,1,.]
# " ".join()

# pd.series()



