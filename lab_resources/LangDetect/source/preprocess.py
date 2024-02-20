from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

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

    for sentence, label in zip(sentences, labels):
        tokens = word_tokenize(sentence)

        stop_words = set(stopwords.words(labels))

        # Convertir a minúsculas y lematización
        tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalpha() and word.lower() not in stop_words]

        # Agregar los tokens preprocesados y el label correspondiente
        preprocessed_sentences.append(tokens)
        preprocessed_labels.append(label)

    return preprocessed_sentences, preprocessed_labels