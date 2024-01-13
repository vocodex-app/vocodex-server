import nltk
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

def get_stem(keywords):
    ps = PorterStemmer()
    stem_keywords = set()

    for word in keywords:
        word = ps.stem(word)
        stem_keywords.add(word)
    # print(stem_keywords)
    return stem_keywords
