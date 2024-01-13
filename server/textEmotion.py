import pickle
import re

import nltk
import numpy as np
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('wordnet')
# load the model to disk
def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model


model = load_model('text_emotion.pkl')
stop_words = set(stopwords.words("english"))


def lemmatization(text):
    lemmatizer = WordNetLemmatizer()

    text = text.split()

    text = [lemmatizer.lemmatize(y) for y in text]

    return " ".join(text)


def remove_stop_words(text):
    Text = [i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)


def Removing_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])
    return text


def lower_case(text):
    text = text.split()

    text = [y.lower() for y in text]

    return " ".join(text)


def Removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛', "", )

    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text = " ".join(text.split())
    return text.strip()


def Removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan


def normalize_text(df):
    df.Text = df.Text.apply(lambda text: lower_case(text))
    df.Text = df.Text.apply(lambda text: remove_stop_words(text))
    df.Text = df.Text.apply(lambda text: Removing_numbers(text))
    df.Text = df.Text.apply(lambda text: Removing_punctuations(text))
    df.Text = df.Text.apply(lambda text: Removing_urls(text))
    df.Text = df.Text.apply(lambda text: lemmatization(text))
    return df


def normalized_sentence(sentence):
    sentence = lower_case(sentence)
    sentence = remove_stop_words(sentence)
    sentence = Removing_numbers(sentence)
    sentence = Removing_punctuations(sentence)
    sentence = Removing_urls(sentence)
    sentence = lemmatization(sentence)
    return sentence


def predict_text_emotion(text):
    data = {'Text': [text]}
    df = pd.DataFrame(data=data)
    df = normalize_text(df)
    text = df['Text'].values
    y_pred = model.predict(text)
    return y_pred[0]