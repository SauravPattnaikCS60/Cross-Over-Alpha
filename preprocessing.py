'''
authors: saurav.pattnaik & srishti.verma :)
'''

import pandas as pd
import numpy as np
import texthero as hero
import re
from num2words import num2words


def basic_preprocessing(series):
    # Removing brackets
    series = hero.remove_brackets(series)

    # Removing diacritics
    series = hero.remove_diacritics(series)  # Words like Café, that top extra char will be removed

    # Removing whitespaces
    series = hero.remove_whitespace(series)

    series = hero.remove_digits(series)

    return series

def custom_preprocessing(text):
    text = re.sub(r"[^A-Za-z0-9,.!']", " ", text)
    text = re.sub(r"(\s)(\s)+", ' ', text)
    return text


def custom_preprocessing_lstm(text):
    # convert 1 to one, etc.
    num2words_function = lambda y: num2words(y) if y.isnumeric() else y
    text_list = text.split(" ")
    text = " ".join(list(map(num2words_function, text_list))).strip()

    # cleaning apostrophes
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"^[A-Za-z]\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)

    text = re.sub(r" e g ", " example ", text)
    text = re.sub(r" eg ", " example ", text)
    text = re.sub(r" i.e ", " that is ", text)

    text = re.sub(r"\$", " dollar ", text)
    text = re.sub(r"\€", " euro ", text)

    text = re.sub(r"[^A-Za-z,.']", " ", text)
    text = re.sub(r"(\s)(\s)+", ' ', text)
    return text
