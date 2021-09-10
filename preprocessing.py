'''
authors: saurav.pattnaik & srishti.verma :)
'''

import pandas as pd
import numpy as np
import texthero as hero
import re


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
    text = re.sub(r"[^A-Za-z0-9,!.']", " ", text)
    text = re.sub(r"(\s)(\s)+", ' ', text)
    return text
