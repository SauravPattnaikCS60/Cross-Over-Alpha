'''
authors: saurav.pattnaik & srishti.verma :)
'''


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import os
from nltk.tokenize import TreebankWordTokenizer,sent_tokenize
from nltk.corpus import stopwords
import re

tokenizer = TreebankWordTokenizer()

def list_to_string(sentence_list):
    text = ''
    for row in sentence_list:
        if text == '':
            text += row+'\n'
        else:
            text += ' ' + row+'\n'
    return text


def string_to_list(sentence_string):
    sentences = sent_tokenize(sentence_string)
    return sentences


def create_index():
    max_words = 10000


    embeddings_index = {}
    f = open(os.path.join(os.getcwd(),'Embeddings','glove.6B.50d.txt'),'r',encoding='utf-8')

    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:],dtype='float32')
        embeddings_index[word] = coefs

    f.close()

    return embeddings_index


def find_most_similar(input_sentence_embedding, target_sentence_embeddings, ignore_list):
    similarity = [cosine_similarity([input_sentence_embedding], [target_sentence])[0][0] for target_sentence in
                  target_sentence_embeddings]
    if len(ignore_list) > 0:
        for i in ignore_list:
            similarity[i] = -1
    most_similar_index = np.argmax(similarity)
    return most_similar_index, similarity[most_similar_index]


def prepare_freq_table(corpus1, corpus2):
    counter = Counter()
    corpus1 = list_to_string(corpus1)
    corpus2 = list_to_string(corpus2)
    tokens1 = [word for word in tokenizer.tokenize(corpus1)]
    tokens2 = [word for word in tokenizer.tokenize(corpus2)]

    counter.update(tokens1)
    counter.update(tokens2)
    return dict(counter)


def prepare_sentence_embeddings(corpus, index, freqs, dimensions=50, a=0.001):
    total_freq = sum(freqs.values())
    embeddings = []
    stopwords_list = stopwords.words('english')

    for sentence in corpus:
        sentence = sentence.lower()
        sentence = re.sub(r"[^A-Za-z]", " ", sentence)
        sentence = re.sub(r"(\s)(\s)+", " ", sentence)
        tokens = list(
            set([word for word in tokenizer.tokenize(sentence) if word not in stopwords_list and word in index.keys()]))
        weights = [a / (a + freqs.get(token, 0) / total_freq) for token in tokens]
        if len(tokens) == 0:
            embeddings.append(np.zeros((dimensions,)))
        else:
            embedding = np.average([index[token] for token in tokens], axis=0, weights=weights)
            embeddings.append(embedding)

    return embeddings


# THIS FUNCTION WILL BE CALLED
def similarity_module(corpus1, corpus2):
    index = create_index()
    freq_dict = prepare_freq_table(corpus1, corpus2)
    corpus_1_embeddings = prepare_sentence_embeddings(corpus1, index, freq_dict)
    corpus_2_embeddings = prepare_sentence_embeddings(corpus2, index, freq_dict)
    similarity_df = pd.DataFrame(columns=['Source', 'Target', 'Similarity_Value'])
    ignore_list = []
    for i, e1 in enumerate(corpus_1_embeddings):
        index, value = find_most_similar(e1, corpus_2_embeddings, ignore_list)
        ignore_list.append(index)
        similarity_df.loc[i, 'Source'] = i
        similarity_df.loc[i, 'Target'] = index
        similarity_df.loc[i, 'Similarity_Value'] = value

    return similarity_df