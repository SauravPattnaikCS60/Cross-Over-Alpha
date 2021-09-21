import joblib
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,GRU
import random
from preprocessing import *
from utils import *

# TODO : training & predicting with GRU
def gru_pipeline_function(train_data,config,starting_text = None) :

    def split_train_data(data, maxlen, step):
    # This function will create the x (sentence) and y (next char) from the data
        sentences = []
        next_chars = []
        for i in range(0, len(data) - maxlen, step):
            sentences.append(data[i:i + maxlen])
            next_chars.append(data[i + maxlen])

        chars = sorted(list(set(data)))  # Consists of all unique characters in the file
        char_to_index = dict((char, chars.index(char)) for char in chars)  # character to index mapping

        print(f'Number of rows = {len(sentences)}')
        print(f'Number of unique characters = {len(chars)}')

        x = np.zeros((len(sentences),maxlen,len(chars)))
        y = np.zeros((len(sentences),len(chars)))

        for s, sentence in enumerate(sentences):
            for c, char in enumerate(sentence):
                x[s, c, char_to_index[char]] = 1  # mark the char as 1 in x
            y[s, char_to_index[next_chars[s]]] = 1  # mark the char as 1 in y

        return x, y, len(chars), chars, char_to_index, sentences


    def build_model(len_chars, maxlen, n_neurons=256):
    # This function will create a model & return it
        model = Sequential()

        model.add(GRU(n_neurons, input_shape=(maxlen, len_chars), recurrent_dropout=0.5))
        model.add(Dropout(0.25))
        model.add(Dense(len_chars, activation='softmax'))

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        return model
    
    def sample(preds,temperature=0.4):
    # This function makes it possible to make predictions a little random by doing sampling based on temperature.
        preds = np.asarray(preds).astype('float64')

        preds = np.log(preds)/temperature # Here temperature acts like a amplifying factor. Lower value will amplify the
                                          # most probable character and it will come again and again
        exp_preds = np.exp(preds)

        exp_preds = exp_preds/np.sum(exp_preds)

        probas = np.random.multinomial(1,exp_preds,1)

        return np.argmax(probas)
    
    
    # getting train data to a text format from list
    if type(train_data) == list:
        train_data = " ".join(train_data).strip()

    train_data = train_data.lower() 
    train_data=custom_preprocessing(train_data)

    maxlen = config['max_sequence_length_gru']
    step = config['step_size_gru']
    n_neurons = config['n_neurons_gru']
    output_length = config['max_len_chars']
    temperature = config['temperature']
    n_epoch = config["num_train_epochs_value"]
    model_path=config["model_name"]

    # splitting the train data
    x, y, len_chars, chars, char_to_index, sentences = split_train_data(train_data, maxlen, step)

    if model_path == "":
        # building & fitting the model
        model = build_model(len_chars, maxlen, n_neurons)
        print("epochs = ", n_epoch)
        model.fit(x, y, batch_size=128, epochs=n_epoch, validation_split=.2, shuffle=False, verbose=-100)
        # saving the trained model
        save_gru(model)
    else:
        model = read_gru(model_path)

    # getting the starting text for the prediction
    if starting_text == None :
        random_index = random.randint(0,len(train_data))
        starting_text = train_data[random_index:random_index+50] # take any random sentence from the train data 
    result = starting_text
    generate = starting_text + ' '*(maxlen - len(starting_text)) # To make it as the same length
    generate = generate.lower()
    
    # getting the predictions on character level
    for i in range(output_length):
        sampled = np.zeros((1,maxlen,len_chars))

        for t,char in enumerate(generate):
            sampled[0,t,char_to_index[char]] = 1

        preds = model.predict(sampled)[0] 
        next_char = chars[sample(preds,temperature)]
        generate += next_char
        result += next_char
        generate = generate[1:]
    return model, result,starting_text