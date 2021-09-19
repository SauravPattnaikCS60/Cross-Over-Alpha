import joblib
import os
import json
from datetime import datetime
import keras


def save_files(file, filename):

    output_dir = os.path.join(os.getcwd(), 'SavedFiles')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_path = os.path.join(os.getcwd(), 'SavedFiles', str(filename + ".pkl"))
    joblib.dump(file, output_path)


def read_pickles(filename):
    path = os.path.join(os.getcwd(), 'SavedFiles', str(filename))
    file = joblib.load(path)
    return file


def save_lstm(model):
    config_filename = 'config.json'
    config = json.load(open(os.path.join(os.getcwd(), 'Data Files', str(config_filename))))
    filename1 = config['filename1'].split(".")[0]
    filename2 = config['filename2'].split(".")[0]
    time_stamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
    filename = str(filename1 + "_" + filename2 + "_model_" + time_stamp)
    path = os.path.join(os.getcwd(), 'SavedFiles', str(filename))
    keras.models.save_model(model, path, save_format='h5')
    print("Model Saved Successfully")


def read_lstm(model_name):
    path = os.path.join(os.getcwd(), 'SavedFiles', str(model_name))
    model = keras.models.load_model(path)
    print("Model Read Successfully")
    return model
