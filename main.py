'''
authors: saurav.pattnaik & srishti.verma

File : This is the main file that will be called to run the entire pipeline.

Data Files : Inputs to be kept in this folder.
Embeddings : Glove file present in this directory.
Total modules identified till now:
    1. Read Data
    2. Preprocessing
    3. Similarity
    4. NER SWAPS

'''
import os
from read_data import read_pdfs
from preprocessing import *
from similarity_module import similarity_module, list_to_string
from ner_swaps import create_data_plus_ner_swap
from GRU_pipeline import gru_pipeline_function
from GPT2_pipeline import gpt2_pipeline_function
import json
from utils import *

config_filename = 'config.json'
config = json.load(open(os.path.join(os.getcwd(), 'Data Files', str(config_filename))))

filename1 = config['filename1']
filename2 = config['filename2']
threshold = config['no_of_sentences']
similarity_threshold = config['similarity_threshold']
starting_text = config['starting_text']
if starting_text == "":
    starting_text = None
preprocessed_file_paths = config['preprocessed_file_names']
similarity_df_name = config['similarity_df_name']
final_data_name = config['train_data_name']
TRAIN_WITH_GPT2 = config['TRAIN_WITH_GPT2']
final_data = None
result = None

############# Initializing ######################
corpus1 = None
corpus2 = None

########## READING & PREPROCESSING ##################
if preprocessed_file_paths == "":

    try:
        corpus1, corpus2 = read_pdfs(filename1, filename2, threshold)
        print('Reading inputs done successfully')

    except Exception as e:
        print(f'{e}: Could not read files')
        exit()

    try:
        if TRAIN_WITH_GPT2 == False:
            corpus1 = pd.Series(corpus1).apply(custom_preprocessing_gru).tolist()
            corpus2 = pd.Series(corpus2).apply(custom_preprocessing_gru).tolist()
        else :
            corpus1 = pd.Series(corpus1).apply(custom_preprocessing).tolist()
            corpus2 = pd.Series(corpus2).apply(custom_preprocessing).tolist()
        print('Preprocessing module ran successfully')
        try:
            save_files(corpus1, filename1.split(".")[0] + "_afterPreprocessing")
            save_files(corpus2, filename2.split(".")[0] + "_afterPreprocessing")
            print("Preprocessed files saved")
        except Exception as e:
            print(f'{e} : Preprocessed files not saved')
    except Exception as e:
        print(f'{e}: Preprocessing module failed')
        exit()
else:
    try:
        corpus1 = read_pickles(preprocessed_file_paths.split(",")[0])
        corpus2 = read_pickles(preprocessed_file_paths.split(",")[1])
        print("Preprocessed files read")
    except Exception as e:
        print(f"{e}: Couldn't read preprocessed files")
        exit()

########### SIMILARITY MODULE ##################
if similarity_df_name == "":
    try:
        ss_df = similarity_module(corpus1, corpus2)
        print('Similarity module ran successfully')
        try:
            save_files(ss_df, filename1.split(".")[0] + "_" + filename2.split(".")[0] + "_similarity_df")
            print("Similarity Dataframe saved")
        except Exception as e:
            print(f"{e}: Similarity module failed")
    except Exception as e:
        print(f'{e}: Similarity module failed')
        exit()
else:
    try:
        ss_df = read_pickles(similarity_df_name)
        print('Similarity Dataframe read')
    except Exception as e:
        print(f"{e}: Couldn't read similarity Dataframe")
        exit()

######### CREATE NER SWAPS ####################
if final_data_name == "":
    try:
        final_data, ner_samples, count_swaps = create_data_plus_ner_swap(corpus1, corpus2, ss_df, 'Source', 'Target',
                                                                         'Similarity_Value', similarity_threshold)
        print(len(final_data), len(ner_samples), count_swaps)
        print('NER SWAPS done successfully')
        try:
            save_files(final_data, filename1.split(".")[0] + "_" + filename2.split(".")[0] + "_train_data")
            print("Train Data saved")
        except Exception as e:
            print(f'{e}: Saving train data failed')
    except Exception as e:
        print(f'{e}: NER SWAPS failed')
        exit()
else:
    try:
        final_data = read_pickles(final_data_name)
        print("Train Data Read")
    except Exception as e:
        print(f"{e}: Couldn't read train data")
        exit()

######### TO DO : TRAINING & SAVING ####################

if TRAIN_WITH_GPT2 == False:
    try:
        trained_model, result, seed = gru_pipeline_function(final_data, config, starting_text)
        print(f'Training with GRU module ran successfully')

    except Exception as e:
        print(f'{e}: Training with GRU module failed')
        exit()

else:
    try:
        trained_model, result, seed = gpt2_pipeline_function(final_data, config, starting_text)
        print(f'Training with GPT2 module ran successfully')

    except Exception as e:
        print(f'{e}: Training with GPT2 module failed')
        exit()

output_dir = os.path.join(os.getcwd(), 'Results')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_file_name = str(filename1[:-4]) + '_' + str(filename2[:-4]) + '_' + 'CrossOvers.txt'
output_path = os.path.join(output_dir, output_file_name)

if TRAIN_WITH_GPT2:
    delimeter = '\n' * 5
    formatted_result = delimeter.join([s for s in result])
else:
    formatted_result = str(result)

with open(output_path, 'w',encoding="utf-8") as f:
    f.writelines(f"Text generation with seed :: {seed}\n\n")
    f.write(formatted_result)
f.close()
