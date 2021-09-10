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
from LSTM_pipeline import lstm_pipeline_function
from GPT2_pipeline import gpt2_pipeline_function
import json


config_filename = 'config.json'
config = json.load(open(os.path.join(os.getcwd(),'Data Files',str(config_filename))))


filename1 = config['filename1']
filename2 = config['filename2']
threshold = config['no_of_sentences']
similarity_threshold = config['similarity_threshold']
starting_text = config['starting_text']
if starting_text == "":
    starting_text = None

path1 = os.path.join(os.getcwd(),'Data Files',str(filename1))
path2 = os.path.join(os.getcwd(),'Data Files',str(filename2))




############# READING ######################
corpus1 = None
corpus2 = None

try:
    corpus1,corpus2 = read_pdfs(path1,path2,threshold)
    print('Reading inputs done successfully')

except Exception as e:
    print(f'{e}: Could not read files')
    exit()


########### PREPROCESSING ##################
try:
    corpus1 = pd.Series(corpus1).apply(custom_preprocessing).tolist()
    corpus2 = pd.Series(corpus2).apply(custom_preprocessing).tolist()

    print('Preprocessing module ran successfully')

except Exception as e:
    print(f'{e}: Preprocessing module failed')
    exit()


########### SIMILARITY MODULE ##################
try:
    ss_df = similarity_module(corpus1, corpus2)
    print('Similarity module ran successfully')

except Exception as e:
    print(f'{e}: Similarity module failed')
    exit()

######### CREATE NER SWAPS ####################
try:
    final_data, ner_samples, count_swaps = create_data_plus_ner_swap(corpus1, corpus2, ss_df, 'Source', 'Target',
                                                                     'Similarity_Value',similarity_threshold)
    print(len(final_data),len(ner_samples),count_swaps)
    print('NER SWAPS done successfully')

except Exception as e:
    print(f'{e}: NER SWAPS failed')
    exit()


######### TO DO : SAVE FINAL DATA ####################
final_data = list_to_string(final_data)
# with open("Data Files/TrainData.txt", "w") as output:
#     output.write(str(final_data))

######### TO DO : TRAINING & SAVING ####################
TRAIN_WITH_GPT2 = config['TRAIN_WITH_GPT2'] # should be an input

if TRAIN_WITH_GPT2 == False :
    try:
        trained_model,result,seed = lstm_pipeline_function(final_data,config,starting_text)
        model_name = "LSTM"
        print(f'Training with LSTM module ran successfully')

    except Exception as e:
        print(f'{e}: Training with LSTM module failed')
        exit()

else :
    try :
        trained_model,result,seed = gpt2_pipeline_function(final_data,config,starting_text)
        model_name = "GPT2"
        print(f'Training with GPT2 module ran successfully')

    except Exception as e:
        print(f'{e}: Training with GPT2 module failed')
        exit()

output_dir = os.path.join(os.getcwd(),'Results')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_file_name = str(filename1[:-4])+'_'+str(filename2[:-4])+'_'+'CrossOvers.txt'
output_path = os.path.join(output_dir,output_file_name)

if TRAIN_WITH_GPT2:
    delimeter = '\n' * 5
    formatted_result = delimeter.join([s for s in result])
else:
    formatted_result = str(result)

with open(output_path,'w') as f:
    f.writelines(f"Text generation with seed :: {seed}\n\n")
    f.write(formatted_result)
f.close()
