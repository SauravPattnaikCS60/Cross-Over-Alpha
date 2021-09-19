from transformers import TFGPT2LMHeadModel
from transformers import GPT2Tokenizer
import random
from preprocessing import *
import subprocess
import os


# TODO : training & predicting with GPT2
def gpt2_pipeline_function(train_data, config,starting_text=None):
    # getting train data to a text format from list
    if type(train_data) == list:
        train_data = " ".join(train_data).strip()

    train_data = train_data.lower()
    train_data = custom_preprocessing(train_data)

    # saving the train data to a txt file
    with open("./Data Files/train_data.txt", "w") as output:
        output.write(str(train_data))

    output_dir_value = "output_crossover"
    model_type_value = "gpt2"
    model_name_value = "gpt2"
    train_data_file_value = '"./Data Files/train_data.txt"'

    num_train_epochs_value = config['num_train_epochs_value']
    per_gpu_train_batch_size_value = config['per_gpu_train_batch_size_value']
    max_length_words = config['max_length_words']
    temperature = config['temperature']
    skip_gpt2_train = config['skip_gpt2_train']

    if not skip_gpt2_train:
        fine_tune_path = os.path.join(os.getcwd(),'run_lm_finetuning.py')

        gpt2_command = f"python {str(fine_tune_path)} " \
                       f"--output_dir={output_dir_value} --model_type={model_type_value} " \
                       f"--model_name_or_path={model_name_value} " \
                       f"--do_train --train_data_file={train_data_file_value} " \
                       f"--num_train_epochs={num_train_epochs_value} " \
                       f"--per_gpu_train_batch_size={per_gpu_train_batch_size_value}"

        subprocess.run(gpt2_command, shell=True)

    # Getting the finetuned model
    model = TFGPT2LMHeadModel.from_pretrained("output_crossover", from_pt=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # getting the starting text of prediction
    if starting_text == None:
        random_index = random.randint(0, len(train_data))
        starting_text = train_data[random_index:random_index + 50]  # take any random sentence from the train data
    input_ids = tokenizer.encode(starting_text, return_tensors='tf')

    # generating the predictions
    generated_text_samples = model.generate(
        input_ids, max_length=max_length_words,
        num_return_sequences=5,
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,
        top_p=0.92,
        temperature=temperature,
        do_sample=True,
        top_k=125,
        early_stopping=True)

    results = []
    for i, beam in enumerate(generated_text_samples):
        results.append(tokenizer.decode(beam, skip_special_tokens=True))

    return model, results,starting_text
