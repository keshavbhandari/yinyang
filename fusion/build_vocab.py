import yaml
import os
import argparse
import pickle
import glob
import numpy as np
import json
from aria.data.midi import MidiDict
from aria.tokenizer import AbsTokenizer
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=os.path.normpath("/homes/kb658/yinyang/configs/configs_os.yaml"),
                    help="Path to the config file")
args = parser.parse_args()

# Load config file
with open(args.config, 'r') as f:
    configs = yaml.safe_load(f)

artifact_folder = configs["raw_data"]["artifact_folder"]
mono_folder = configs["raw_data"]["mono_folder"]
json_folder = configs["raw_data"]["json_folder"]
raw_data_folders = configs["raw_data"]["raw_data_folders"]
data_path = raw_data_folders['classical']['folder_path']

# Build the vocabulary
vocab = {}
# MIDI velocity range from 0 to 127
velocity = [0, 15, 30, 45, 60, 75, 90, 105, 120, 127]
# MIDI pitch range from 0 to 127
midi_pitch = list(range(0, 128))
# Onsets are quantized in 10 milliseconds up to 5 seconds
onset = list(range(0, 5001, 10))
duration = list(range(0, 5001, 10))

# Add the tokens to the vocabulary
for v in velocity:
    for p in midi_pitch:
        vocab[("piano", p, v)] = len(vocab) + 1
for o in onset:
    vocab[("onset", o)] = len(vocab) + 1
for d in duration:
    vocab[("dur", d)] = len(vocab) + 1

# Special tokens
vocab[('prefix', 'instrument', 'piano')] = len(vocab) + 1
vocab["<T>"] = len(vocab) + 1
vocab["<D>"] = len(vocab) + 1
vocab["<S>"] = len(vocab) + 1
vocab["<E>"] = len(vocab) + 1
vocab["SEP"] = len(vocab) + 1

# Print the vocabulary length
print(f"Vocabulary length: {len(vocab)}")

# Save the vocabulary
# Directory path
fusion_folder = os.path.join(artifact_folder, "fusion")
# Make directory if it doesn't exist
os.makedirs(fusion_folder, exist_ok=True)
vocab_path = os.path.join(fusion_folder, "vocab.json")
with open(os.path.join(fusion_folder, "vocab.pkl"), 'wb') as f:
    pickle.dump(vocab, f)



# Get all the midi file names in the data path recursively
file_list = glob.glob(data_path + '/**/*.midi', recursive=True)
print(f"Number of MIDI files: {len(file_list)}")

# Shuffle the file list
np.random.shuffle(file_list)

# Split the data into train and validation sets
train_data = file_list[:int(0.9 * len(file_list))]
val_data = file_list[int(0.9 * len(file_list)):]
print(f"Number of training files: {len(train_data)}")
print(f"Number of validation files: {len(val_data)}")

# Save the train and validation file lists
with open(os.path.join(fusion_folder, "train_file_list.json"), "w") as f:
    json.dump(train_data, f)

with open(os.path.join(fusion_folder, "valid_file_list.json"), "w") as f:
    json.dump(val_data, f)



def store_files_as_json(file_list, json_folder, file_name="train"):
    aria_tokenizer = AbsTokenizer()
    list_of_tokens = []
    
    for idx, file_path in tqdm(enumerate(file_list), total=len(file_list), desc="Processing files"):
        mid = MidiDict.from_midi(file_path)
        tokenized_sequence = aria_tokenizer.tokenize(mid)
        list_of_tokens.append(tokenized_sequence)
    
    # Save the list_of_tokens as a json file
    json_file_path = os.path.join(json_folder, f"{file_name}.json")
    with open(json_file_path, "w") as f:
        json.dump(list_of_tokens, f)

# Store the training and validation files as json
store_files_as_json(train_data, fusion_folder, file_name="train")
store_files_as_json(val_data, fusion_folder, file_name="valid")