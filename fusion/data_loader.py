import yaml
import jsonlines
import glob
import random
import os
import sys
import pickle
import argparse
from torch.utils.data import Dataset
import torch
from torch.nn import functional as F
from aria.data.midi import MidiDict
from aria.tokenizer import AbsTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.utils import flatten, skyline


class DataLoader(Dataset):
    def __init__(self, configs, file_list, mode="train", shuffle = False):
        self.mode = mode
        # Data dir
        self.data_dir = configs['raw_data']['json_folder']
        self.file_list = file_list
        if shuffle:
            random.shuffle(self.file_list)

        raw_data_folders = configs["raw_data"]["raw_data_folders"]
        data_path = raw_data_folders['classical']['folder_path']
        # self.file_paths = []
        # for file in self.file_list:
        #     self.file_paths.append(os.path.join(data_path, file))

        # Artifact folder
        self.artifact_folder = configs['raw_data']['artifact_folder']
        # Load encoder tokenizer json file dictionary
        tokenizer_filepath = os.path.join(self.artifact_folder, "fusion", "vocab.pkl")

        self.aria_tokenizer = AbsTokenizer()
        # Load the pickled tokenizer dictionary
        with open(tokenizer_filepath, 'rb') as f:
            self.tokenizer = pickle.load(f)

        # Get the maximum sequence length
        self.encoder_max_sequence_length = configs['model']['phrase_generation_model']['encoder_max_sequence_length']
        self.decoder_max_sequence_length = configs['model']['phrase_generation_model']['decoder_max_sequence_length']

        # Print length of dataset
        print("Length of dataset: ", len(self.file_list))

    def __len__(self):
        return len(self.file_list)
    
    def transpose(self, encoding, pitch_change):
        pass

    def __getitem__(self, idx):
        file_path = self.file_list[idx]

        mid = MidiDict.from_midi(file_path)
        tokenized_sequence = self.aria_tokenizer.tokenize(mid)

        # Take the 3rd token as the start token until the 2nd last token
        tokenized_sequence = tokenized_sequence[2:] + tokenized_sequence[:-1]

        # Get random crop of the sequence of length max_sequence_length
        piano_token_indices = [i for i in range(len(tokenized_sequence)) if tokenized_sequence[i][0] == "piano"]
        # Exclude the last token of piano_token_indices to generate at least one token
        piano_token_indices = piano_token_indices[:-1]

        if len(piano_token_indices) > 0:
            # Choose the start index randomly
            start_idx = random.choice(piano_token_indices)
        else:
            print("No piano tokens found in the sequence for file", file_path)
            assert len(piano_token_indices) > 0
        # Crop the sequence
        end_idx = start_idx + self.encoder_max_sequence_length - 2
        tokenized_sequence = tokenized_sequence[start_idx:end_idx]

        # Call the flatten function
        flattened_sequence = flatten(tokenized_sequence)
        # Call the skyline function
        melody, harmony = skyline(flattened_sequence, diff_threshold=30)
        # Add the start and end tokens
        tokenized_sequence = ["<S>"] + tokenized_sequence + ["<E>"]

        # Tokenize the melody and harmony sequences
        melody = [self.tokenizer[token] for token in melody]
        harmony = [self.tokenizer[token] for token in harmony]
        tokenized_sequence = [self.tokenizer[token] for token in tokenized_sequence]

        # Pad the sequences
        if len(tokenized_sequence) < self.encoder_max_sequence_length:
            tokenized_sequence = F.pad(torch.tensor(tokenized_sequence), (0, self.encoder_max_sequence_length - len(tokenized_sequence)))
        else:
            tokenized_sequence = torch.tensor(tokenized_sequence[:self.encoder_max_sequence_length])
        if len(melody) < self.encoder_max_sequence_length:
            melody = F.pad(torch.tensor(melody), (0, self.encoder_max_sequence_length - len(melody)))
        else:
            melody = torch.tensor(melody[:self.encoder_max_sequence_length])
        if len(harmony) < self.encoder_max_sequence_length:
            harmony = F.pad(torch.tensor(harmony), (0, self.encoder_max_sequence_length - len(harmony)))
        else:
            harmony = torch.tensor(harmony[:self.encoder_max_sequence_length])

        # Attention mask based on non-padded tokens of the phrase
        attention_mask = torch.where(tokenized_sequence != 0, 1, 0).type(torch.bool)

        train_data = {"input_ids": melody, "labels": tokenized_sequence, "attention_mask": attention_mask}

        return train_data





if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.normpath("configs/configs_os.yaml"),
                        help="Path to the config file")
    args = parser.parse_args()

    # Load config file
    with open(args.config, 'r') as f:
        configs = yaml.safe_load(f)
        
    artifact_folder = configs["raw_data"]["artifact_folder"]
    raw_data_folders = configs["raw_data"]["raw_data_folders"]
    data_path = raw_data_folders['classical']['folder_path']

    # Get all the midi file names in the data path recursively
    file_list = glob.glob(data_path + '/**/*.midi', recursive=True)

    # file_list = ["/import/c4dm-datasets/maestro-v3.0.0/2009/MIDI-Unprocessed_10_R1_2009_03-05_ORIG_MID--AUDIO_10_R1_2009_10_R1_2009_04_WAV.midi"]
    
    # Call the DataLoader class
    data_loader = DataLoader(configs, file_list, mode="train", shuffle=True)
    # Get the first item
    for n, data in enumerate(data_loader):
        print(data["input_ids"], '\n')
        print(data["labels"], '\n')
        print(data["input_ids"].shape)
        print(data["labels"].shape)
        print(data["attention_mask"].shape)
        if n == 10:
            break


    # # Get all the midi files in the data path recursively
    # midi_files = glob.glob(data_path + '/**/*.midi', recursive=True)


    # tokenizer = AbsTokenizer()
    # midi_file_path = random.choice(midi_files)
    # midi_file_path = "/import/c4dm-datasets/maestro-v3.0.0/2008/MIDI-Unprocessed_07_R2_2008_01-05_ORIG_MID--AUDIO_07_R2_2008_wav--2.midi"
    # print(midi_file_path)
    # # midi_file_path = midi_files[0]
    # mid = MidiDict.from_midi(midi_file_path)
    # tokenized_sequence = tokenizer.tokenize(mid)

    # # Call the flatten function
    # flattened_sequence = flatten(tokenized_sequence)
    # print(flattened_sequence[0:100], '\n')

    # # Call the skyline function
    # melody, harmony = skyline(flattened_sequence, diff_threshold=30)
    # melody = tokenized_sequence[0:2] + melody + tokenized_sequence[-1:]
    # harmony = tokenized_sequence[0:2] + harmony + tokenized_sequence[-1:]
    # print(melody[0:100], '\n')
    # print(harmony[0:100], '\n')

    # print(len(tokenized_sequence))
    # print(len(melody))
    # print(len(harmony))

    # mid_dict = tokenizer.detokenize(melody) # mid_dict is a MidiDict object
    # mid = mid_dict.to_midi() # mid is a mido.MidiFile
    # mid.save('test_file_melody.mid')

    # mid_dict = tokenizer.detokenize(harmony) # mid_dict is a MidiDict object
    # mid = mid_dict.to_midi() # mid is a mido.MidiFile
    # mid.save('test_file_harmony.mid')