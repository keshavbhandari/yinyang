import yaml
import jsonlines
import glob
import random
import os
import sys
import pickle
import json
import argparse
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset
import torch
from torch.nn import functional as F
from aria.data.midi import MidiDict
from aria.tokenizer import AbsTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.utils import flatten, unflatten, skyline, chord_density_ratio


class Fusion_Dataset(Dataset):
    def __init__(self, configs, data_list, mode="train", shuffle = False):
        self.mode = mode
        self.data_list = data_list
        if shuffle:
            random.shuffle(self.data_list)

        # Artifact folder
        self.artifact_folder = configs['raw_data']['artifact_folder']
        # Load encoder tokenizer json file dictionary
        tokenizer_filepath = os.path.join(self.artifact_folder, "fusion", "vocab.pkl")

        self.aria_tokenizer = AbsTokenizer()
        # Load the pickled tokenizer dictionary
        with open(tokenizer_filepath, 'rb') as f:
            self.tokenizer = pickle.load(f)

        # Get the maximum sequence length
        self.encoder_max_sequence_length = configs['model']['fusion_model']['encoder_max_sequence_length']
        self.decoder_max_sequence_length = configs['model']['fusion_model']['decoder_max_sequence_length']

        # Print length of dataset
        print("Length of dataset: ", len(self.data_list))

    def __len__(self):
        return len(self.data_list)

    def get_melody_provided_sequences(self, target_sequence, melody_indices):
        
        # Call the flatten function
        flattened_sequence = flatten(target_sequence, add_special_tokens=True)
        flattened_sequence_no_sp_tokens = flatten(target_sequence, add_special_tokens=False)
        # Get random crop of the flattened sequence of length max_sequence_length
        if len(flattened_sequence) <= self.decoder_max_sequence_length:
            start_idx = 0
        else:
            # Choose the start index randomly between 0 and the length of the sequence minus the max_sequence_length divided by 2 so there is always something to generate
            start_idx = random.randint(0, len(flattened_sequence) - self.decoder_max_sequence_length/2)

        end_idx = start_idx + self.decoder_max_sequence_length - 2
        target_sequence = flattened_sequence[start_idx:end_idx]
        if start_idx == 0:
            context_sequence = []
        else:
            context_sequence = flattened_sequence[0:start_idx]
        
        melody = [value for i, value in enumerate(flattened_sequence_no_sp_tokens) if int(start_idx) <= i < int(end_idx) and i in melody_indices]

        # Get chord density ratio
        chord_density = chord_density_ratio(target_sequence)

        # Unflatten the sequences
        melody = unflatten(melody, static_velocity=True) 
        context_sequence = unflatten(context_sequence, static_velocity=False)
        target_sequence = unflatten(target_sequence, static_velocity=False)

        return target_sequence, melody, context_sequence, chord_density

    def get_melody_extracted_sequences(self, target_sequence):
        # Take the 3rd token as the start token until the 2nd last token
        target_sequence = target_sequence[2:-1]

        # Get random crop of the sequence of length max_sequence_length
        piano_token_indices = [i for i in range(len(target_sequence)) if target_sequence[i][0] == "piano"]
        # Exclude the last token of piano_token_indices to generate at least one token
        piano_token_indices = piano_token_indices[:-1]

        if len(piano_token_indices) > 0:
            # Choose the start index randomly
            start_idx = random.choice(piano_token_indices)
        else:
            print("No piano tokens found in the sequence for file")
            assert len(piano_token_indices) > 0

        # Crop the sequence
        end_idx = start_idx + self.decoder_max_sequence_length - 2
        target_sequence = target_sequence[start_idx:end_idx]
        context_sequence = target_sequence[0:start_idx]

        # Call the flatten function
        flattened_sequence = flatten(target_sequence)
        # Call the skyline function
        melody, harmony = skyline(flattened_sequence, diff_threshold=30, static_velocity=True)

        # Get chord density ratio
        chord_density = chord_density_ratio(flattened_sequence)

        return target_sequence, melody, context_sequence, chord_density
        

    def __getitem__(self, idx):
        sequence_info = self.data_list[idx]
        genre = sequence_info[-1]
        tokenized_sequence = sequence_info[0]

        # Apply augmentations
        pitch_aug_function = self.aria_tokenizer.export_pitch_aug(5)
        tokenized_sequence = pitch_aug_function(tokenized_sequence)

        meta_tokens = [genre]

        if len(sequence_info) == 2:
            tokenized_sequence, melody, context, chord_density = self.get_melody_extracted_sequences(tokenized_sequence)
            if random.random() < 0.5:
                meta_tokens = meta_tokens + chord_density
        else:
            melody_indices = sequence_info[1]
            if random.random() < 0.5:
                tokenized_sequence = sequence_info[2]
                meta_tokens.append("no_bridge")
            tokenized_sequence, melody, context, chord_density = self.get_melody_provided_sequences(tokenized_sequence, melody_indices)
            if random.random() < 0.5:
                meta_tokens = meta_tokens + chord_density

        if random.random() < 0.8:
            # Add context sequence to the melody with a SEP token
            input_tokens = context + ["SEP"] + meta_tokens + ["SEP"] + melody
        else:
            # Don't add context sequence to the melody
            input_tokens = meta_tokens + ["SEP"] + melody

        # Add the start and end tokens
        tokenized_sequence = ["<S>"] + tokenized_sequence + ["<E>"]

        # Tokenize the melody and harmony sequences
        input_tokens = [self.tokenizer[tuple(token)] if isinstance(token, list) else self.tokenizer[token] for token in input_tokens]
        tokenized_sequence = [self.tokenizer[tuple(token)] if isinstance(token, list) else self.tokenizer[token] for token in tokenized_sequence]

        # Pad the sequences
        if len(tokenized_sequence) < self.decoder_max_sequence_length:
            tokenized_sequence = F.pad(torch.tensor(tokenized_sequence), (0, self.decoder_max_sequence_length - len(tokenized_sequence))).to(torch.int64)
        else:
            tokenized_sequence = torch.tensor(tokenized_sequence[-self.decoder_max_sequence_length:]).to(torch.int64)
        if len(input_tokens) < self.encoder_max_sequence_length:
            input_tokens = F.pad(torch.tensor(input_tokens), (0, self.encoder_max_sequence_length - len(input_tokens))).to(torch.int64)
        else:
            input_tokens = torch.tensor(input_tokens[-self.encoder_max_sequence_length:]).to(torch.int64)

        # Attention mask based on non-padded tokens of the phrase
        attention_mask = torch.where(input_tokens != 0, 1, 0).type(torch.bool)

        train_data = {"input_ids": input_tokens, "labels": tokenized_sequence, "attention_mask": attention_mask}

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

    # Open the train, validation, and test sets json files
    with open(os.path.join(artifact_folder, "fusion", "train.json"), "r") as f:
        train_sequences = json.load(f)
    # with open(os.path.join(artifact_folder, "fusion", "valid.json"), "r") as f:
    #     valid_sequences = json.load(f)
    
    # Call the Fusion_Dataset class
    data_loader = Fusion_Dataset(configs, train_sequences, mode="val", shuffle=True)
    # Get the first item
    for n, data in enumerate(data_loader):
        print(data["input_ids"], '\n')
        print(data["labels"], '\n')
        print(data["input_ids"].shape)
        print(data["labels"].shape)
        print(data["attention_mask"].shape)
        if n == 10:
            break