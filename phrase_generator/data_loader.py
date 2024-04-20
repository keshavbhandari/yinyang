import os
import json
import random
import yaml
import copy
import argparse
from torch.utils.data import Dataset
import torch
from torch.nn import functional as F
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.utils import list_to_remi_encoding


class JSONDataset(Dataset):
    def __init__(self, configs, file_list, mode="train", shuffle = False):
        self.mode = mode
        # Data dir
        self.data_dir = configs['raw_data']['json_folder']
        self.file_list = file_list
        if shuffle:
            random.shuffle(self.file_list)
        # Get number of phrases in each file and store in list as [file_name, phrase_number_{n}]
        self.file_number_phrase_number = []
        for file_path in self.file_list:
            file_path = os.path.join(self.data_dir, file_path)
            with open(file_path, 'r') as f:
                data = json.load(f)
            phrase_number = len(data["phrases"].keys())
            # Exclude the last phrase as this will be target
            for i in range(phrase_number-1):
                self.file_number_phrase_number.append([file_path, i])

        # Artifact folder
        self.artifact_folder = configs['raw_data']['artifact_folder']
        # Load encoder tokenizer json file dictionary
        tokenizer_filepath = os.path.join(self.artifact_folder, "tokenizer.json")
        # Load the tokenizer dictionary
        with open(tokenizer_filepath, 'r') as f:
            self.tokenizer = json.load(f)

        # Get the maximum sequence length
        self.encoder_max_sequence_length = configs['model']['phrase_generation_model']['encoder_max_sequence_length']
        self.decoder_max_sequence_length = configs['model']['phrase_generation_model']['decoder_max_sequence_length']

        # Print length of dataset
        print("Length of dataset: ", len(self.file_list))
        print("Length of phrases in dataset: ", len(self.file_number_phrase_number))

    def __len__(self):
        return len(self.file_number_phrase_number)
    
    def transpose(self, phrase, pitch_change):
        encoding = copy.deepcopy(phrase)

        transposed_encoding = [
            [event[0], event[1], event[2], event[3] + pitch_change, *event[4:]]
            for event in encoding
        ]

        return transposed_encoding
    
    def transpose_key(self, current_key, semitones):
        keys = ["KS_A-", "KS_A", "KS_B-", "KS_B", "KS_C", "KS_D-", "KS_D", "KS_E-", "KS_E", "KS_F", "KS_F#", "KS_G"]
        
        # Find the index of the current key in the list
        current_index = keys.index(current_key)
        
        # Calculate the new index after transposing by the given semitones
        new_index = (current_index + semitones) % len(keys)
        
        # Return the new key
        return keys[new_index]

    def augment_phrase(self, phrase_1, target, current_key):
        if random.random() < 0.5:
            pitch_change = random.choice([i for i in range(-12,12) if i not in [0]])

            encoding = phrase_1 + target

            # Find highest and lowest pitch values
            pitch_values = [event[3] for event in encoding]
            highest_pitch = max(pitch_values)
            lowest_pitch = min(pitch_values)
            # Choose a random pitch change value but ensure it is not 0 and within the midi pitch range of 0 to 127
            pitch_change = random.choice([i for i in range(-12,12) if i not in [0]])
            while highest_pitch + pitch_change > 127 or lowest_pitch + pitch_change < 0:
                if pitch_change < 0:
                    pitch_change += 1
                else:
                    pitch_change -= 1

            phrase_1 = self.transpose(phrase_1, pitch_change)
            target = self.transpose(target, pitch_change)
            current_key = self.transpose_key(current_key, pitch_change)

        return phrase_1, target, current_key

    def __getitem__(self, idx):
        file_path = self.file_number_phrase_number[idx][0]
        phrase_number = self.file_number_phrase_number[idx][1]
        with open(file_path, 'r') as f:
            data = json.load(f)

        time_signature = data["metadata"]["time_signature"]
        key_signature = data["metadata"]["key_signature"]
        major_or_minor = data["metadata"]["major_or_minor"]
        # # Get the phrase and the target
        # # Get all the phrases before the target and concatenate them
        phrases = []
        for i in range(phrase_number+1):
            phrase = data["phrases"][str(i)][0]
            phrases += phrase

        # Extract an arbitrary phrase from random point till the last element of the list
        total_phrase_length = len(phrases)
        current_phrase_length = len(data["phrases"][str(phrase_number)][0])
        
        go_back = random.randint(current_phrase_length, total_phrase_length)
        phrase_1 = phrases[-go_back:]
        # phrase_1 = data["phrases"][str(phrase_number)][0]
        target = data["phrases"][str(phrase_number+1)][0]
        target_cadence = data["phrases"][str(phrase_number + 1)][2]
        target_pitch_range = data["phrases"][str(phrase_number + 1)][3]
        target_length = data["phrases"][str(phrase_number + 1)][4]

        # Augment the phrases
        if self.mode == "train":
            phrase_1, target, key_signature = self.augment_phrase(phrase_1, target, key_signature)

        tempo_location = data["metadata"]["tempo"]

        # List to remi encoding
        phrase = list_to_remi_encoding(phrase_1, tempo_location, time_signature)
        # Add the BOS and EOS tokens to the phrase
        if random.random() < 0.4:
            phrase = ["BOS"] + phrase + ["SEP"] + [target_pitch_range] + [major_or_minor] + [key_signature] + [target_length] + [target_cadence] + ["SEP"] + ["EOS"]
        elif random.random() < 0.55:
            phrase = ["BOS"] + phrase + ["SEP"] + [target_pitch_range] + [key_signature] + [target_length] + [target_cadence] + ["SEP"] + ["EOS"]
        elif random.random() < 0.7:
            phrase = ["BOS"] + phrase + ["SEP"] + [target_pitch_range] + [major_or_minor] + [target_length] + [target_cadence] + ["SEP"] + ["EOS"]
        elif random.random() < 0.85:
            phrase = ["BOS"] + phrase + ["SEP"] + [major_or_minor] + [key_signature] + [target_length] + [target_cadence] + ["SEP"] + ["EOS"]
        else:
            phrase = ["BOS"] + phrase + ["SEP"] + [target_cadence] + ["SEP"] + ["EOS"]
        
        # Tokenize the phrase
        phrase = [self.tokenizer[note] for note in phrase if note in self.tokenizer]

        # Add the BOS and EOS tokens to the target
        target = list_to_remi_encoding(target, tempo_location, time_signature)
        target = target + ["EOS"]
        # Tokenize the target
        target = [self.tokenizer[note] for note in target if note in self.tokenizer]

        # Convert to tensor and pad the phrase to a fixed length of max_sequence_length if the phrase is shorter than max_sequence_length
        phrase = torch.tensor(phrase)
        if len(phrase) < self.encoder_max_sequence_length:
            phrase = F.pad(phrase, (0, self.encoder_max_sequence_length - len(phrase)))
        else:
            phrase = phrase[-self.encoder_max_sequence_length:]
        # Attention mask based on non-padded tokens of the phrase
        phrase_attention_mask = torch.where(phrase != 0, 1, 0)
        phrase_attention_mask = phrase_attention_mask.type(torch.bool)

        # Do the same for the target
        target = torch.tensor(target)
        if len(target) < self.decoder_max_sequence_length:
            target = F.pad(target, (0, self.decoder_max_sequence_length - len(target)))
        else:
            target = target[:self.decoder_max_sequence_length]
        
        train_data = {"input_ids": phrase, "labels": target, "attention_mask": phrase_attention_mask}

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

    batch_size = configs['training']['phrase_generation']['batch_size']

    # Artifact folder
    artifact_folder = configs['raw_data']['artifact_folder']
    # Load encoder tokenizer json file dictionary
    tokenizer_filepath = os.path.join(artifact_folder, "tokenizer.json")
    # Load the tokenizer dictionary
    with open(tokenizer_filepath, 'r') as f:
        tokenizer = json.load(f)

        
    # Open the train, validation, and test sets json files
    with open(os.path.join(artifact_folder, "train_file_list.json"), "r") as f:
        train_file_list = json.load(f)
    with open(os.path.join(artifact_folder, "valid_file_list.json"), "r") as f:
        valid_file_list = json.load(f)
    with open(os.path.join(artifact_folder, "test_file_list.json"), "r") as f:
        test_file_list = json.load(f)

    # Print length of train, validation, and test sets
    print("Length of train set: ", len(train_file_list))
    print("Length of validation set: ", len(valid_file_list))
    print("Length of test set: ", len(test_file_list))

    # Load the dataset
    dataset = JSONDataset(configs, train_file_list, mode="train")
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for n, data in enumerate(dataset):
        # print shape and type of tensor
        print(data["input_ids"].shape, data["input_ids"].dtype)
        print(data["labels"].shape, data["labels"].dtype)
        print(data["attention_mask"].shape, data["attention_mask"].dtype)
        if n > 0:
            break