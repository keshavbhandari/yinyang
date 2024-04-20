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
from transformations import Phrase_Corruption, Melodic_Development
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.utils import list_to_remi_encoding, duration_mapping, find_beats_in_bar


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
        self.encoder_max_sequence_length = configs['model']['phrase_refinement_model']['encoder_max_sequence_length']
        self.decoder_max_sequence_length = configs['model']['phrase_refinement_model']['decoder_max_sequence_length']

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

    def augment_phrase(self, melodic_development_obj, phrase_1, phrase_2, target):
        if random.random() < 0.5:
            pitch_change = random.choice([i for i in range(-12,12) if i not in [0]])

            encoding = phrase_1 + phrase_2 + target

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
            phrase_2 = self.transpose(phrase_2, pitch_change)
            target = self.transpose(target, pitch_change)
        
        if random.random() < 0.5:
            # Modify the position of the phrase 2
            phrase_2 = melodic_development_obj.fix_bars(phrase_2, start_onset=0)            

        return phrase_1, phrase_2, target
    
    def corrupt_phrase(self, phrase, key_signature, mode, beats_in_bar):
        # Load the phrase corruption class
        self.phrase_corruption_obj = Phrase_Corruption(beats_in_bar)
        phrase = self.phrase_corruption_obj.apply_corruptions(phrase, key_signature, mode)

        return phrase
    
    def get_last_bar(self, melodic_development_obj, phrase):
        # Get the last bar of the phrase
        grouped_phrase = melodic_development_obj.group_by_bar(phrase)
        last_bar = grouped_phrase[-1]
        return last_bar

    def __getitem__(self, idx):
        file_path = self.file_number_phrase_number[idx][0]
        phrase_number = self.file_number_phrase_number[idx][1]
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Get the phrase and the target
        time_signature = data["metadata"]["time_signature"]
        key_signature = data["metadata"]["key_signature"]
        major_or_minor = data["metadata"]["major_or_minor"]
        phrase_1 = data["phrases"][str(phrase_number)][0]
        phrase_2 = data["phrases"][str(phrase_number + 1)][0]
        phrase_2_position = data["phrases"][str(phrase_number + 1)][1]
        phrase_2_cadence = data["phrases"][str(phrase_number + 1)][2]
        phrase_2_pitch_range = data["phrases"][str(phrase_number + 1)][3]
        phrase_2_length = data["phrases"][str(phrase_number + 1)][4]
        target = data["phrases"][str(phrase_number + 1)][0]

        beats_in_bar = find_beats_in_bar(time_signature)
        melodic_development_obj = Melodic_Development(beats_in_bar)

        # Augment the phrases
        if self.mode == "train":
            phrase_1, phrase_2, target = self.augment_phrase(melodic_development_obj, phrase_1, phrase_2, target)

        tempo_location = data["metadata"]["tempo"]

        if random.random() < 0.2:
            # Take the last bar of phrase_1 as phrase_1
            phrase_1 = self.get_last_bar(melodic_development_obj, phrase_1)
            # Corrupt the phrase 2 here
            phrase_2, corruption_tokens = self.corrupt_phrase(phrase_2, key_signature, major_or_minor, beats_in_bar)
            
            # Just take last note of phrase_1
            phrase_1 = phrase_1[-1:]
            phrase = phrase_1 + ["SEP"] + [key_signature] + [major_or_minor] + [phrase_2_length] + [phrase_2_cadence] + corruption_tokens + ["SEP"] + phrase_2
        else:
            # Corrupt the phrase 2 here
            phrase_2, corruption_tokens = self.corrupt_phrase(phrase_2, key_signature, major_or_minor, beats_in_bar)

            # Add phrase 1 to phrase 2
            if random.random() < 0.33:
                phrase = phrase_1 + ["SEP"] + [key_signature] + [major_or_minor] + [phrase_2_length] + [phrase_2_cadence] + corruption_tokens + ["SEP"] + phrase_2
            elif random.random() < 0.67:
                phrase = phrase_1 + ["SEP"] + [phrase_2_length] + [phrase_2_cadence] + corruption_tokens + ["SEP"] + phrase_2
            else:
                phrase = phrase_1 + ["SEP"] + [phrase_2_cadence] + corruption_tokens + ["SEP"] + phrase_2

        # List to remi encoding
        phrase = list_to_remi_encoding(phrase, tempo_location, time_signature)
        # Add the BOS and EOS tokens to the phrase
        phrase = ["BOS"] + phrase + ["EOS"]
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
        if n > 5:
            break