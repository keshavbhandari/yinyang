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
from utils.utils import list_to_remi_encoding, find_beats_in_bar
from phrase_refiner.transformations import Melodic_Development


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

        # Length of file_number_phrase_number
        self.length_phrases = len(self.file_number_phrase_number)
        # Length of file_list
        self.length_files = len(self.file_list)

        # Print length of dataset
        print("Length of dataset: ", self.length_files)
        print("Length of phrases in dataset: ", self.length_phrases)

        # Artifact folder
        self.artifact_folder = configs['raw_data']['artifact_folder']
        # Load encoder tokenizer json file dictionary
        tokenizer_filepath = os.path.join(self.artifact_folder, "tokenizer.json")
        # Load the tokenizer dictionary
        with open(tokenizer_filepath, 'r') as f:
            self.tokenizer = json.load(f)

        # Get the maximum sequence length
        self.max_sequence_length = configs['model']['phrase_similarity_model']['max_sequence_length']

    def __len__(self):
        return self.length_files
    
    def transpose(self, phrase, pitch_change):
        encoding = copy.deepcopy(phrase)

        transposed_encoding = [
            [event[0], event[1], event[2], event[3] + pitch_change, *event[4:]]
            for event in encoding
        ]

        return transposed_encoding

    def augment_phrase(self, phrase_1):
        if random.random() < 0.5:
            pitch_change = random.choice([i for i in range(-12,12) if i not in [0]])

            encoding = copy.deepcopy(phrase_1)

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

        return phrase_1
    
    def reindex_phrase(self, phrase_1, phrase_2, beats_in_bar):
        self.melodic_development_obj = Melodic_Development(beats_in_bar=beats_in_bar)
        phrase_2 = self.melodic_development_obj.reindex_bars(phrase_2, start_bar=phrase_1[-1][0]+1)

        return phrase_2

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        # Get max phrases in file
        with open(os.path.join(self.data_dir, file_path), 'r') as f:
            data = json.load(f)
        
        time_signature = data["metadata"]["time_signature"]
        # Get key and major or minor for phrase
        key_signature = data["metadata"]["key_signature"]
        major_or_minor = data["metadata"]["major_or_minor"]

        # Choose a random phrase number
        total_phrases = len(data["phrases"].keys())
        phrase_number = random.randint(0, total_phrases-1)
        
        phrase_1 = data["phrases"][str(phrase_number)][0]
        if random.random() < 0.5:
            # Get the phrase and the target from the same file as a positive sample            
            all_phrases = list(range(0, total_phrases))
            all_phrases.remove(phrase_number)
            phrase_number = random.choice(all_phrases)
            phrase_2 = data["phrases"][str(phrase_number)][0]
            # Reindex phrase_2 to match the last bar + 1 of phrase_1
            beats_in_bar = find_beats_in_bar(time_signature)
            phrase_2 = self.reindex_phrase(phrase_1, phrase_2, beats_in_bar)
            target = torch.tensor(1)
        else:
            # Choose a random file from self.file_list that is not idx
            random_file = random.choice([i for i in range(self.length_files) if i != idx])
            random_file_path = self.file_list[random_file]
            with open(os.path.join(self.data_dir, random_file_path), 'r') as f:
                random_data = json.load(f)
            # Choose a random phrase from the random file as a negative sample
            random_phrase_number = random.randint(0, len(random_data["phrases"].keys())-1)
            phrase_2 = random_data["phrases"][str(random_phrase_number)][0]
            # Reindex phrase_2 to match the last bar + 1 of phrase_1
            beats_in_bar = find_beats_in_bar(time_signature)
            phrase_2 = self.reindex_phrase(phrase_1, phrase_2, beats_in_bar)
            key_signature = random_data["metadata"]["key_signature"]
            major_or_minor = random_data["metadata"]["major_or_minor"]
            target = torch.tensor(0)

        # Augment the phrases
        if self.mode == "train":
            phrase_1 = self.augment_phrase(phrase_1)
            phrase_2 = self.augment_phrase(phrase_2)
            # phrase_1, phrase_2 = self.augment_phrase(phrase_1, phrase_2)

        tempo_location = data["metadata"]["tempo"]

        # Add phrase 1 to phrase 2
        phrase = phrase_1 + ["SEP"] + phrase_2
        # List to remi encoding
        phrase = list_to_remi_encoding(phrase, tempo_location, time_signature)
        # Add the BOS and EOS tokens to the phrase
        phrase = ["BOS"] + phrase + ["EOS"]
        # Tokenize the phrase
        phrase = [self.tokenizer[note] for note in phrase if note in self.tokenizer]

        # Convert to tensor and pad the phrase to a fixed length of max_sequence_length if the phrase is shorter than max_sequence_length
        phrase = torch.tensor(phrase)
        if len(phrase) < self.max_sequence_length:
            phrase = F.pad(phrase, (0, self.max_sequence_length - len(phrase)))
        else:
            phrase = phrase[-self.max_sequence_length:]
        # Attention mask based on non-padded tokens of the phrase
        phrase_attention_mask = torch.where(phrase != 0, 1, 0)
        phrase_attention_mask = phrase_attention_mask.type(torch.bool)
        
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
    dataset = JSONDataset(configs, train_file_list, beats_in_bar=32, mode="train")
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for n, data in enumerate(dataset):
        # print shape and type of tensor
        print(data["input_ids"].shape, data["input_ids"].dtype)
        print(data["labels"].shape, data["labels"].dtype)
        if n > 5:
            break