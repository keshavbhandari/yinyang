from miditok import REMIPlus, TokenizerConfig, TokSequence
from miditoolkit import MidiFile
import glob
import random
import copy
import pickle
import json
import os
from tqdm import tqdm
from pathlib import Path
import yaml
import argparse
import jsonlines
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.utils import annotation_to_encoding

def load_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for line in reader:
            data.append(line)
    return data

def extract_phrases(encoding, phrase_annotation):

    # Check if length of encoding is equal to length of phrase_endings
    if len(encoding) != len(phrase_annotation["features"]["phrase_end"]):
        return []

    # Find indices where phrase_endings is True
    true_indices = [i for i, value in enumerate(phrase_annotation["features"]["phrase_end"]) if value]

    # Segment the encoding based on true_indices
    segmented_encoding = []
    start_index = 0
    for end_index in true_indices:
        segment = encoding[start_index:end_index + 1]
        segmented_encoding.append(segment)
        start_index = end_index + 1

    # Add the remaining part if any
    if start_index < len(encoding):
        segmented_encoding.append(encoding[start_index:])

    return segmented_encoding

def main(configs):
    # Make artifact folder if it doesn't exist
    artifact_folder = configs["raw_data"]["artifact_folder"]
    os.makedirs(artifact_folder, exist_ok=True)

    # mono_folder = configs["raw_data"]["mono_folder"]
    # os.makedirs(mono_folder, exist_ok=True)
    # # Get all the files in the folder
    # mono_files = glob.glob(mono_folder + "/*.mid")
    raw_data_folders = configs["raw_data"]["raw_data_folders"]

    # Create a folder to store the json files
    json_folder = configs["raw_data"]["json_folder"]
    os.makedirs(json_folder, exist_ok=True)

    # Create a dictionary to store the tokenizer
    tokenizer_dict = {}
    # Initialize midi pitches from 0 to 127 to tokenizer dictionary
    for i in range(128):
        tokenizer_dict[f"Pitch_{i}"] = len(tokenizer_dict) + 1

    phrase_lengths = []
    # Iterate through each dataset, extract phrases and save them as json files
    for dataset_name, dataset_info in raw_data_folders.items():

        annotation_file = load_jsonl(dataset_info.get('annotation_filepath'))

        # Load the midi files that are in phrase annotations
        for phrase_annotation in tqdm(annotation_file):
            midi_filepath = os.path.join(json_folder, phrase_annotation["id"])

            encoding, time_signature, key_signature, major_or_minor = annotation_to_encoding(phrase_annotation)
            if len(encoding) == 0:
                print(f"Skipping {midi_filepath} as encoding is most likely corrupt")
                continue

            # Create a dictionary to store the phrases
            phrase_dict = {'metadata': {'tempo': {0: 120}, 
                                'time_signature': f"TimeSig_{time_signature}",
                                'key_signature': f"KS_{key_signature}",
                                'major_or_minor': f"MM_{major_or_minor}"},
                    'phrases': {}}

            # Extract the phrases from the encoding
            phrases = extract_phrases(encoding, phrase_annotation)

            if len(phrases) <= 1:
                print(f"Skipping {midi_filepath} as there is only one phrase")
                continue
            
            for n, phrase in enumerate(phrases):
                phrase_info = [phrase]                    

                # Phrase position: beginning, middle or end relative to the total number of phrases
                phrase_position = "PP_middle" if n > 0 and n < len(phrases) - 1 else "PP_beginning" if n == 0 else "PP_end"
                phrase_info.append(phrase_position)

                # Check if last note duration is greater than equal to minim and last note pitch is lower than second last note pitch
                if len(phrase) > 1:
                    if phrase[-1][4] >= 2 and phrase[-1][3] < phrase[-2][3]:
                        # Add True to phrase_info
                        phrase_info.append("CA_True")                        
                    else:
                        # Add False to phrase_info
                        phrase_info.append("CA_False")
                else:
                    # Add False to phrase_info
                    phrase_info.append("CA_False")

                # Pitch range of the phrase
                pitch_range = max([note[3] for note in phrase]) - min([note[3] for note in phrase])
                if f"PR_{pitch_range}" not in tokenizer_dict.keys():
                    tokenizer_dict[f"PR_{pitch_range}"] = len(tokenizer_dict) + 1
                phrase_info.append(f"PR_{pitch_range}")

                # Phrase length
                phrase_length = len(phrase)
                if f"PL_{phrase_length}" not in tokenizer_dict.keys():
                    tokenizer_dict[f"PL_{phrase_length}"] = len(tokenizer_dict) + 1
                phrase_info.append(f"PL_{phrase_length}")

                # Add phrase info to phrase dictionary
                phrase_dict['phrases'][n] = phrase_info

                # Add durations and onsets of each note in the phrase to tokenizer dictionary if it doesn't exist
                for note in phrase:
                    note_duration = round(note[4], 2)
                    note_onset = round(note[1], 2)
                    if f"Duration_{note_duration}" not in tokenizer_dict.keys():
                        tokenizer_dict[f"Duration_{note_duration}"] = len(tokenizer_dict) + 1
                    if f"Position_{note_onset}" not in tokenizer_dict.keys():
                        tokenizer_dict[f"Position_{note_onset}"] = len(tokenizer_dict) + 1
                
                # Add time signature to tokenizer dictionary if it doesn't exist
                if f"TimeSig_{time_signature}" not in tokenizer_dict.keys():
                    tokenizer_dict[f"TimeSig_{time_signature}"] = len(tokenizer_dict) + 1
                # Add key signature to tokenizer dictionary if it doesn't exist
                if f"KS_{key_signature}" not in tokenizer_dict.keys():
                    tokenizer_dict[f"KS_{key_signature}"] = len(tokenizer_dict) + 1
                # Add major or minor to tokenizer dictionary if it doesn't exist
                if f"MM_{major_or_minor}" not in tokenizer_dict.keys():
                    tokenizer_dict[f"MM_{major_or_minor}"] = len(tokenizer_dict) + 1

                phrase_lengths.append(len(phrase))

            # Write phrases as a json file
            midi_file = Path(f"{midi_filepath}.json")
            with open(midi_file, "w") as f:
                json.dump(phrase_dict, f)

    # Add special tokens to tokenizer dictionary
    tokenizer_dict["Bar_None"] = len(tokenizer_dict) + 1
    # Add phrase position tokens to tokenizer dictionary
    tokenizer_dict["PP_beginning"] = len(tokenizer_dict) + 1
    tokenizer_dict["PP_middle"] = len(tokenizer_dict) + 1
    tokenizer_dict["PP_end"] = len(tokenizer_dict) + 1
    # Add cadence tokens to tokenizer dictionary
    tokenizer_dict["CA_True"] = len(tokenizer_dict) + 1
    tokenizer_dict["CA_False"] = len(tokenizer_dict) + 1
    # Add special tokens to tokenizer dictionary
    tokenizer_dict["BOS"] = len(tokenizer_dict) + 1
    tokenizer_dict["EOS"] = len(tokenizer_dict) + 1
    tokenizer_dict["SEP"] = len(tokenizer_dict) + 1
    # Add corruption tokens to tokenizer dictionary
    tokenizer_dict["COR_incorrect_transposition"] = len(tokenizer_dict) + 1
    tokenizer_dict["COR_incorrect_inversion"] = len(tokenizer_dict) + 1
    tokenizer_dict["COR_note_swapping"] = len(tokenizer_dict) + 1
    tokenizer_dict["COR_melodic_stripping"] = len(tokenizer_dict) + 1
    tokenizer_dict["COR_melodic_addition"] = len(tokenizer_dict) + 1
    tokenizer_dict["COR_same_note_modification"] = len(tokenizer_dict) + 1
    tokenizer_dict["COR_permute_note_pitch"] = len(tokenizer_dict) + 1
    tokenizer_dict["COR_permute_note_duration"] = len(tokenizer_dict) + 1
    tokenizer_dict["COR_permute_note_pitch_duration"] = len(tokenizer_dict) + 1
    tokenizer_dict["COR_BAR_MASK"] = len(tokenizer_dict) + 1
    tokenizer_dict["COR_PITCH_MASK"] = len(tokenizer_dict) + 1
    tokenizer_dict["COR_DURATION_MASK"] = len(tokenizer_dict) + 1
    tokenizer_dict["COR_FRAGMENT_NOTES"] = len(tokenizer_dict) + 1
    # Add special tokens to tokenizer dictionary
    tokenizer_dict["UNK"] = len(tokenizer_dict) + 1

    # Save the encoder tokenizer dictionary as a json file
    encoder_tokenizer_filepath = os.path.join(artifact_folder, "tokenizer.json")
    with open(encoder_tokenizer_filepath, "w") as f:
        json.dump(tokenizer_dict, f)

    return phrase_lengths



if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.normpath("configs/configs_os.yaml"),
                        help="Path to the config file")
    args = parser.parse_args()

    # Load config file
    with open(args.config, 'r') as f:
        configs = yaml.safe_load(f)

    phrase_lengths = main(configs)

    # Describe the phrase lengths
    max_phrase_length = max(phrase_lengths)
    min_phrase_length = min(phrase_lengths)
    average_phrase_length = sum(phrase_lengths) / len(phrase_lengths)
    no_of_phrases = len(phrase_lengths)
    print(f"Number of phrases: {no_of_phrases}, Max phrase length: {max_phrase_length}, Min phrase length: {min_phrase_length}, Average phrase length: {average_phrase_length}")

    # Save the phrase lengths list in a json file in the artifact folder
    phrase_lengths_filepath = os.path.join(configs["raw_data"]["artifact_folder"], "phrase_lengths.json")

    with open(phrase_lengths_filepath, "w") as f:
        json.dump(phrase_lengths, f)
    print(f"Phrase lengths saved to {phrase_lengths_filepath}")

  