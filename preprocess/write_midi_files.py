import os
import json
import random
import yaml
import copy
import argparse
import glob
from tqdm import tqdm
import sys
import jsonlines
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.utils import annotation_to_encoding, encoding_to_midi

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=os.path.normpath("configs/configs_os.yaml"),
                    help="Path to the config file")
args = parser.parse_args()

# Load config file
with open(args.config, 'r') as f:
    configs = yaml.safe_load(f)

# Get the raw data folders
artifact_folder = configs["raw_data"]["artifact_folder"]
mono_folder = configs["raw_data"]["mono_folder"]
json_folder = configs["raw_data"]["json_folder"]
raw_data_folders = configs["raw_data"]["raw_data_folders"]
mono_folder = configs["raw_data"]["mono_folder"]
os.makedirs(mono_folder, exist_ok=True)

# Get all files from json_folder
json_files = glob.glob(json_folder + "/*.json")

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

for dataset_name, dataset_info in raw_data_folders.items():

    annotation_file = load_jsonl(dataset_info.get('annotation_filepath'))

    # Load the midi files that are in phrase annotations
    for phrase_annotation in tqdm(annotation_file):

        midi_filepath = os.path.join(mono_folder, phrase_annotation["id"] + '.mid')

        encoding, time_signature, key_signature, major_or_minor = annotation_to_encoding(phrase_annotation)
        if len(encoding) == 0:
            print(f"Skipping {phrase_annotation['id']} as encoding is most likely corrupt")
            continue

        # Extract the phrases from the encoding
        phrases = extract_phrases(encoding, phrase_annotation)

        if len(phrases) <= 1:
            print(f"Skipping {midi_filepath} as there is only one phrase")
            continue
            
        encoding_to_midi(encoding, {0: 120}, f"TimeSig_{time_signature}", midi_filepath)
