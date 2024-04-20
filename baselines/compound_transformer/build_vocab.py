import os
import json
import random
import yaml
import copy
import argparse
import glob
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=os.path.normpath(r"configs/configs_os.yaml"),
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

# Get all files from json_folder
json_files = glob.glob(json_folder + "/*.json")

# Create dictionaries to store the tokenizer
all_tokenizer_dict = {}
pitch_tokenizer = {}
family_tokenizer = {}
metric_tokenizer = {}
velocity_tokenizer = {}
chord_tokenizer = {}
duration_tokenizer = {}
time_signature_tokenizer = {}

# Initialize midi pitches from 0 to 127 to tokenizer dictionary
pitch_tokenizer["Ignore_None"] = len(pitch_tokenizer) + 1
for i in range(128):
    pitch_tokenizer[f"Pitch_{i}"] = len(pitch_tokenizer) + 1

family_tokenizer["Family_Metric"] = len(family_tokenizer) + 1
family_tokenizer["Family_Note"] = len(family_tokenizer) + 1
family_tokenizer["BOS"] = len(family_tokenizer) + 1
family_tokenizer["EOS"] = len(family_tokenizer) + 1

metric_tokenizer["Ignore_None"] = len(metric_tokenizer) + 1
metric_tokenizer["Bar_None"] = len(metric_tokenizer) + 1

chord_tokenizer["Ignore_None"] = len(chord_tokenizer) + 1

velocity_tokenizer["Ignore_None"] = len(velocity_tokenizer) + 1

duration_tokenizer["Ignore_None"] = len(duration_tokenizer) + 1

time_signature_tokenizer["Ignore_None"] = len(time_signature_tokenizer) + 1

for file in tqdm(json_files):
    # Load the JSON file
    with open(file, 'r') as f:
        data = json.load(f)
    for phrase_number in data["phrases"].keys():
        phrase = data['phrases'][phrase_number][0]
        tempo_location = data["metadata"]["tempo"]
        time_signature = data["metadata"]["time_signature"]

        if time_signature not in time_signature_tokenizer:
            time_signature_tokenizer[time_signature] = len(time_signature_tokenizer) + 1

        for note in phrase:
            if f'Position_{note[1]}' not in metric_tokenizer:
                metric_tokenizer[f'Position_{note[1]}'] = len(metric_tokenizer) + 1
            if f'Velocity_{note[5]}' not in velocity_tokenizer:
                velocity_tokenizer[f'Velocity_{note[5]}'] = len(velocity_tokenizer) + 1
            if f'Duration_{note[4]}' not in duration_tokenizer:
                duration_tokenizer[f'Duration_{note[4]}'] = len(duration_tokenizer) + 1

# Add the tokenizer dictionaries to the all_tokenizer dictionary
all_tokenizer_dict["pitch_tokenizer"] = pitch_tokenizer
all_tokenizer_dict["family_tokenizer"] = family_tokenizer
all_tokenizer_dict["metric_tokenizer"] = metric_tokenizer
all_tokenizer_dict["velocity_tokenizer"] = velocity_tokenizer
all_tokenizer_dict["chord_tokenizer"] = chord_tokenizer
all_tokenizer_dict["duration_tokenizer"] = duration_tokenizer
all_tokenizer_dict["time_signature_tokenizer"] = time_signature_tokenizer

# Save the tokenizer dictionary to the artifact folder
tokenizer_filepath = os.path.join(artifact_folder, "cp_tokenizer.json")
print(f"Saving tokenizer dictionary to {tokenizer_filepath}")
with open(tokenizer_filepath, 'w') as f:
    json.dump(all_tokenizer_dict, f)