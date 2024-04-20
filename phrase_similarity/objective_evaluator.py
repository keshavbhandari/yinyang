from transformers import EncoderDecoderModel, AutoModelForSequenceClassification
import torch
from torch.cuda import is_available as cuda_available, is_bf16_supported
import torch.nn.functional as F
from miditok import TokSequence
import pickle
import yaml
import json
import os
import argparse
import random
import copy
import numpy as np
from vendi_score import vendi
from sklearn.metrics.pairwise import cosine_similarity
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.utils import parse_midi, list_to_remi_encoding, encoding_to_midi, string_to_list_encoding, find_beats_in_bar
from phrase_refiner.transformations import Melodic_Development

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=os.path.normpath("/homes/kb658/yinyang/configs/configs_os.yaml"),
                    help="Path to the config file")
args = parser.parse_args()

# Load config file
with open(args.config, 'r') as f:
    configs = yaml.safe_load(f)

max_sequence_length = configs['model']['phrase_similarity_model']['max_sequence_length']

# Artifact folder
artifact_folder = configs['raw_data']['artifact_folder']

# Load tokenizer json file dictionary
dec_tokenizer_filepath = os.path.join(artifact_folder, 'tokenizer.json')
with open(dec_tokenizer_filepath, 'r') as f:
    dec_tokenizer = json.load(f)
reverse_dec_tokenizer = {str(v): k for k, v in dec_tokenizer.items()}

# # Load the phrase similarity model
# phrase_similarity_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(artifact_folder, "phrase_similarity"))
# phrase_similarity_model.eval()
# phrase_similarity_model.to("cuda" if cuda_available() else "cpu")

def reindex_phrase(phrase_1, phrase_2, beats_in_bar):
    melodic_development_obj = Melodic_Development(beats_in_bar=beats_in_bar)
    phrase_2 = melodic_development_obj.reindex_bars(phrase_2, start_bar=phrase_1[-1][0]+1)

    return phrase_2

# Test folder name
test_folder = "/homes/kb658/PhraseBuilder/output/yin_yang_ablated_low"
# test_folder = "/homes/kb658/PhraseBuilder/data/Mono_Midi_Files"
run_test = False

if run_test:
    # Load test file list
    with open(os.path.join(artifact_folder, "test_file_list.json"), "r") as f:
        test_file_list = json.load(f)
    # Get all the files in the test folder
    test_files = [os.path.basename(file) for file in test_file_list]
    # Convert .json to .mid
    test_files = [file.replace(".json", ".mid") for file in test_files]
else:
    # Get all the files in the test folder
    test_files = os.listdir(test_folder)

avg_pr_per_song = []
avg_pitches_per_song = []
# Loop through all the test files
for i, test_file in enumerate(test_files):

    # # Load a random test file (midi file)
    # test_file = random.choice(test_files)
    print("Test file: ", test_file)
    test_file_path = os.path.join(test_folder, test_file)

    # Load the midi file
    midi_data, time_signature = parse_midi(test_file_path)
    beats_per_bar = find_beats_in_bar(time_signature)

    melodic_development_obj = Melodic_Development(beats_in_bar=beats_per_bar)
    # Chop the midi data into bars
    bars = melodic_development_obj.group_by_bar(midi_data)
    total_bars = len(bars)

    # Load the same test file from extracted_phrases folder within data and extract the first phrase
    extracted_phrases_folder = "/homes/kb658/PhraseBuilder/data/extracted_phrases"
    extracted_phrases_file = os.path.join(extracted_phrases_folder, test_file.split(".")[0] + ".json")
    # Load the extracted phrases
    with open(extracted_phrases_file, 'r') as f:
        extracted_phrases = json.load(f)

    first_phrase = extracted_phrases["phrases"]['0'][0]
    # Get number of bars in the first phrase
    num_bars = first_phrase[-1][0] + 1
    increment = num_bars

    # Get the bars of phrase 1
    phrase_1 = bars[0:num_bars]
    # Flatten the phrase
    phrase_1 = [note for bar in phrase_1 for note in bar]
    print("Phrase 1: ", phrase_1)

    pitch_ranges = []
    while True:
        # Get the next two bars as phrase 2
        if increment + num_bars <= total_bars:
            phrase_2 = bars[increment:increment+num_bars]
            increment += num_bars
        else:
            break
        # Flatten the phrase
        phrase_2 = [note for bar in phrase_2 for note in bar]

        # Calculate average pitch range of the song
        pitches = []
        for note in phrase_2:
            pitches.append(note[3])
        pitch_range = max(pitches) - min(pitches)
        pitch_ranges.append(pitch_range)
        
        # Calculate unique pitches in the phrase
        unique_pitches = set(pitches) 

    avg_pitch_range = sum(pitch_ranges) / len(pitch_ranges)
    avg_pr_per_song.append(avg_pitch_range)
    print("Average pitch range of bars in the song: ", avg_pitch_range)
    avg_pitches_per_song.append(len(unique_pitches))
    print("Unique pitches in the phrase: ", len(unique_pitches))

# Print the average probability of the phrases in the test folder
avg_pr = sum(avg_pr_per_song) / len(avg_pr_per_song)
print("Average pitch range of bars in the folder: ", avg_pr)

# Print the average number of unique pitches in the phrases in the test folder
avg_pitches = sum(avg_pitches_per_song) / len(avg_pitches_per_song)
print("Average number of unique pitches in the folder: ", avg_pitches)