from transformers import EncoderDecoderModel, AutoModelForSequenceClassification
import torch
from torch.cuda import is_available as cuda_available, is_bf16_supported
import torch.nn.functional as F
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

max_sequence_length = configs['model']['structure_derivation_model']['max_sequence_length']

# Artifact folder
artifact_folder = configs['raw_data']['artifact_folder']

# Load tokenizer json file dictionary
dec_tokenizer_filepath = os.path.join(artifact_folder, 'tokenizer.json')
with open(dec_tokenizer_filepath, 'r') as f:
    dec_tokenizer = json.load(f)
reverse_dec_tokenizer = {str(v): k for k, v in dec_tokenizer.items()}

# Load the phrase similarity model
structure_derivation_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(artifact_folder, "phrase_similarity"))
structure_derivation_model.eval()
structure_derivation_model.to("cuda" if cuda_available() else "cpu")

def reindex_phrase(phrase_1, phrase_2, beats_in_bar):
    melodic_development_obj = Melodic_Development(beats_in_bar=beats_in_bar)
    phrase_2 = melodic_development_obj.reindex_bars(phrase_2, start_bar=phrase_1[-1][0]+1)

    return phrase_2

# Test folder name
test_folder = "/homes/kb658/PhraseBuilder/output/yin_yang_ablated_all"
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

avg_prob_per_song = []
avg_vendi_per_song = []
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

    list_of_probs = []
    list_of_embeddings = []

    while True:
        # Get the next two bars as phrase 2
        if increment + num_bars <= total_bars:
            phrase_2 = bars[increment:increment+num_bars]
            increment += num_bars
        else:
            break
        # Flatten the phrase
        phrase_2 = [note for bar in phrase_2 for note in bar]

        # Reindex phrase 2
        phrase_2 = reindex_phrase(phrase_1, phrase_2, beats_per_bar)

        # Add phrase 1 to phrase 2
        phrase = phrase_1 + ["SEP"] + phrase_2
        # List to remi encoding
        phrase = list_to_remi_encoding(phrase, {}, time_signature)
        # Add the BOS and EOS tokens to the phrase
        phrase = ["BOS"] + phrase + ["EOS"]
        # Tokenize the phrase
        phrase = [dec_tokenizer[note] for note in phrase if note in dec_tokenizer]

        # Convert the phrase to tensor
        input_ids = torch.tensor(phrase).unsqueeze(0).to("cuda" if cuda_available() else "cpu")

        output = structure_derivation_model(input_ids, output_hidden_states=True)
        logits = output.logits
        # Get the probability of the phrase as sigmoid of the logits
        prob = F.sigmoid(logits)
        prob = prob[-1, -1].item()
        # print("Probability of the phrase: ", prob)
        list_of_probs.append(prob)

        # Get the hidden states as the phrase embedding
        last_hidden_states = output.hidden_states[-1]
        embedding = last_hidden_states[0, 0, :]
        # print("Phrase embedding: ", embedding.shape)
        # Convert the embedding to numpy
        embedding = embedding.cpu().detach().numpy()
        list_of_embeddings.append(embedding)
        

    # Print average probability of the phrases
    if len(list_of_probs) == 0:
        continue
    avg_prob = sum(list_of_probs) / len(list_of_probs)
    print("Average probability of the phrases: ", avg_prob)
    avg_prob_per_song.append(avg_prob)

    # Calculate the similarity matrix
    similarity_matrix = cosine_similarity(list_of_embeddings)
    # Print average vendi score of the phrases
    avg_vendi_score = vendi.score_K(similarity_matrix)
    print("Average vendi score of the phrases: ", avg_vendi_score)
    avg_vendi_per_song.append(avg_vendi_score)

# Print the average probability of the phrases in the test folder
avg_prob_per_song = sum(avg_prob_per_song) / len(avg_prob_per_song)
print("Average probability of the phrases in the test folder: ", avg_prob_per_song)

avg_vendi_per_song = sum(avg_vendi_per_song) / len(avg_vendi_per_song)
print("Average vendi score of the phrases in the test folder: ", avg_vendi_per_song)