from transformers import EncoderDecoderModel, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from torch.cuda import is_available as cuda_available, is_bf16_supported
from miditok import TokSequence
import pickle
import yaml
import json
import os
import argparse
import random
# from phrase_refiner.data_loader import JSONDataset
from phrase_refiner.transformations import Melodic_Development, Phrase_Corruption, duration_mapping, reverse_duration_mapping
from utils.utils import list_to_remi_encoding, encoding_to_midi, string_to_list_encoding, find_beats_in_bar

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=os.path.normpath("/homes/kb658/yinyang/configs/configs_os.yaml"),
                    help="Path to the config file")
args = parser.parse_args()

# Load config file
with open(args.config, 'r') as f:
    configs = yaml.safe_load(f)

refinement_max_sequence_length = configs['model']['phrase_refinement_model']['decoder_max_sequence_length']
generation_max_sequence_length = configs['model']['phrase_generation_model']['decoder_max_sequence_length']

# Artifact folder
artifact_folder = configs['raw_data']['artifact_folder']

# Load tokenizer json file dictionary
dec_tokenizer_filepath = os.path.join(artifact_folder, 'tokenizer.json')
with open(dec_tokenizer_filepath, 'r') as f:
    dec_tokenizer = json.load(f)
reverse_dec_tokenizer = {str(v): k for k, v in dec_tokenizer.items()}

# Load test file list
with open(os.path.join(artifact_folder, "test_file_list.json"), "r") as f:
    test_file_list = json.load(f)

# Choose random test file from test_file_list
test_file = "NLB183423_01_mono.mid.json" #random.choice(test_file_list) # "NLB145289_01_mono.mid.json"
print("Test file: ", test_file)

# Read test file as json
with open(os.path.join(configs['raw_data']['json_folder'], test_file), "r") as f:
    test_phrases = json.load(f)

# Load the phrase refiner model
phrase_refiner_model = EncoderDecoderModel.from_pretrained(os.path.join(artifact_folder, "phrase_refinement"))
phrase_refiner_model.eval()
phrase_refiner_model.to("cuda" if cuda_available() else "cpu")

# Load the motif refiner model
motif_refiner_model = EncoderDecoderModel.from_pretrained(os.path.join(artifact_folder, "motif_refinement"))
motif_refiner_model.eval()
motif_refiner_model.to("cuda" if cuda_available() else "cpu")

# Load the phrase generation model
phrase_generation_model = EncoderDecoderModel.from_pretrained(os.path.join(artifact_folder, "phrase_generation"))
phrase_generation_model.eval()
phrase_generation_model.to("cuda" if cuda_available() else "cpu")

# Load the phrase selection model
phrase_selection_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(artifact_folder, "phrase_selection"))
phrase_selection_model.eval()
phrase_selection_model.to("cuda" if cuda_available() else "cpu")

# Function to transform phrase
def transform_phrase(phrase, major_or_minor, key_signature, reset_bar_index=0):
    # Reindex the phrase based on last bar
    phrase = melodic_development_obj.reindex_bars(phrase, start_bar = reset_bar_index)

    major_or_minor = True if major_or_minor == "major" else False
    # Get number of notes in the phrase
    phrase_length = len([note for note in phrase])

    transformation_names, symmetric_corruptions = [], []

    # Apply permutation transformation
    transformed_phrase = melodic_development_obj.permute_melody_pitch_rhythm(phrase)
    transformation_names.append(melodic_development_obj.sequence_melody.__name__)
    symmetric_corruptions.append("COR_permute_note_pitch_duration")

    # # Apply fragmentation transformation
    # transformed_phrase = phrase_corruption_obj.fragment_notes(phrase, strategy="bar")
    # transformation_names.append(phrase_corruption_obj.fragment_notes.__name__)
    # symmetric_corruptions.append("COR_FRAGMENT_NOTES")
    # symmetric_corruptions.append("COR_incorrect_transposition")


    # Apply sequencing transformation
    # transformed_phrase = melodic_development_obj.sequence_melody(phrase, pitch_change=-5)
    # transformation_names.append(melodic_development_obj.sequence_melody.__name__)
    # symmetric_corruptions.append("COR_permute_note_duration")

    # mask_type = "pitch"
    # transformed_phrase = phrase_corruption_obj.masking(phrase, mask_type=mask_type)
    # transformation_names.append(phrase_corruption_obj.masking.__name__)
    # symmetric_corruptions.append("COR_PITCH_MASK")

    return transformed_phrase, symmetric_corruptions, transformation_names




    # if random.random() < 0.2:
    #     mask_type = random.choice(["bar", "pitch", "duration"])
    #     transformed_phrase = phrase_corruption_obj.masking(phrase, mask_type=mask_type)
    #     transformation_names.append(phrase_corruption_obj.masking.__name__)
    #     if mask_type == "pitch":
    #         symmetric_corruptions.append("COR_PITCH_MASK")
    #     elif mask_type == "duration":
    #         symmetric_corruptions.append("COR_DURATION_MASK")
    #     else:
    #         symmetric_corruptions.append("COR_BAR_MASK")
    # else:
    #     # Midi to bar encoding
    #     midi_bar_encoding = melodic_development_obj.group_by_bar(phrase)
    #     # Fragment the phrase
    #     if len(midi_bar_encoding) > 2 and random.random() < 0.5:
    #         # fragmented_phrase = phrase
    #         # Choose randomly between bar and random_crop
    #         fragment_type = random.choice(["bar", "random_crop"])
    #         fragmented_phrase = phrase_corruption_obj.fragment_notes(phrase, strategy=fragment_type)
    #         transformation_names.append(phrase_corruption_obj.fragment_notes.__name__)
    #         symmetric_corruptions.append("COR_FRAGMENT_NOTES")
    #     else:
    #         fragmented_phrase = phrase

    #     # Transform the phrase from a randomly selected function
    #     mirroring_corruptions = {"retrograde_melody_pitch_rhythm": "COR_incorrect_transposition",
    #                             "invert_melody_strict": "COR_incorrect_transposition",
    #                             "invert_melody_tonal": "COR_incorrect_inversion",
    #                             "sequence_melody": "COR_FRAGMENT_NOTES",
    #                             "permute_melody_pitch": "COR_permute_note_pitch",
    #                             "permute_melody_rhythm": "COR_permute_note_duration",
    #                             "permute_melody_pitch_rhythm": "COR_permute_note_pitch_duration",
    #                             "permute_melody_new_pitch": "COR_incorrect_transposition",
    #                             "contract_melody": "COR_permute_note_duration",
    #                             "expand_melody": "COR_incorrect_transposition",
    #                             "reduce_melody": "COR_same_note_modification",
    #                             "embellish_melody": "COR_same_note_modification",
    #                             "change_major_minor": "COR_incorrect_inversion",
    #                             "metric_displacement": "COR_DURATION_MASK",
    #                          }
    #     # Randomly select a transformation function
    #     transformation_function = random.choice([
    #         melodic_development_obj.retrograde_melody_pitch_rhythm,
    #         melodic_development_obj.invert_melody_strict,
    #         melodic_development_obj.invert_melody_tonal,
    #         melodic_development_obj.sequence_melody,
    #         melodic_development_obj.permute_melody_pitch,
    #         melodic_development_obj.permute_melody_rhythm,
    #         melodic_development_obj.permute_melody_pitch_rhythm,
    #         melodic_development_obj.permute_melody_new_pitch,
    #         melodic_development_obj.contract_melody,
    #         melodic_development_obj.expand_melody,
    #         melodic_development_obj.reduce_melody,
    #         # # melodic_development_obj.embellish_melody,
    #         # # melodic_development_obj.change_major_minor,
    #         # # melodic_development_obj.metric_displacement
    #     ])
    #     transformation_function = melodic_development_obj.permute_melody_rhythm
    #     transformation_names.append(transformation_function.__name__)
    #     symmetric_corruptions.append(mirroring_corruptions[transformation_function.__name__])

    #     # Apply the randomly selected transformation function to the fragmented_phrase
    #     transformed_phrase = transformation_function(fragmented_phrase, major=major_or_minor)

    # return transformed_phrase, symmetric_corruptions, transformation_names

# Function to refine phrase using phrase refinement model
def refine_phrase(model, phrase_1, phrase_2, transformation_type, tempo_location, gen_length, meta_data, last_phrase=False):
    cadence = "CA_True" if last_phrase else "CA_False"
    major_or_minor, key_signature, time_signature = meta_data[0], meta_data[1], meta_data[2]
    # Add phrase 1 to phrase 2
    # phrase = phrase_1 + ["SEP"] + [major_or_minor] + [cadence] + [f"PL_{gen_length}"] + transformation_type + ["SEP"] + phrase_2
    phrase = phrase_1 + ["SEP"] + [major_or_minor] + [cadence] + transformation_type + ["SEP"] + phrase_2
    # List to remi encoding
    phrase = list_to_remi_encoding(phrase, tempo_location, time_signature)
    # Add the BOS and EOS tokens to the phrase
    phrase = ["BOS"] + phrase + ["EOS"]
    # Tokenize the phrase
    phrase = [dec_tokenizer[note] for note in phrase if note in dec_tokenizer]

    # Generate a phrase
    input_ids = torch.tensor(phrase).unsqueeze(0).to("cuda" if cuda_available() else "cpu")

    generated = model.generate(input_ids, decoder_start_token_id=dec_tokenizer["BOS"], num_beams=1, do_sample=False, max_length=refinement_max_sequence_length, early_stopping=False)

    # Write the generated phrase to a MIDI file
    generated_phrase = [reverse_dec_tokenizer[str(token)] for token in generated[0].tolist()]
    # Remove special tokens
    special_tokens = ["PAD", "BOS", "EOS", "SEP"] + [i for i in dec_tokenizer.keys() if i.startswith("PL_") or i.startswith("PP_") or i.startswith("COR_")]
    generated_phrase = [token for token in generated_phrase if token not in special_tokens]
 
    generated_phrase = string_to_list_encoding(generated_phrase)

    return generated_phrase

def refine_motif(model, phrase_1, phrase_2, transformation_type, tempo_location, gen_length, meta_data, last_phrase=False):
    cadence = "CA_True" if last_phrase else "CA_False"
    major_or_minor, key_signature, time_signature = meta_data[0], meta_data[1], meta_data[2]
    # Add phrase 1 to phrase 2
    # phrase = phrase_1 + ["SEP"] + [major_or_minor] + [cadence] + [f"PL_{gen_length}"] + transformation_type + ["SEP"] + phrase_2
    phrase = phrase_1 + ["SEP"] + [key_signature] + [major_or_minor] + [f"PL_{gen_length}"] + [cadence] + transformation_type + ["SEP"] + phrase_2
    # List to remi encoding
    phrase = list_to_remi_encoding(phrase, tempo_location, time_signature)
    # Add the BOS and EOS tokens to the phrase
    phrase = ["BOS"] + phrase + ["EOS"]
    # Tokenize the phrase
    phrase = [dec_tokenizer[note] for note in phrase if note in dec_tokenizer]

    # Generate a phrase
    input_ids = torch.tensor(phrase).unsqueeze(0).to("cuda" if cuda_available() else "cpu")

    generated = model.generate(input_ids, decoder_start_token_id=dec_tokenizer["BOS"], num_beams=1, do_sample=False, max_length=refinement_max_sequence_length, early_stopping=False)

    # Write the generated phrase to a MIDI file
    generated_phrase = [reverse_dec_tokenizer[str(token)] for token in generated[0].tolist()]
    # Remove special tokens
    special_tokens = ["PAD", "BOS", "EOS", "SEP"] + [i for i in dec_tokenizer.keys() if i.startswith("PL_") or i.startswith("PP_") or i.startswith("COR_")]
    generated_phrase = [token for token in generated_phrase if token not in special_tokens]

    generated_phrase = string_to_list_encoding(generated_phrase)

    return generated_phrase

# Function to select phrase using phrase selection model
def select_phrase(model, phrase_1, phrase_2, reset_bar_index=0):

    # Reindex the phrase based on last bar
    phrase_2 = melodic_development_obj.reindex_bars(phrase_2, start_bar = reset_bar_index)

    # Add phrase 1 to phrase 2
    phrase = phrase_1 + ["SEP"] + phrase_2
    # List to remi encoding
    phrase = list_to_remi_encoding(phrase, tempo_location, time_signature)
    # Add the BOS and EOS tokens to the phrase
    phrase = ["BOS"] + phrase + ["EOS"]
    # Tokenize the phrase
    phrase = [dec_tokenizer[note] for note in phrase if note in dec_tokenizer]

    # Generate a phrase
    input_ids = torch.tensor(phrase).unsqueeze(0).to("cuda" if cuda_available() else "cpu")

    output = model(input_ids)
    logits = output.logits
    # Get the probability of the phrase as sigmoid of the logits
    prob = F.sigmoid(logits)
    prob = prob[-1, -1].item()

    return prob

def fix_bar_onset(previous_phrase, generated_phrase):
    # Get the bar number and onset of the last element in previous phrase
    bar_number, onset, duration = previous_phrase[-1][0], previous_phrase[-1][1], previous_phrase[-1][4]
    # Get the bar number and onset of the first element in the generated phrase
    bar_number_gen, onset_gen = generated_phrase[0][0], generated_phrase[0][1]
    if int(onset_gen) < int(onset):
        bar_number_gen = int(bar_number) + 1
    else:
        bar_number_gen = int(bar_number)
    # Fix the bar number and onset of the generated phrase based on the previous phrase
    generated_phrase = melodic_development_obj.fix_bars(generated_phrase, onset_gen, bar_number_gen)
    return generated_phrase


# Load test file list
with open(os.path.join(artifact_folder, "test_file_list.json"), "r") as f:
    test_file_list = json.load(f)

# Choose random test file from test_file_list
test_file = "han1006_mono.mid.json" # random.choice(test_file_list)
print("Test file: ", test_file)

# Read test file as json
with open(os.path.join(configs['raw_data']['json_folder'], test_file), "r") as f:
    test_phrases = json.load(f)

tempo_location = test_phrases['metadata']['tempo']
key_signature = test_phrases['metadata']['key_signature']
time_signature = test_phrases['metadata']['time_signature']
beats_in_bar = find_beats_in_bar(time_signature)
# Load the transformation and phrase corruption class
melodic_development_obj = Melodic_Development(beats_in_bar=beats_in_bar)
phrase_corruption_obj = Phrase_Corruption(beats_in_bar=beats_in_bar)
major_or_minor = test_phrases['metadata']['major_or_minor']
motif, motif_position = test_phrases['phrases']['0'][0], test_phrases['phrases']['0'][1]

def generate_stuff(phrase_refiner_model, motif, major_or_minor, key_signature, tempo_location):

    generation_length = len(motif)
    last_bar_motif = motif[-1][0]
    previous_pitch_range = max([note[3] for note in motif]) - min([note[3] for note in motif])
    generation_pitch_range = max(random.choice([i for i in range(8, 16)]), previous_pitch_range)
    print(f"Generation pitch range: {generation_pitch_range}")
    print(f"Generation length: {generation_length}")

    # Transform the motif
    transformed_phrase, corruption_tokens, transformation_names = transform_phrase(motif, major_or_minor, key_signature, reset_bar_index=last_bar_motif+1)
    # Refine the transformed motif
    # refined_phrase = refine_phrase(phrase_refiner_model, motif, transformed_phrase, corruption_tokens, tempo_location, generation_length, meta_data=[major_or_minor, key_signature, time_signature], last_phrase=False)
    # Refine the motif
    one_bar_motif = melodic_development_obj.group_by_bar(motif)
    one_bar_motif = one_bar_motif[-1]
    refined_phrase = refine_motif(motif_refiner_model, one_bar_motif, transformed_phrase, corruption_tokens, tempo_location, generation_length, meta_data=[major_or_minor, key_signature, time_signature], last_phrase=False)
    # # Get probability from phrase selection model
    probability = select_phrase(phrase_selection_model, motif, refined_phrase, reset_bar_index=last_bar_motif+1)
    print(f"Probability: {probability}")

    # Fix the bar onset of the generated phrase
    generated_phrase = fix_bar_onset(motif, refined_phrase)

    combined_phrases = motif + generated_phrase

    return motif, generated_phrase, combined_phrases, transformation_names, corruption_tokens, tempo_location

# Create an output folder if it doesn't exist
output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

motif, generated_phrase, combined_phrases, transformation_names, corruption_tokens, tempo_location = generate_stuff(phrase_refiner_model, motif, major_or_minor, key_signature, tempo_location)
print(f"Transformations: {transformation_names}")
print(f"Corruptions: {corruption_tokens}")

# Create an output folder if it doesn't exist
output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
output_filepath = os.path.join(output_folder, "Combined_Generated.mid")

# Write the structure to a MIDI file
encoding_to_midi(combined_phrases, tempo_location, time_signature, output_filepath)


# for i in range(16):
#     motif, generated_phrase, combined_phrases, transformation_names, tempo_location = generate_stuff(phrase_refiner_model, motif, major_or_minor, key_signature, tempo_location)
#     print(f"Iteration: {i}, Transformations: {transformation_names}")

#     # Write the motif, generated phrase and combined phrases to a MIDI file
#     encoded_phrase = list_to_remi_encoding(motif, tempo_location, reverse_duration_mapping)
#     new_midi = remi_tokenizer([encoded_phrase])
#     new_midi.dump(os.path.join(output_folder, test_file.split(".")[0] + "_motif.mid"))

#     encoded_phrase = list_to_remi_encoding(generated_phrase, tempo_location, reverse_duration_mapping)
#     new_midi = remi_tokenizer([encoded_phrase])
#     new_midi.dump(os.path.join(output_folder, test_file.split(".")[0] + f"_{transformation_names[-1]}_generated.mid"))

#     encoded_phrase = list_to_remi_encoding(combined_phrases, tempo_location, reverse_duration_mapping)
#     new_midi = remi_tokenizer([encoded_phrase])
#     new_midi.dump(os.path.join(output_folder, test_file.split(".")[0] + f"_{transformation_names[-1]}_combined.mid"))