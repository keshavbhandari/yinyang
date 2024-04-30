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
from phrase_refiner.transformations import Melodic_Development, Phrase_Corruption
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

# Load the phrase refiner model
phrase_refiner_model = EncoderDecoderModel.from_pretrained(os.path.join(artifact_folder, "phrase_refinement"))
phrase_refiner_model.eval()
phrase_refiner_model.to("cuda" if cuda_available() else "cpu")

# Load the phrase generation model
phrase_generation_model = EncoderDecoderModel.from_pretrained(os.path.join(artifact_folder, "phrase_generation"))
phrase_generation_model.eval()
phrase_generation_model.to("cuda" if cuda_available() else "cpu")

# Load the phrase selection model
phrase_selection_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(artifact_folder, "phrase_selection"))
phrase_selection_model.eval()
phrase_selection_model.to("cuda" if cuda_available() else "cpu")

# Function to transform phrase
def transform_phrase(melodic_development_obj, phrase_corruption_obj, phrase, major_or_minor, key_signature, 
                     similarity, transformation, allow_fragment=True, reset_bar_index=0):
    # Reindex the phrase based on last bar
    phrase = melodic_development_obj.reindex_bars(phrase, start_bar = reset_bar_index)

    major_or_minor = True if major_or_minor == "major" else False
    # Get number of notes in the phrase
    phrase_length = len([note for note in phrase])

    low_similarity_transformations = {"retrograde_melody_pitch_rhythm": ["COR_incorrect_inversion", "COR_permute_note_pitch_duration"], 
                                       "retrograde_melody_pitch": ["COR_incorrect_inversion"],
                                       "invert_melody_strict": ["COR_incorrect_inversion", "COR_incorrect_transposition"],                                         
                                       "permute_melody_pitch": ["COR_permute_note_pitch", "COR_incorrect_inversion"],
                                       "permute_melody_pitch_rhythm": ["COR_permute_note_pitch_duration"],
                                       "permute_melody_new_pitch": ["COR_permute_note_pitch", "COR_incorrect_inversion", "COR_incorrect_transposition"],
                                       "invert_melody_tonal": ["COR_incorrect_inversion"],
                                       "pitch_mask": ["COR_PITCH_MASK"]}
    
    high_similarity_transformations = {"contract_melody": ["COR_same_note_modification", "COR_melodic_addition"], 
                                      "expand_melody": ["COR_melodic_stripping"], #"COR_note_swapping"
                                      "reduce_melody": ["COR_same_note_modification", "COR_melodic_addition"], 
                                      "permute_melody_rhythm": ["COR_permute_note_duration"],                                      
                                      "sequence_melody": ["COR_FRAGMENT_NOTES", "COR_permute_note_pitch"],
                                      "bar_mask": ["COR_DURATION_MASK"], 
                                      "duration_mask": ["COR_DURATION_MASK"]
                                      }

    all_similarity_transformations = {"contract_melody": ["COR_same_note_modification", "COR_melodic_addition"],
                                        "expand_melody": ["COR_melodic_stripping"],
                                        "reduce_melody": ["COR_same_note_modification", "COR_melodic_addition"],
                                        "permute_melody_rhythm": ["COR_permute_note_duration"],
                                        "sequence_melody": ["COR_FRAGMENT_NOTES", "COR_permute_note_pitch"],
                                        "retrograde_melody_pitch_rhythm": ["COR_incorrect_inversion", "COR_permute_note_pitch_duration"],
                                        "retrograde_melody_pitch": ["COR_incorrect_inversion"],
                                        "invert_melody_strict": ["COR_incorrect_inversion", "COR_incorrect_transposition"],
                                        "permute_melody_pitch": ["COR_permute_note_pitch", "COR_incorrect_inversion"],
                                        "permute_melody_pitch_rhythm": ["COR_permute_note_pitch_duration"],
                                        "permute_melody_new_pitch": ["COR_permute_note_pitch", "COR_incorrect_inversion", "COR_incorrect_transposition"],
                                        "invert_melody_tonal": ["COR_incorrect_inversion"],
                                        "pitch_mask": ["COR_PITCH_MASK"],
                                        "bar_mask": ["COR_DURATION_MASK"],
                                        "duration_mask": ["COR_DURATION_MASK"]
                                        }

    if transformation is not None:
        transformation_names, symmetric_corruptions = [], []
        if transformation == "pitch_mask":
            transformed_phrase = phrase_corruption_obj.masking(phrase, mask_type="pitch")
            transformation_names.append(phrase_corruption_obj.masking.__name__)
            symmetric_corruptions.append("COR_PITCH_MASK")
        elif transformation == "bar_mask":
            transformed_phrase = phrase_corruption_obj.masking(phrase, mask_type="bar")
            transformation_names.append(phrase_corruption_obj.masking.__name__)
            symmetric_corruptions.append("COR_BAR_MASK")
        elif transformation == "duration_mask":
            transformed_phrase = phrase_corruption_obj.masking(phrase, mask_type="duration")
            transformation_names.append(phrase_corruption_obj.masking.__name__)
            symmetric_corruptions.append("COR_DURATION_MASK")
        else:
            # Add fragmentation
            if len(phrase) > 2 and random.random() < 0.5 and allow_fragment:
                # Choose randomly between bar and random_crop
                fragment_type = random.choice(["bar", "random_crop"])
                phrase = phrase_corruption_obj.fragment_notes(phrase, strategy=fragment_type)
                transformation_names.append(phrase_corruption_obj.fragment_notes.__name__)
                symmetric_corruptions.append("COR_FRAGMENT_NOTES")

            transformation_function = getattr(melodic_development_obj, transformation)
            transformed_phrase = transformation_function(phrase, major=major_or_minor)
            transformation_names.append(transformation_function.__name__)
            random_corruption = random.choice(all_similarity_transformations[transformation_function.__name__])
            symmetric_corruptions.append(random_corruption)
        return transformed_phrase, symmetric_corruptions, transformation_names
    
    else:
        transformation_names, symmetric_corruptions = [], []

        # Choose the bar or duration mask transformation
        if random.random() < 0.2 and similarity == "high":
            mask_type = random.choice(["duration", "bar", "first_bar"])
            transformed_phrase = phrase_corruption_obj.masking(phrase, mask_type=mask_type)
            transformation_names.append(phrase_corruption_obj.masking.__name__)
            if mask_type == "duration":
                symmetric_corruptions.append("COR_DURATION_MASK")
            else:
                symmetric_corruptions.append("COR_BAR_MASK")

        # Choose the pitch mask transformation
        elif random.random() < 0.2 and similarity == "low":
            mask_type = "pitch"
            transformed_phrase = phrase_corruption_obj.masking(phrase, mask_type=mask_type)
            transformation_names.append(phrase_corruption_obj.masking.__name__)
            symmetric_corruptions.append("COR_PITCH_MASK")

        # Choose any mask transformation
        elif random.random() < 0.2 and similarity == "all":
            mask_type = random.choice(["duration", "bar", "first_bar", "pitch"])
            transformed_phrase = phrase_corruption_obj.masking(phrase, mask_type=mask_type)
            transformation_names.append(phrase_corruption_obj.masking.__name__)
            if mask_type == "duration":
                symmetric_corruptions.append("COR_DURATION_MASK")
            elif mask_type == "bar" or mask_type == "first_bar":
                symmetric_corruptions.append("COR_BAR_MASK")
            else:
                symmetric_corruptions.append("COR_PITCH_MASK")

        else:
            # Midi to bar encoding
            midi_bar_encoding = melodic_development_obj.group_by_bar(phrase)
            # Fragment the phrase
            if len(midi_bar_encoding) > 2 and random.random() < 0.5 and allow_fragment:
                # Choose randomly between bar and random_crop
                fragment_type = random.choice(["bar", "random_crop"])
                phrase = phrase_corruption_obj.fragment_notes(phrase, strategy=fragment_type)
                transformation_names.append(phrase_corruption_obj.fragment_notes.__name__)
                symmetric_corruptions.append("COR_FRAGMENT_NOTES")

            if similarity == "high":
                # Choose a transformation from high similarity transformations
                transformation_function = random.choice([melodic_development_obj.contract_melody, 
                                                        melodic_development_obj.expand_melody, 
                                                        melodic_development_obj.reduce_melody, 
                                                        melodic_development_obj.permute_melody_rhythm, 
                                                        # melodic_development_obj.sequence_melody
                                                        ])
                transformation_names.append(transformation_function.__name__)
                # Choose a symmetric corruption at random
                random_corruption = random.choice(high_similarity_transformations[transformation_function.__name__])
                symmetric_corruptions.append(random_corruption)

            elif similarity == "low":
                # Choose a transformation from low similarity transformations
                transformation_function = random.choice([melodic_development_obj.retrograde_melody_pitch_rhythm, 
                                                        melodic_development_obj.retrograde_melody_pitch, 
                                                        melodic_development_obj.invert_melody_strict, 
                                                        melodic_development_obj.permute_melody_pitch, 
                                                        melodic_development_obj.permute_melody_pitch_rhythm, 
                                                        # melodic_development_obj.permute_melody_new_pitch, 
                                                        melodic_development_obj.invert_melody_tonal])
                transformation_names.append(transformation_function.__name__)
                # Choose a symmetric corruption at random
                random_corruption = random.choice(low_similarity_transformations[transformation_function.__name__])
                symmetric_corruptions.append(random_corruption)

            elif similarity == "all":
                # Choose a transformation from all similarity transformations
                transformation_function = random.choice([melodic_development_obj.contract_melody, 
                                                        melodic_development_obj.expand_melody, 
                                                        melodic_development_obj.reduce_melody, 
                                                        melodic_development_obj.permute_melody_rhythm, 
                                                        # melodic_development_obj.sequence_melody, 
                                                        melodic_development_obj.retrograde_melody_pitch_rhythm, 
                                                        melodic_development_obj.retrograde_melody_pitch, 
                                                        melodic_development_obj.invert_melody_strict, 
                                                        melodic_development_obj.permute_melody_pitch, 
                                                        melodic_development_obj.permute_melody_pitch_rhythm, 
                                                        # melodic_development_obj.permute_melody_new_pitch, 
                                                        melodic_development_obj.invert_melody_tonal])
                transformation_names.append(transformation_function.__name__)
                # Choose a symmetric corruption at random
                random_corruption = random.choice(all_similarity_transformations[transformation_function.__name__])
                symmetric_corruptions.append(random_corruption)

            # Apply the transformation
            transformed_phrase = transformation_function(phrase, major=major_or_minor)

        return transformed_phrase, symmetric_corruptions, transformation_names


# Function to refine phrase using phrase refinement model
def refine_phrase(model, phrase_1, phrase_2, corruption_tokens, tempo_location, gen_length, meta_data, last_phrase=False, temperature=1.0):
    cadence = "CA_True" if last_phrase else "CA_False"
    major_or_minor, key_signature, time_signature = meta_data[0], meta_data[1], meta_data[2]
    # Add phrase 1 to phrase 2
    if gen_length is not None:
        phrase = phrase_1 + ["SEP"] + [key_signature] + [major_or_minor] + [cadence] + [f"PL_{gen_length}"] + corruption_tokens + ["SEP"] + phrase_2
    else:
        phrase = phrase_1 + ["SEP"] + [key_signature] + [major_or_minor] + [cadence] + corruption_tokens + ["SEP"] + phrase_2
    # phrase = phrase_1 + ["SEP"] + [major_or_minor] + [cadence] + corruption_tokens + ["SEP"] + phrase_2
    # List to remi encoding
    phrase = list_to_remi_encoding(phrase, tempo_location, time_signature)
    # Add the BOS and EOS tokens to the phrase
    phrase = ["BOS"] + phrase + ["EOS"]
    # Tokenize the phrase
    phrase = [dec_tokenizer[note] for note in phrase if note in dec_tokenizer]

    # Generate a phrase
    input_ids = torch.tensor(phrase).unsqueeze(0).to("cuda" if cuda_available() else "cpu")

    generated = model.generate(input_ids, decoder_start_token_id=dec_tokenizer["BOS"], num_beams=1, do_sample=True, max_length=refinement_max_sequence_length, early_stopping=False, temperature=temperature)

    # Get the generated phrase from the tokens
    generated_phrase = [reverse_dec_tokenizer[str(token)] for token in generated[0].tolist()]
    # Remove special tokens
    special_tokens = ["PAD", "BOS", "EOS", "SEP"] + [i for i in dec_tokenizer.keys() if i.startswith("PL_") or i.startswith("PP_") or i.startswith("COR_")]
    generated_phrase = [token for token in generated_phrase if token not in special_tokens]
    generated_phrase = string_to_list_encoding(generated_phrase)

    return generated_phrase

# Function to generate new phrase using phrase generation model
def generate_phrase(model, context, tempo_location, gen_length, gen_pitch_range, meta_data, last_phrase=False, temperature=1.0):
    cadence = "CA_True" if last_phrase else "CA_False"
    major_or_minor, key_signature, time_signature = meta_data[0], meta_data[1], meta_data[2]
    # List to remi encoding
    phrase = list_to_remi_encoding(context, tempo_location, time_signature)
    # Add the BOS and EOS tokens to the phrase
    # [f"PR_{gen_pitch_range}"]
    if gen_length is not None:
        phrase = ["BOS"] + phrase + ["SEP"] + [major_or_minor] + [key_signature] + [f"PL_{gen_length}"] + [cadence] + ["SEP"] + ["EOS"]
    else:
        phrase = ["BOS"] + phrase + ["SEP"] + [major_or_minor] + [key_signature] + [cadence] + ["SEP"] + ["EOS"]
    # phrase = ["BOS"] + phrase + ["SEP"] + [major_or_minor] + [key_signature] + [cadence] + ["SEP"] + ["EOS"]
    # Tokenize the phrase
    phrase = [dec_tokenizer[note] for note in phrase if note in dec_tokenizer]

    # Generate a phrase
    input_ids = torch.tensor(phrase).unsqueeze(0).to("cuda" if cuda_available() else "cpu")
    generated = model.generate(input_ids, decoder_start_token_id=dec_tokenizer["BOS"], num_beams=1, do_sample=True, max_length=generation_max_sequence_length, early_stopping=False, temperature=temperature)

    # Get the generated phrase from the tokens
    generated_phrase = [reverse_dec_tokenizer[str(token)] for token in generated[0].tolist()]
    # Remove special tokens
    special_tokens = ["PAD", "BOS", "EOS", "SEP"] + [i for i in dec_tokenizer.keys() if i.startswith("PL_") or i.startswith("PP_") or i.startswith("COR_")]
    generated_phrase = [token for token in generated_phrase if token not in special_tokens]
    generated_phrase = string_to_list_encoding(generated_phrase)

    return generated_phrase

# Function to select phrase using phrase selection model
def select_phrase(melodic_development_obj, time_signature, model, phrase_1, phrase_2, reset_bar_index=0):

    # Reindex the phrase based on last bar
    phrase_2 = melodic_development_obj.reindex_bars(phrase_2, start_bar = reset_bar_index)

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

    output = model(input_ids)
    logits = output.logits
    # Get the probability of the phrase as sigmoid of the logits
    prob = F.sigmoid(logits)
    prob = prob[-1, -1].item()

    return prob

def generate_best_phrase(melodic_development_obj, time_signature, phrase_generation_model, phrase_selection_model, previous_phrase, context, tempo_location, gen_length, gen_pitch_range, meta_data, last_phrase=False):
    list_of_generated_phrases = []
    for _ in range(5):
        temperature = random.choice([0.7, 0.8, 0.9, 0.95, 1.0])
        generated_phrase = generate_phrase(phrase_generation_model, context, tempo_location, gen_length, gen_pitch_range, meta_data, last_phrase, temperature)
        list_of_generated_phrases.append(generated_phrase)

    # Get the pitch range of the phrases
    # pitch_ranges = get_pitch_range(list_of_generated_phrases)

    # Choose the phrase with the highest pitch range
    # best_phrase = list_of_generated_phrases[pitch_ranges.index(max(pitch_ranges))]

    # Get the probability of the phrase
    list_of_probs = []
    for phrase in list_of_generated_phrases:
        prob = select_phrase(melodic_development_obj, time_signature, phrase_selection_model, previous_phrase, phrase)
        list_of_probs.append(prob)

    # Choose the phrase with the highest probability
    best_phrase = list_of_generated_phrases[list_of_probs.index(max(list_of_probs))]

    # Print the probability of the phrase
    print("Probabilities: ", list_of_probs)

    return best_phrase

def refine_best_phrase(melodic_development_obj, time_signature, phrase_refiner_model, previous_phrase, context, corruption_tokens, tempo_location, gen_length, gen_pitch_range, meta_data, last_phrase=False):
    list_of_generated_phrases = []
    for _ in range(5):
        temperature = random.choice([0.7, 0.8, 0.9, 0.95, 1.0])
        generated_phrase = refine_phrase(phrase_refiner_model, previous_phrase, context, corruption_tokens, tempo_location, gen_length, meta_data, last_phrase, temperature)
        list_of_generated_phrases.append(generated_phrase)

    # Get the probability of the phrase
    list_of_probs = []
    for phrase in list_of_generated_phrases:
        prob = select_phrase(melodic_development_obj, time_signature, phrase_selection_model, previous_phrase, phrase)
        list_of_probs.append(prob)

    # Choose the phrase with the highest probability
    best_phrase = list_of_generated_phrases[list_of_probs.index(max(list_of_probs))]

    # Print the probability of the phrase
    print("Probabilities: ", list_of_probs)
    
    return best_phrase

def fix_bar_onset(melodic_development_obj, structure, generated_phrase, new_section=False):
    # Get the section number and phrase number of the last phrase in the structure that is not empty
    section_number, phrase_number = 0, 0
    for section in structure['sections']:
        for phrase in structure['sections'][section]:
            if structure['sections'][section][phrase] != []:
                section_number = int(section.split("_")[1])
                phrase_number = int(phrase.split("_")[1])

    # Get the bar number and onset of the last element in previous phrase
    bar_number, onset, duration = structure['sections'][f'section_{section_number}'][f'phrase_{phrase_number}'][-1][0], structure['sections'][f'section_{section_number}'][f'phrase_{phrase_number}'][-1][1], structure['sections'][f'section_{section_number}'][f'phrase_{phrase_number}'][-1][4]
    # Get the bar number and onset of the first element in the generated phrase
    bar_number_gen, onset_gen = generated_phrase[0][0], generated_phrase[0][1]
    if new_section:
        onset_gen = onset + duration + 1
        bar_number_gen = bar_number
        if onset_gen >= melodic_development_obj.beats_in_bar:
            # Onset_gen should be remainder of onset_gen and beats_in_bar
            onset_gen = onset_gen % melodic_development_obj.beats_in_bar
            # Get quotient of onset_gen and beats_in_bar
            quotient = (onset + duration + 1) // melodic_development_obj.beats_in_bar
            bar_number_gen = bar_number + quotient
    elif onset_gen < onset + duration:
        onset_gen = onset + duration #+ 1
        bar_number_gen = bar_number
        if onset_gen >= melodic_development_obj.beats_in_bar:
            if duration < 1:
                # onset_gen = 1
                onset_gen = onset_gen % melodic_development_obj.beats_in_bar
                bar_number_gen = bar_number + 1
            else:
                # onset_gen = 0
                onset_gen = onset + duration
                if onset_gen >= melodic_development_obj.beats_in_bar:
                    onset_gen = onset_gen % melodic_development_obj.beats_in_bar
                    bar_number_gen = bar_number + 1
                else:
                    bar_number_gen = bar_number
                # bar_number_gen = bar_number + 1
    else:
        bar_number_gen = bar_number
    # Fix the bar number and onset of the generated phrase based on the previous phrase
    generated_phrase = melodic_development_obj.fix_bars(generated_phrase, onset_gen, bar_number_gen)
    return generated_phrase

def flatten_structure(structure, section_number):
    # Assemble the structure into a single list
    flat_structure = []
    for section in structure['sections'].keys():
        for phrase in structure['sections'][section].keys():
            if section == f'section_{section_number}':
                flat_structure += structure['sections'][section][phrase]
    return flat_structure

def get_note_density(section):
    note_density = 0
    for phrase in section:
        note_density += len(phrase)
    return note_density

def get_pitch_range(phrases: list):
    pitch_ranges = []
    for phrase in phrases:
        phrase = [note[3] for note in phrase]
        pitch_ranges.append(max(phrase) - min(phrase))
    return pitch_ranges

def get_average_note_duration(phrases):
    note_durations = 0
    note_count = 0
    for phrase in phrases:
        for note in phrase:
            note_durations += note[4]
            note_count += 1
    note_durations /= note_count
    return note_durations

def get_average_pitch_value(phrase):
    pitch_values = 0
    note_count = 0
    for note in phrase:
        pitch_values += note[3]
        note_count += 1
    pitch_values /= note_count
    return pitch_values

def transpose_key(current_key, semitones):
    keys = ["KS_A-", "KS_A", "KS_B-", "KS_B", "KS_C", "KS_D-", "KS_D", "KS_E-", "KS_E", "KS_F", "KS_F#", "KS_G"]

    # Find the index of the current key in the list
    current_index = keys.index(current_key)

    # Calculate the new index after transposing by the given semitones
    new_index = (current_index + semitones) % len(keys)

    # Return the new key
    return keys[new_index]


def generate(generation_configs, test_file, phrase_refiner_model, phrase_generation_model, phrase_selection_model):

    use_phrase_selection = generation_configs['use_phrase_selection']
    print("Using phrase selection: ", use_phrase_selection)
    use_velocity = generation_configs['use_velocity']
    print("Using velocity: ", use_velocity)
    transformations = generation_configs['transformations']
    print("Transformations: ", transformations)
    allow_modulation = generation_configs['allow_modulation']
    print("Allow modulation: ", allow_modulation)
    phrase_generation_frequency = generation_configs['ratio']['phrase_generation_frequency']
    phrase_refinement_frequency = generation_configs['ratio']['phrase_refinement_frequency']
    motif_repetition_frequency = generation_configs['ratio']['motif_repetition_frequency']
    structure_string = generation_configs['structure'] # e.g. "AABAC"
    phrases_per_section = generation_configs['phrases_per_section'] # e.g. [3, 5, 4, 2, 5]
    print("Structure string: ", structure_string, " Phrases per section: ", phrases_per_section)
    structure = {'sections': {}}
    for i, section in enumerate(structure_string):
        structure['sections'][f"section_{i}"] = {f"phrase_{j}": [] for j in range(phrases_per_section[i])}

    # Read test file as json
    with open(os.path.join(configs['raw_data']['json_folder'], test_file), "r") as f:
        test_phrases = json.load(f)
    
    tempo_location = test_phrases['metadata']['tempo']
    key_signature = test_phrases['metadata']['key_signature']
    major_or_minor = test_phrases['metadata']['major_or_minor']
    time_signature = test_phrases['metadata']['time_signature']
    print("Tempo: ", tempo_location, " Key signature: ", key_signature, " Major or minor: ", major_or_minor, " Time signature: ", time_signature)
    beats_in_bar = find_beats_in_bar(time_signature)
    # Load the transformation and phrase corruption class
    melodic_development_obj = Melodic_Development(beats_in_bar=beats_in_bar)
    phrase_corruption_obj = Phrase_Corruption(beats_in_bar=beats_in_bar)


    # Generate the structure
    for section_number, section in enumerate(structure['sections']):
        # If section number is 0, choose motif as the first phrase
        if section_number == 0:
            motif, motif_position = test_phrases['phrases']['0'][0], test_phrases['phrases']['0'][1]
        # Check if next section in structure_string has appeared anywhere before up till now
        if section_number > 0: 
            if structure_string[section_number] in structure_string[:section_number]:
                # Get the key signature of the first phrase for repeating sections
                key_signature = test_phrases['metadata']['key_signature']
                # Choose motif from the that section
                motif = structure['sections'][f'section_{structure_string[:section_number].rfind(structure_string[section_number])}'][f'phrase_0']
                # Sequence the motif to be an octave higher or lower
                avg_pitch_value = get_average_pitch_value(motif)
                if random.random() < 0.5:
                    if avg_pitch_value < 60:
                        motif = melodic_development_obj.sequence_melody(motif, pitch_change=12)
                        print("Taking motif from section number: ", structure_string[:section_number].rfind(structure_string[section_number]), " and sequencing it up by an octave")
                    elif avg_pitch_value > 72:
                        motif = melodic_development_obj.sequence_melody(motif, pitch_change=-12)
                        print("Taking motif from section number: ", structure_string[:section_number].rfind(structure_string[section_number]), " and sequencing it down by an octave")
                    else:
                        # Choose randomly between up and down
                        motif = melodic_development_obj.sequence_melody(motif, pitch_change=random.choice([-12, 12]))
                        print("Taking motif from section number: ", structure_string[:section_number].rfind(structure_string[section_number]), " and sequencing it by an octave")
                motif = fix_bar_onset(melodic_development_obj, structure, motif, new_section=True)
            else:
                # Generate a new motif using the phrase refinement model with low similarity to a previous phrase
                # Get a random phrase from the previous section
                previous_section_phrases = structure['sections'][f'section_{section_number-1}']
                random_phrase_key = random.choice(list(previous_section_phrases.keys()))
                random_phrase_number = int(random_phrase_key.split('_')[1])
                print("\n", "Random phrase chosen from previous section: section number: ", section_number-1, ", phrase number: ", random_phrase_number)
                random_phrase = previous_section_phrases[random_phrase_key]
                # random_phrase = random.choice(list(previous_section_phrases.values()))
                # Length of the phrase
                # phrase_length = len(structure['current_motif'])
                phrase_length = random.randint(9, 16)

                # Get last phrase of the previous section
                previous_phrase = structure['sections'][f'section_{section_number-1}'][f'phrase_{phrases_per_section[section_number-1]-1}'] 
                print("Generating motif for section number: ", section_number, " from previous section number: ", section_number-1)

                # Get the last note of the last bar of the previous phrase
                one_note_phrase = melodic_development_obj.group_by_bar(previous_phrase)
                # one_note_phrase = [one_note_phrase[-1][-1]]
                one_note_phrase = one_note_phrase[-1]
                        
                # Get the bar index of the last note in the phrase
                duration = previous_phrase[-1][4]
                bar_number = previous_phrase[-1][0]
                onset = previous_phrase[-1][1]
                if onset + duration >= melodic_development_obj.beats_in_bar:
                    reset_bar_index = bar_number + 1
                else:
                    reset_bar_index = bar_number
                # reset_bar_index = previous_phrase[-1][0] #+ 1
                if allow_modulation:
                    # Get a new key signature that is either 4, 5 or 7 semitones higher
                    key_signature = transpose_key(key_signature, random.choice([4, 5, 7]))
                    print("New key signature: ", key_signature)
                # Get the transformation if any
                if len(transformations)>0:
                    transformation = transformations.pop(0)
                else:
                    transformation = None
                # Transform the random phrase
                transformed_motif, corruption_tokens, transformation_names = transform_phrase(melodic_development_obj, phrase_corruption_obj, random_phrase, major_or_minor, key_signature, "low", transformation, allow_fragment=False, reset_bar_index=reset_bar_index)
                print("Creating new section number: ", section_number, "Corruption tokens: ", corruption_tokens, " Transformation names: ", transformation_names)
                
                motif = refine_phrase(phrase_refiner_model, one_note_phrase, transformed_motif, corruption_tokens, tempo_location, phrase_length, meta_data=[major_or_minor, key_signature, time_signature], last_phrase=True)
                motif = fix_bar_onset(melodic_development_obj, structure, motif, new_section=True)
                print("Added motif to section number: ", section_number)
            
        i = 1
        generate_count = 0
        transform_count = 0
        while i <= phrases_per_section[section_number]:
            if i == 1:
                # Add the motif to the structure dictionary
                structure['sections'][f'section_{section_number}']['phrase_0'] = motif
                structure['current_motif'] = motif
                print("Phrase number: ", i, " in section number: ", section_number, " is the motif")
                i += 1
            else:
                # Check if phrase is the last phrase in the section
                if i == phrases_per_section[section_number]:
                    is_last_phrase = True
                else:
                    is_last_phrase = False

                if i % motif_repetition_frequency == 0:
                    print("Choosing motif to generate phrase number: ", i, " in section number: ", section_number)
                    generated_phrase = structure['current_motif']
                    generated_phrase = fix_bar_onset(melodic_development_obj, structure, generated_phrase)
                    structure['sections'][f'section_{section_number}'][f'phrase_{i-1}'] = generated_phrase
                    i+=1

                for _ in range(phrase_generation_frequency):
                    if i % motif_repetition_frequency == 0:
                        print("Choosing motif to generate phrase number: ", i, " in section number: ", section_number)
                        generated_phrase = structure['current_motif']
                        generated_phrase = fix_bar_onset(melodic_development_obj, structure, generated_phrase)
                        structure['sections'][f'section_{section_number}'][f'phrase_{i-1}'] = generated_phrase
                        i+=1
                    if i <= phrases_per_section[section_number]:
                        print("Generating phrase number: ", i, " in section number: ", section_number)
                        previous_phrase = structure['sections'][f'section_{section_number}'][f'phrase_{i-2}']
                        generation_length = len(previous_phrase) #max(random.randint(6, 12), len(previous_phrase))
                        # generation_length = None
                        previous_pitch_range = max([note[3] for note in previous_phrase]) - min([note[3] for note in previous_phrase])
                        generation_pitch_range = max(random.choice([i for i in range(8, 16)]), previous_pitch_range)
                        context = flatten_structure(structure, section_number)
                        if use_phrase_selection:                        
                            generated_phrase = generate_best_phrase(melodic_development_obj, time_signature, phrase_generation_model, phrase_selection_model, previous_phrase, context, tempo_location, generation_length, generation_pitch_range, meta_data=[major_or_minor, key_signature, time_signature], last_phrase=is_last_phrase)
                        else:
                            generated_phrase = generate_phrase(phrase_generation_model, context, tempo_location, generation_length, previous_pitch_range, meta_data=[major_or_minor, key_signature, time_signature], last_phrase=is_last_phrase)
                        generated_phrase = fix_bar_onset(melodic_development_obj, structure, generated_phrase)
                        structure['sections'][f'section_{section_number}'][f'phrase_{i-1}'] = generated_phrase

                        generate_count += 1
                        i += 1

                for _ in range(phrase_refinement_frequency):
                    if i % motif_repetition_frequency == 0:
                        print("Choosing motif to generate phrase number: ", i, " in section number: ", section_number)
                        generated_phrase = structure['current_motif']
                        generated_phrase = fix_bar_onset(melodic_development_obj, structure, generated_phrase)
                        structure['sections'][f'section_{section_number}'][f'phrase_{i-1}'] = generated_phrase
                        i+=1
                    if i <= phrases_per_section[section_number]:
                        # Get a copy of motif as context phrase
                        context_phrase = copy.deepcopy(structure['current_motif'])
                        # Get the bar index of the last note in the previous phrase
                        previous_phrase = structure['sections'][f'section_{section_number}'][f'phrase_{i-2}']
                        # generation_length = len(previous_phrase)
                        generation_length = None
                        reset_bar_index = previous_phrase[-1][0] #+ 1
                        # Get the transformation if any
                        if len(transformations)>0:
                            transformation = transformations.pop(0)
                        else:
                            transformation = None
                        # Transform the previous phrase
                        transformed_phrase, corruption_tokens, transformation_names = transform_phrase(melodic_development_obj, phrase_corruption_obj, context_phrase, major_or_minor, key_signature, "high", transformation, allow_fragment=True, reset_bar_index=reset_bar_index)
                        if not any('MASK' in s for s in corruption_tokens):
                            generated_phrase = fix_bar_onset(melodic_development_obj, structure, transformed_phrase)
                        print("Transforming motif", " in section number: ", section_number, " to generate phrase number: ", i, " in section number: ", section_number)
                        print("Corruption tokens: ", corruption_tokens, " Transformation names: ", transformation_names)
                        # Refine the phrase
                        previous_pitch_range = max([note[3] for note in previous_phrase]) - min([note[3] for note in previous_phrase])
                        generation_pitch_range = max(random.choice([i for i in range(8, 16)]), previous_pitch_range)
                        if use_phrase_selection:
                            generated_phrase = refine_best_phrase(melodic_development_obj, time_signature, phrase_refiner_model, previous_phrase, transformed_phrase, corruption_tokens, tempo_location, generation_length, generation_pitch_range, meta_data=[major_or_minor, key_signature, time_signature], last_phrase=is_last_phrase)
                        else:
                            generated_phrase = refine_phrase(phrase_refiner_model, previous_phrase, transformed_phrase, corruption_tokens, tempo_location, generation_length, meta_data=[major_or_minor, key_signature, time_signature], last_phrase=is_last_phrase)
                        generated_phrase = fix_bar_onset(melodic_development_obj, structure, generated_phrase)
                        structure['sections'][f'section_{section_number}'][f'phrase_{i-1}'] = generated_phrase
                        
                        transform_count += 1
                        i += 1

    # Assemble the structure into a single list
    structure_midi = []
    for section in structure['sections'].keys():
        for phrase_number, phrase in enumerate(structure['sections'][section].keys()):
            bar_number = int(structure['sections'][section][phrase][0][0])
            # Add tempo changes
            if section == "section_0" and phrase == "phrase_0" and bar_number not in tempo_location.keys():
                tempo_location[bar_number] = 120
                # Get average note duration
                average_note_duration = get_average_note_duration(structure['sections'][section].values())
                print("Average note duration: ", average_note_duration)
            elif section != "section_0" and phrase == "phrase_0" and bar_number not in tempo_location.keys():
                # Get average note duration
                average_note_duration = get_average_note_duration(structure['sections'][section].values())
                print("Average note duration: ", average_note_duration)
                # if average_note_duration >= 0.75:
                #     tempo_location[bar_number] = 130
                # elif average_note_duration <= 0.5:
                #     tempo_location[bar_number] = 110
                # else:
                #     tempo_location[bar_number] = random.choice([90, 100, 110, 120, 130])
            if use_velocity:
                # Check if phrase = 0 
                if phrase_number == 0:
                    # Check how many notes are in the first phrase of the section
                    note_density = len(structure['sections'][section][phrase])
                    # Get current velocity
                    current_velocity = structure['sections'][section][phrase][0][5]
                    # Choose random velocity between 60 and 110
                    velocity_target = random.choice([80, 90, 100])
                    velocity_factor = int((velocity_target - current_velocity) / note_density)
                for note in structure['sections'][section][phrase]:
                    # Increment or decrement it by the new velocity / number of notes in the phrase
                    velocity = current_velocity + velocity_factor
                    note[5] = velocity
                    if current_velocity != velocity_target:
                        current_velocity += velocity_factor

            structure_midi += structure['sections'][section][phrase]

    # Fix the bar numbers and onsets of the structure
    # structure_midi = melodic_development_obj.fix_bars(structure_midi)

    # Create an output folder if it doesn't exist
    output_folder = "output/yin_yang"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_filepath = os.path.join(output_folder, test_file.split(".")[0] + ".mid")

    # Write the structure to a MIDI file
    encoding_to_midi(structure_midi, tempo_location, time_signature, output_filepath)


if __name__ == "__main__":
    
    # Assemble phrases based on structure in generation_configs 
    generation_configs = configs['generation']
    generate_all = generation_configs['generate_all']
    test_filepath = generation_configs['test_filepath']

    # Load test file list
    with open(os.path.join(artifact_folder, "test_file_list.json"), "r") as f:
        test_file_list = json.load(f)

    # # test_file = "NLB183423_01.json" #"NLB145289_01.json" # #random.choice(test_file_list)
    # print("Test file: ", test_file)
    if generate_all:
        for test_file in test_file_list:
            while True:
                try:
                    generate(generation_configs, test_file, phrase_refiner_model, phrase_generation_model, phrase_selection_model)
                    print(f"Generated: {test_file}")
                    break
                except Exception as e:
                    print(f"Error generating: {test_file}")
                    print(f"Error message: {str(e)}")
                    continue
        print("All files generated successfully!")
    else:
        # Choose random test file from test_file_list if test_filepath is not provided
        if test_filepath == "" or test_filepath is None:
            test_file = random.choice(test_file_list)
        else:
            test_file = test_filepath

        print("Test file: ", test_file)

        generate(generation_configs, test_file, phrase_refiner_model, phrase_generation_model, phrase_selection_model)