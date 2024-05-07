import yaml
import json
import pickle
import os
import random
import sys
import argparse
import torch
from torch.nn import functional as F
from transformers import EncoderDecoderModel
from torch.cuda import is_available as cuda_available
from aria.data.midi import MidiDict
from aria.tokenizer import AbsTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.utils import flatten, skyline


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=os.path.normpath("configs/configs_os.yaml"),
                    help="Path to the config file")
args = parser.parse_args()

# Load config file
with open(args.config, 'r') as f:
    configs = yaml.safe_load(f)
    
artifact_folder = configs["raw_data"]["artifact_folder"]
raw_data_folders = configs["raw_data"]["raw_data_folders"]
data_path = raw_data_folders['classical']['folder_path']
output_folder = os.path.join("/homes/kb658/yinyang/output/fusion/")
# Create output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get the encoder and decoder max sequence length
encoder_max_sequence_length = configs['model']['fusion_model']['encoder_max_sequence_length']
decoder_max_sequence_length = configs['model']['fusion_model']['decoder_max_sequence_length']

# Get tokenizer
tokenizer_filepath = os.path.join(artifact_folder, "fusion", "vocab.pkl")
# Load the tokenizer dictionary
with open(tokenizer_filepath, "rb") as f:
    tokenizer = pickle.load(f)

# Reverse the tokenizer
decode_tokenizer = {v: k for k, v in tokenizer.items()}

# Load the fusion model
fusion_model = EncoderDecoderModel.from_pretrained(os.path.join(artifact_folder, "fusion", "model"))
fusion_model.eval()
fusion_model.to("cuda" if cuda_available() else "cpu")


file_path = os.path.join("/homes/kb658/yinyang/output/yin_yang/deut2034.mid")
# file_path = os.path.join("/import/c4dm-datasets/maestro-v3.0.0/2008/MIDI-Unprocessed_07_R2_2008_01-05_ORIG_MID--AUDIO_07_R2_2008_wav--2.midi")
mid = MidiDict.from_midi(file_path)
aria_tokenizer = AbsTokenizer()
tokenized_sequence = aria_tokenizer.tokenize(mid)
instrument_token = tokenized_sequence[0]
tokenized_sequence = tokenized_sequence[2:-1]
# Call the flatten function
flattened_sequence = flatten(tokenized_sequence)
# Call the skyline function
tokenized_sequence, harmony = skyline(flattened_sequence, diff_threshold=30, static_velocity=True)
tokenized_sequence = [tokenizer[tuple(token)] if isinstance(token, list) else tokenizer[token] for token in tokenized_sequence]
# Pad the sequences
if len(tokenized_sequence) < encoder_max_sequence_length:
    tokenized_sequence = F.pad(torch.tensor(tokenized_sequence), (0, encoder_max_sequence_length - len(tokenized_sequence))).to(torch.int64)
else:
    tokenized_sequence = torch.tensor(tokenized_sequence[0:decoder_max_sequence_length:]).to(torch.int64)

# Generate the sequence
input_ids = tokenized_sequence.unsqueeze(0).to("cuda" if cuda_available() else "cpu")
output = fusion_model.generate(input_ids, decoder_start_token_id=tokenizer["<S>"], max_length=decoder_max_sequence_length, num_beams=1, do_sample=True, early_stopping=False, temperature=1.0)

# Decode the generated sequences
generated_sequences = [decode_tokenizer[token] for token in output[0].tolist()]
# Remove special tokens
generated_sequences = [token for token in generated_sequences if token not in ["<S>", "<E>", "<SEP>"]]
generated_sequences = [instrument_token, "<S>"] + generated_sequences + ["<E>"]

# Print the generated sequences
print("Generated sequences:", generated_sequences)

# Write the generated sequences to a MIDI file
mid_dict = aria_tokenizer.detokenize(generated_sequences)
mid = mid_dict.to_midi()
filename = os.path.basename(file_path)
mid.save(os.path.join(output_folder, filename))