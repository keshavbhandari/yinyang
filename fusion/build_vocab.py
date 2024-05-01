import yaml
import os
import argparse
import pickle

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=os.path.normpath("/homes/kb658/yinyang/configs/configs_os.yaml"),
                    help="Path to the config file")
args = parser.parse_args()

# Load config file
with open(args.config, 'r') as f:
    configs = yaml.safe_load(f)

artifact_folder = configs["raw_data"]["artifact_folder"]
mono_folder = configs["raw_data"]["mono_folder"]
json_folder = configs["raw_data"]["json_folder"]
raw_data_folders = configs["raw_data"]["raw_data_folders"]
data_path = raw_data_folders['classical']['folder_path']

# Build the vocabulary
vocab = {}
# MIDI velocity range from 0 to 127
velocity = [0, 15, 30, 45, 60, 75, 90, 105, 120, 127]
# MIDI pitch range from 0 to 127
midi_pitch = list(range(0, 128))
# Onsets are quantized in 10 milliseconds up to 5 seconds
onset = list(range(0, 5001, 10))
duration = list(range(0, 5001, 10))

# Add the tokens to the vocabulary
for v in velocity:
    for p in midi_pitch:
        vocab[("piano", p, v)] = len(vocab)
for o in onset:
    vocab[("onset", o)] = len(vocab)
for d in duration:
    vocab[("dur", d)] = len(vocab)

# Special tokens
vocab[('prefix', 'instrument', 'piano')] = len(vocab)
vocab["<T>"] = len(vocab)
vocab["<D>"] = len(vocab)
vocab["<S>"] = len(vocab)
vocab["<E>"] = len(vocab)
vocab["SEP"] = len(vocab)

# Print the vocabulary length
print(f"Vocabulary length: {len(vocab)}")

# Save the vocabulary
# Directory path
tokenizer_folder = os.path.join(artifact_folder, "fusion")
# Make directory if it doesn't exist
os.makedirs(tokenizer_folder, exist_ok=True)
vocab_path = os.path.join(tokenizer_folder, "vocab.json")
with open(os.path.join(tokenizer_folder, "vocab.pkl"), 'wb') as f:
    pickle.dump(vocab, f)
