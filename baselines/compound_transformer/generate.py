import torch
from torch.cuda import is_available as cuda_available
import yaml
import json
import os
import argparse
import random
import numpy as np
from cp_model import LinearAttentionTransformerLM
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Append the path to the current working directory
sys.path.append(os.getcwd())
from utils.utils import find_beats_in_bar, list_to_cp_encoding, cp_to_list_encoding, encoding_to_midi
from phrase_refiner.transformations import Melodic_Development

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=os.path.normpath("configs/configs_os.yaml"),
                    help="Path to the config file")
args = parser.parse_args()

# Load config file
with open(args.config, 'r') as f:
    configs = yaml.safe_load(f)

# Artifact folder
artifact_folder = configs['raw_data']['artifact_folder']

# Load tokenizer json file dictionary
tokenizer_filepath = os.path.join(artifact_folder, 'cp_tokenizer.json')
with open(tokenizer_filepath, 'r') as f:
    tokenizer = json.load(f)

reverse_dec_tokenizer = {}
for key, value in tokenizer.items():
    reverse_dec_tokenizer[key] = {v: k for k, v in value.items()}

# Get the vocab size
vocab_size = [len(tokenizer["time_signature_tokenizer"])+1, len(tokenizer["chord_tokenizer"])+1, len(tokenizer["metric_tokenizer"])+1, len(tokenizer["family_tokenizer"])+1, len(tokenizer["pitch_tokenizer"])+1, len(tokenizer["duration_tokenizer"])+1, len(tokenizer["velocity_tokenizer"])+1]
print("Vocab size: ", vocab_size)

# Load the model
model = LinearAttentionTransformerLM(
    num_tokens = vocab_size,
    dim = 512,
    heads = 4,
    depth = 4,
    max_seq_len = 2048,
    causal = True,
    ff_dropout = 0,
    attn_layer_dropout = 0,
    attn_dropout = 0,
    emb_dim = 512,
    dim_head = 128,
    blindspot_size = 64,
    n_local_attn_heads = 4,
    local_attn_window_size = 128,
    reversible = True,
    ff_chunks = 2,
    ff_glu = True,
    attend_axially = False,
    shift_tokens = True
)

device=0
# Load the state dictionary from the .bin file
model_state_dict = torch.load('artifacts/cp_transformer/pytorch_model.bin') #torch.device(device)

# Load the state dictionary into the model
model.load_state_dict(model_state_dict)

model.eval()
model.to("cuda" if cuda_available() else "cpu")


# Load test file list
with open(os.path.join(artifact_folder, "test_file_list.json"), "r") as f:
    test_file_list = json.load(f)


def generate(test_file, configs, model, gen_length=256):
    # Read test file as json
    with open(os.path.join(configs['raw_data']['json_folder'], test_file), "r") as f:
        test_phrases = json.load(f)
        
    tempo_location = test_phrases['metadata']['tempo']
    key_signature = test_phrases['metadata']['key_signature']
    time_signature = test_phrases['metadata']['time_signature']
    beats_in_bar = find_beats_in_bar(time_signature)

    # Get the first phrase from test file
    motif = test_phrases['phrases']['0'][0]

    # Convert motif to cp encoding
    cp_encoding = list_to_cp_encoding(motif, tempo_location, time_signature)

    # Tokenize cp encoding
    BOS = ["BOS", "Ignore_None", "Ignore_None", "Ignore_None", "Ignore_None", "Ignore_None", "Ignore_None"]
    cp_encoding = [BOS] + cp_encoding

    # Convert cp encoding to tensor
    family_tensor = []
    metric_tensor = []
    pitch_tensor = []
    velocity_tensor = []
    duration_tensor = []
    chord_tensor = []
    time_signature_tensor = []

    for event in cp_encoding:
        family_tensor.append(tokenizer["family_tokenizer"][event[0]])
        metric_tensor.append(tokenizer["metric_tokenizer"][event[1]])
        pitch_tensor.append(tokenizer["pitch_tokenizer"][event[2]])
        velocity_tensor.append(tokenizer["velocity_tokenizer"][event[3]])
        duration_tensor.append(tokenizer["duration_tokenizer"][event[4]])
        chord_tensor.append(tokenizer["chord_tokenizer"][event[5]])
        time_signature_tensor.append(tokenizer["time_signature_tokenizer"][event[6]])

    # Create tensors from the lists
    family_tensor = torch.tensor(family_tensor)
    metric_tensor = torch.tensor(metric_tensor)
    pitch_tensor = torch.tensor(pitch_tensor)
    velocity_tensor = torch.tensor(velocity_tensor)
    duration_tensor = torch.tensor(duration_tensor)
    chord_tensor = torch.tensor(chord_tensor)
    time_signature_tensor = torch.tensor(time_signature_tensor)

    # Length of the tensor
    input_length = len(family_tensor)
    print("Length of family tensor: ", input_length)

    # Pad the tensors to the maximum sequence length
    if len(family_tensor) < 2048:
        family_tensor = torch.cat((family_tensor, torch.zeros(2048 - len(family_tensor)).long()))
        metric_tensor = torch.cat((metric_tensor, torch.zeros(2048 - len(metric_tensor)).long()))
        pitch_tensor = torch.cat((pitch_tensor, torch.zeros(2048 - len(pitch_tensor)).long()))
        velocity_tensor = torch.cat((velocity_tensor, torch.zeros(2048 - len(velocity_tensor)).long()))
        duration_tensor = torch.cat((duration_tensor, torch.zeros(2048 - len(duration_tensor)).long()))
        chord_tensor = torch.cat((chord_tensor, torch.zeros(2048 - len(chord_tensor)).long()))
        time_signature_tensor = torch.cat((time_signature_tensor, torch.zeros(2048 - len(time_signature_tensor)).long()))


    # Concatenate the tensors
    input_data = torch.cat((time_signature_tensor.unsqueeze(0), chord_tensor.unsqueeze(0), metric_tensor.unsqueeze(0), family_tensor.unsqueeze(0), pitch_tensor.unsqueeze(0), duration_tensor.unsqueeze(0), velocity_tensor.unsqueeze(0)), dim=0)
    # Switch the dimensions
    input_data = input_data.permute(1, 0)

    # Add batch dimension
    input_data = input_data.unsqueeze(0)

    # Move to cuda
    input_data = input_data.to("cuda" if cuda_available() else "cpu")

    # Generate the continuation
    with torch.no_grad():
        final_res = []
        # Append input_data to final_res as an array after removing the batch dimension
        final_res.append(input_data[0, :input_length, :].cpu().numpy())

        h, y_type = model.forward_hidden(input_data, is_training=True)

        while True:
            # Get the time step corresponding to the length of the input before padding
            h = h[:, input_length-1, :]
            y_type = y_type[:, input_length-1, :]

            # sample others
            next_arr = model.forward_output_sampling(h, y_type, intervene=False)
            if reverse_dec_tokenizer['family_tokenizer'][next_arr[3]] == 'EOS' and input_length < gen_length:
                next_arr = model.forward_output_sampling(h, y_type, intervene=True)

            final_res.append(next_arr[None, ...])

            # forward
            input_ = torch.from_numpy(next_arr).long().to("cuda" if cuda_available() else "cpu")
            input_  = input_.unsqueeze(0).unsqueeze(0) # (1, 1, 7)
            # Add input_ to the input_data at the input_length time step before the paddings
            input_data[:, input_length, :] = input_
            input_length += 1
            
            # Do a forward pass again 
            h, y_type = model.forward_hidden(
                input_data, is_training=True)

            # end of sequence
            if reverse_dec_tokenizer['family_tokenizer'][next_arr[3]] == 'EOS' or input_length > gen_length-1:
                break

    print('\n--------[Done]--------')
    final_res = np.concatenate(final_res)
    print(final_res.shape) # (2048, 7)

    # Convert the final_res to a list
    final_res_list = final_res.tolist()

    # Convert the list to cp encoding
    cp_encoding = []
    for event in final_res_list:
        family_event = reverse_dec_tokenizer['family_tokenizer'][event[3]]
        metric_event = reverse_dec_tokenizer['metric_tokenizer'][event[2]]
        pitch_event = reverse_dec_tokenizer['pitch_tokenizer'][event[4]]
        velocity_event = reverse_dec_tokenizer['velocity_tokenizer'][event[6]] if event[6] != 0 else "Ignore_None"
        duration_event = reverse_dec_tokenizer['duration_tokenizer'][event[5]]
        chord_event = reverse_dec_tokenizer['chord_tokenizer'][event[1]] if event[1] != 0 else "Ignore_None"
        time_signature_event = reverse_dec_tokenizer['time_signature_tokenizer'][event[0]]
        cp_encoding.append([family_event, metric_event, pitch_event, velocity_event, duration_event, chord_event, time_signature_event])

    # Remove the BOS token
    cp_encoding = cp_encoding[1:]
    # Remove the EOS token if it exists
    if cp_encoding[-1][0] == "EOS":
        cp_encoding = cp_encoding[:-1]

    # Convert cp encoding to list encoding
    list_encoding = cp_to_list_encoding(cp_encoding)

    # Fix the bars
    melodic_development_obj = Melodic_Development(beats_in_bar=beats_in_bar)
    list_encoding = melodic_development_obj.fix_bars(list_encoding)

    # Create an output folder if it doesn't exist
    output_folder = "output/compound_word"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_filepath = os.path.join(output_folder, test_file.split(".")[0] + ".mid")

    # Write the structure to a MIDI file
    encoding_to_midi(list_encoding, tempo_location, time_signature, output_filepath)


if __name__ == "__main__":
    for test_file in test_file_list:
        while True:
            try:
                generate(test_file, configs, model, 300)
                print(f"Generated: {test_file}")
                break
            except Exception as e:
                print(f"Error generating: {test_file}")
                print(f"Error message: {str(e)}")
                continue
    print("All files generated successfully!")