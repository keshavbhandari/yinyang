import yaml
import jsonlines
import glob
from aria.data.midi import MidiDict
from aria.tokenizer import AbsTokenizer

config_path = "/homes/kb658/yinyang/configs/configs_os.yaml"

# Load config file
with open(config_path, 'r') as f:
    configs = yaml.safe_load(f)
    
artifact_folder = configs["raw_data"]["artifact_folder"]
mono_folder = configs["raw_data"]["mono_folder"]
json_folder = configs["raw_data"]["json_folder"]
raw_data_folders = configs["raw_data"]["raw_data_folders"]
data_path = raw_data_folders['classical']['folder_path']

# Get all the midi files in the data path recursively
midi_files = glob.glob(data_path + '/**/*.midi', recursive=True)


tokenizer = AbsTokenizer()
midi_file_path = midi_files[0]
mid = MidiDict.from_midi(midi_file_path)
tokenized_sequence = tokenizer.tokenize(mid)


# Define a function to flatten the tokenized sequence
def flatten(sequence):
    flattened_sequence = []
    note_info = []
    for i in range(len(sequence)):
        if sequence[i] == "<T>":
            flattened_sequence.append(sequence[i])
        if sequence[i][0] == "piano":
            note_info.append(sequence[i][1])
            note_info.append(sequence[i][2])
        elif sequence[i][0] == "onset":
            note_info.append(sequence[i][1])
        elif sequence[i][0] == "dur":
            note_info.append(sequence[i][1])
            flattened_sequence.append(note_info) 
            note_info = []

    return flattened_sequence  


# Call the flatten function
flattened_sequence = flatten(tokenized_sequence)
print(flattened_sequence[0:100])


def skyline(sequence: list):
    melody = []
    pitch = 0
    velocity = sequence[0][1]
    onset = sequence[0][2]
    duration = sequence[0][3]
    i = 0

    for i in range(len(sequence)):
        if type(sequence[i]) != str:
            current_pitch = sequence[i][0]
            current_velocity = sequence[i][1]
            current_onset = sequence[i][2]
            current_duration = sequence[i][3]

            # Get next note onset
            if i+1 < len(sequence):
                next_onset = sequence[i+1][2]
            
            # Check if current_onset is different from next onset
            if current_onset != next_onset:
                if current_onset == onset and current_pitch > pitch:
                    pitch = current_pitch
                    onset = current_onset
                    duration = current_duration
                    velocity = current_velocity
                else:
                    pitch = current_pitch
                    onset = current_onset
                    duration = current_duration
                    velocity = current_velocity
                # Append the last note
                melody.append(("piano", pitch, velocity))
                melody.append(("onset", onset))
                melody.append(("dur", duration))

            else:
                if current_pitch > pitch:
                    pitch = current_pitch
                    onset = current_onset
                    duration = current_duration
                    velocity = current_velocity
                else:
                    continue
        else:
            melody.append(sequence[i])
    return melody
                

# Call the skyline function
melody = skyline(flattened_sequence[0:100])
print(melody[0:100], '\n')