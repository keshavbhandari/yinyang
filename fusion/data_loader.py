import yaml
import jsonlines
import glob
import random
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
midi_file_path = random.choice(midi_files)
midi_file_path = "/import/c4dm-datasets/maestro-v3.0.0/2008/MIDI-Unprocessed_07_R2_2008_01-05_ORIG_MID--AUDIO_07_R2_2008_wav--2.midi"
print(midi_file_path)
# midi_file_path = midi_files[0]
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


def skyline(sequence: list, diff_threshold=50):
    melody = []
    harmony = []
    pointer_pitch = sequence[0][0]
    pointer_velocity = sequence[0][1]
    pointer_onset = sequence[0][2]
    pointer_duration = sequence[0][3]
    melody_onset_duration_tracker = pointer_onset + pointer_duration
    harmony_onset_duration_tracker = pointer_onset + pointer_duration
    i = 0

    for i in range(1, len(sequence)):
        if type(sequence[i]) != str:
            current_pitch = sequence[i][0]
            current_velocity = sequence[i][1]
            current_onset = sequence[i][2]
            current_duration = sequence[i][3]

            if type(sequence[i-1]) == str and type(sequence[i-2]) == str:
                diff_curr_prev_onset = 5000
            elif type(sequence[i-1]) == str and type(sequence[i-2]) != str:
                diff_curr_prev_onset = abs(current_onset - sequence[i-2][2])
            else:
                diff_curr_prev_onset = abs(current_onset - sequence[i-1][2])

            if diff_curr_prev_onset > diff_threshold:
                # Append <t> based on condition
                if melody_onset_duration_tracker > 5000 and pointer_onset + pointer_duration < 5000:
                    melody.append("<T>")
                # Append the previous note
                melody.append(("piano", pointer_pitch, pointer_velocity))
                melody.append(("onset", pointer_onset))
                melody.append(("dur", pointer_duration))
                melody_onset_duration_tracker = pointer_onset + pointer_duration
                # Update the pointer
                pointer_pitch = current_pitch
                pointer_velocity = current_velocity
                pointer_onset = current_onset
                pointer_duration = current_duration
            else:
                if current_pitch > pointer_pitch:
                     # Append <t> based on condition
                    if harmony_onset_duration_tracker > 5000 and pointer_onset + pointer_duration < 5000:
                        harmony.append("<T>")
                    # Append the previous note
                    harmony.append(("piano", pointer_pitch, pointer_velocity))
                    harmony.append(("onset", pointer_onset))
                    harmony.append(("dur", pointer_duration))
                    harmony_onset_duration_tracker = pointer_onset + pointer_duration
                    # Update the pointer
                    pointer_pitch = current_pitch
                    pointer_velocity = current_velocity
                    pointer_onset = current_onset
                    pointer_duration = current_duration
                else:
                    # Append <t> based on condition
                    if harmony_onset_duration_tracker > 5000 and pointer_onset + pointer_duration < 5000:
                        harmony.append("<T>")
                    # Append the previous note
                    harmony.append(("piano", current_pitch, current_velocity))
                    harmony.append(("onset", current_onset))
                    harmony.append(("dur", current_duration))
                    harmony_onset_duration_tracker = current_onset + current_duration
                    continue

            # Append the last note
            if i == len(sequence) - 1: 
                if diff_curr_prev_onset > diff_threshold:
                    melody.append(("piano", pointer_pitch, pointer_velocity))
                    melody.append(("onset", pointer_onset))
                    melody.append(("dur", pointer_duration))
                else:
                    if current_pitch > pointer_pitch:
                        melody.append(("piano", current_pitch, current_velocity))
                        melody.append(("onset", current_onset))
                        melody.append(("dur", current_duration))
                    else:
                        harmony.append(("piano", current_pitch, current_velocity))
                        harmony.append(("onset", current_onset))
                        harmony.append(("dur", current_duration))

    return melody, harmony


# Call the flatten function
flattened_sequence = flatten(tokenized_sequence)
print(flattened_sequence[0:100], '\n')

# Call the skyline function
melody, harmony = skyline(flattened_sequence, diff_threshold=30)
melody = tokenized_sequence[0:2] + melody + tokenized_sequence[-1:]
harmony = tokenized_sequence[0:2] + harmony + tokenized_sequence[-1:]
print(melody[0:100], '\n')
print(harmony[0:100], '\n')

print(len(tokenized_sequence))
print(len(melody))
print(len(harmony))

# mid_dict = tokenizer.detokenize(melody) # mid_dict is a MidiDict object
# mid = mid_dict.to_midi() # mid is a mido.MidiFile
# mid.save('test_file_melody.mid')

# mid_dict = tokenizer.detokenize(harmony) # mid_dict is a MidiDict object
# mid = mid_dict.to_midi() # mid is a mido.MidiFile
# mid.save('test_file_harmony.mid')