import glob
import tqdm
import os
import argparse
import yaml
from music21 import harmony, pitch, converter, stream, chord, note, instrument

class Preprocess:
    def __init__(self, folderpath, ouput_folderpath):
        self.folderpath = folderpath
        self.ouput_folderpath = ouput_folderpath
        # Create the output folder if it does not exist
        # if not os.path.exists(self.ouput_folderpath):
        #     os.makedirs(self.ouput_folderpath)

    def process_folder(self, file_extension = "mxl"):
        files = glob.glob(os.path.join(self.folderpath, f"**/*.{file_extension}"), recursive=True)
        print("Number of files: ", len(files))
        for file in tqdm.tqdm(files):
            self.process_file(file, file_extension)

    def process_file(self, filepath, file_extension = "mxl"):
        try:
            score = converter.parse(filepath)
        except (harmony.ChordStepModificationException, pitch.PitchException) as e:
            print(f"Skipping file {os.path.basename(filepath)} due to chord parsing error")
            return
        score = self.change_instrument(score)
        mono_stream = self.get_mono_stream(score)
        self.write_midi(mono_stream, filepath, file_extension)

    def change_instrument(self, score):
        # Change the instrument to piano
        for part in score.parts:
            if part.getInstrument().midiProgram != 0:
                part.insert(1, instrument.Piano())
                part.insert(0, instrument.Piano())
        return score
    
    def get_mono_stream(self, score):
        # Create an empty stream to store the filtered notes
        filtered_stream = stream.Score()

        # Iterate through each element in the score
        for element in score.recurse():
            # Check if the element is a Note
            if 'Note' in element.classes:
                # # Check if the Note is in the treble clef (G-clef) range
                # if element.activeSite.clef.name == 'treble':
                    # Add the Note to the filtered stream
                filtered_stream.append(element)

            # Check if the element is a Chord
            elif 'Chord' in element.classes and 'Harmony' not in element.classes:
                # Get the highest pitch in the chord
                highest_pitch = max(n.midi for n in element.pitches)

                # Create a new Note with the highest pitch and add it to the filtered stream
                filtered_stream.append(note.Note(highest_pitch))
            
            # Add rest too
            elif 'Rest' in element.classes:
                filtered_stream.append(element)
        
        return filtered_stream

    def write_midi(self, mono_stream, filepath, file_extension):
        filename = os.path.basename(filepath).split(f".{file_extension}")[0] + "_mono.mid"
        output_filepath = os.path.join(self.ouput_folderpath, filename)
        mono_stream.write('midi', fp=output_filepath)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.normpath("/homes/kb658/yinyang/configs/configs_os.yaml"),
                        help="Path to the config file")
    args = parser.parse_args()

    # Load config file
    with open(args.config, 'r') as f:
        configs = yaml.safe_load(f)

    raw_data_folders = configs["raw_data"]["raw_data_folders"]
    output_folder = configs["raw_data"]["mono_folder"]
    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for dataset_name, dataset_info in raw_data_folders.items():
        print(f"Dataset: {dataset_name}")
        print(f"Path: {dataset_info.get('folder_path')}")
        print(f"Type: {dataset_info.get('file_extension')}")
        
        folderpath = dataset_info.get('folder_path')
        preprocessor = Preprocess(folderpath, output_folder)
        preprocessor.process_folder(file_extension = dataset_info.get('file_extension'))
        print("Processed all files in {dataset_name}")