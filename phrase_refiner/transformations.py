import miditoolkit
import copy
from random import shuffle
from datetime import datetime
import json
import yaml
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.utils import *


class Melodic_Development:
    def __init__(self, beats_in_bar=4):
        # self.note_durations = {1: 1, 2: 2, 3: 4, 4: 4, 5: 6, 6: 8, 7: 10, 8: 12, 9: 16, 10: 20, 11: 24, 12: 28, 13: 36, 14: 44, 15: 52, 16: 52}
        self.note_durations = {i: i for i in range(1, 100)}
        self.beats_in_bar = beats_in_bar

    def extract_phrases(self, midi_encoding, instrument_number):
        encoding = copy.deepcopy(midi_encoding)
        phrases = []
        current_phrase = []

        for i in range(len(encoding)):
            event = encoding[i]
            current_instrument = event[2]
            if current_instrument == instrument_number:
                current_phrase.append(event)

        start_idx = 0
        for i in range(len(current_phrase)):
            event = current_phrase[i]
            event_onset = (event[0] * self.beats_in_bar) + event[1]
            if i < len(current_phrase) - 1:
                next_event = current_phrase[i + 1]
                next_event_onset = (next_event[0] * self.beats_in_bar) + next_event[1]
                if next_event_onset - (event_onset + event[4]) >= 6:
                    phrases.append(current_phrase[start_idx : i + 1])
                    start_idx = i + 1
            else:
                phrases.append(current_phrase[start_idx:i])

        return phrases

    def group_by_bar(self, events):
        grouped_events = {}

        for event in events:
            bar_number = event[0]

            if bar_number not in grouped_events:
                grouped_events[bar_number] = []

            grouped_events[bar_number].append(event)

        return list(grouped_events.values())

    def reindex_bars(self, midi_encoding, start_onset=None, start_bar=None):
        encoding = copy.deepcopy(midi_encoding)

        if start_onset is not None:
            first_bar_onset = encoding[0][1]
            diff_onset = start_onset - first_bar_onset

        if start_bar is not None:
            first_bar = encoding[0][0]
            diff_bar = start_bar - first_bar

        for event in encoding:
            if start_bar is not None:
                event[0] += diff_bar

            if start_onset is not None:
                event[1] += diff_onset

        for event in encoding:
            if event[1] >= self.beats_in_bar:
                event[0] += 1
                event[1] = event[1] % self.beats_in_bar

        return encoding

    def fix_bars(self, midi_encoding, start_onset=0, current_bar="None"):
        encoding = copy.deepcopy(midi_encoding)
        if current_bar == "None" or current_bar is None:
            current_bar = encoding[0][0]
        current_onset = start_onset

        for note in encoding:
            bar, _, _, _, duration, *_ = note[:6]

            # Update the note with the corrected values
            note[0] = current_bar
            note[1] = current_onset

            # Adjust beat onset and bar number
            if current_onset + duration >= self.beats_in_bar:
                current_bar += 1
                current_onset = (current_onset + duration) % self.beats_in_bar
            else:
                current_onset += duration

        return encoding

    def change_instrument(
        self,
        midi_encoding,
        change_from=None,
        instrument_number=80,
        bars_to_skip=1,
        include_original=False,
    ):
        encoding = copy.deepcopy(midi_encoding)
        changed_encoding = copy.deepcopy(midi_encoding)
        last_bar = encoding[-1][0]
        start_bar = encoding[0][0]
        diff_bar = last_bar - start_bar

        for event in changed_encoding:
            if change_from is not None:
                if event[2] == int(change_from):
                    event[2] = instrument_number
            else:
                event[2] = instrument_number

        if include_original:
            # Increment fragment by last bar number of encoding + bars_to_skip + 1, leaving bars_to_skip for infilling
            changed_encoding = [
                [event[0] + diff_bar + bars_to_skip + 1, *event[1:]]
                for event in changed_encoding
            ]
            encoding += changed_encoding
        else:
            return changed_encoding

        return encoding

    # def fragment_melody(
    #     self,
    #     midi_encoding: list,
    #     strategy="bar",
    #     start_onset=0,
    #     bars_to_skip=1,
    #     include_original=False,
    # ):
    #     encoding = copy.deepcopy(midi_encoding)
    #     last_bar = encoding[-1][0]
    #     grouped_by_bar = self.group_by_bar(encoding)

    #     outermost_list_indices = list(range(len(grouped_by_bar)))

    #     if len(outermost_list_indices) < 2:
    #         return encoding

    #     if strategy == "bar":
    #         fragments = random.choice([1, 2, 3])
    #         fragmented_encoding = random.sample(grouped_by_bar, min(fragments, len(grouped_by_bar)))
    #         if len(fragmented_encoding) > 1:
    #             fragmented_encoding = [
    #                 item for sublist in fragmented_encoding for item in sublist
    #             ]
    #             # Fix bars
    #             fragmented_encoding = self.fix_bars(
    #                 fragmented_encoding, start_onset, current_bar="None"
    #             )

    #     elif strategy == "random_crop":
    #         # Select a random start and end index from encoding
    #         start_index, end_index = random.sample(range(len(encoding)), 2)

    #         # Ensure start_index is less than or equal to end_index
    #         start_index, end_index = min(start_index, end_index), max(
    #             start_index, end_index
    #         )

    #         fragmented_encoding = encoding[start_index : end_index + 1]

    #     elif strategy == "sub_phrase":

    #         def calculate_average_duration(encoding):
    #             total_duration = sum(note[4] for note in encoding)
    #             num_notes = len(encoding)
    #             average_duration = total_duration / num_notes
    #             return average_duration

    #         # Calculate average duration of notes in the melody
    #         average_duration = calculate_average_duration(encoding)

    #         # Split the other notes into segments based on the highest duration notes
    #         segments = []
    #         current_segment = []
    #         for note in encoding:
    #             if note[4] > int(average_duration):
    #                 current_segment.append(note)
    #                 segments.append(current_segment)
    #                 current_segment = []
    #             else:
    #                 current_segment.append(note)

    #         if len(current_segment) > 0:
    #             segments.append(current_segment)

    #         # If segment has length greater than 1, check the unique bars in each segment and remove the segment with the least number of unique bars if there are only 3 notes in the segment
    #         if len(segments) > 1:
    #             unique_bars = []
    #             for segment in segments:
    #                 unique_bars.append(len(set([note[0] for note in segment])))
    #             # Get the index of the segment with the least number of unique bars
    #             index_to_remove = unique_bars.index(min(unique_bars))
    #             # Check if the segment has 3 notes or less
    #             if len(segments[index_to_remove]) <= 3:
    #                 # Remove the segment with the least number of unique bars
    #                 segments.pop(index_to_remove)
            
    #         # select a random segment
    #         fragmented_encoding = random.choice(segments)

    #     print("Notes in fragment:", len(fragmented_encoding))
    #     if start_onset is None:
    #         start_onset = fragmented_encoding[0][1]

    #     if include_original:
    #         # Increment fragment by last bar number of encoding + bars_to_skip + 1, leaving bars_to_skip for infilling
    #         fragmented_encoding = self.reindex_bars(
    #             fragmented_encoding, start_onset, start_bar=last_bar + bars_to_skip + 1
    #         )
    #         encoding += fragmented_encoding
    #     else:
    #         fragmented_encoding = self.reindex_bars(
    #             fragmented_encoding, start_onset, start_bar=0
    #         )
    #         # fragmented_encoding = self.fix_bars(fragmented_encoding, start_onset, current_bar = 0)
    #         return fragmented_encoding

    #     return encoding

    def repeat_melody(
        self, midi_encoding, start_onset=None, bars_to_skip=1, include_original=False, **kwargs
    ):
        encoding = copy.deepcopy(midi_encoding)
        repeated_encoding = copy.deepcopy(midi_encoding)
        last_bar = encoding[-1][0]
        if start_onset is None:
            start_onset = encoding[0][1]

        if include_original:
            # Increment fragment by last bar number of encoding + bars_to_skip + 1, leaving bars_to_skip for infilling
            repeated_encoding = self.reindex_bars(
                repeated_encoding, start_onset, start_bar=last_bar + bars_to_skip + 1
            )
            encoding += repeated_encoding
        else:
            repeated_encoding = self.reindex_bars(
                repeated_encoding, start_onset, start_bar=0
            )
            return repeated_encoding

        return encoding

    def sequence_melody(
        self,
        midi_encoding,
        pitch_change=None,
        start_onset=None,
        bars_to_skip=1,
        include_original=False,
        **kwargs
    ):
        encoding = copy.deepcopy(midi_encoding)
        last_bar = encoding[-1][0]
        start_bar = encoding[0][0]
        diff_bar = last_bar - start_bar
        if start_onset is None:
            start_onset = encoding[0][1]
        if pitch_change is None:
            pitch_change = random.choice([i for i in range(-12,12) if i not in [0]])

        print("Pitch Change:", pitch_change)

        sequenced_encoding = [
            [event[0], event[1], event[2], event[3] + pitch_change, *event[4:]]
            for event in encoding
        ]

        if include_original:
            sequenced_encoding = self.reindex_bars(
                sequenced_encoding, start_onset, start_bar=None
            )
            # Increment fragment by last bar number of encoding + bars_to_skip + 1, leaving bars_to_skip for infilling
            sequenced_encoding = [
                [
                    event[0] + diff_bar + bars_to_skip + 1,
                    event[1],
                    event[2],
                    event[3],
                    *event[4:],
                ]
                for event in sequenced_encoding
            ]
            encoding += sequenced_encoding
        else:
            sequenced_encoding = self.reindex_bars(
                sequenced_encoding, start_onset, start_bar=None
            )
            return sequenced_encoding

        return encoding

    def retrograde_melody_pitch(
        self, midi_encoding, start_onset=None, bars_to_skip=1, include_original=False, **kwargs
    ):
        encoding = copy.deepcopy(midi_encoding)
        last_bar = encoding[-1][0]
        if start_onset is None:
            start_onset = encoding[0][1]

        reversed_encoding = [events for events in encoding[::-1]]

        retrograde_encoding = copy.deepcopy(encoding)
        for i in range(len(reversed_encoding)):
            retrograde_encoding[i][3] = reversed_encoding[i][3]

        if include_original:
            # Increment fragment by last bar number of encoding + bars_to_skip + 1, leaving bars_to_skip for infilling
            retrograde_encoding = self.reindex_bars(
                retrograde_encoding, start_onset, start_bar=last_bar + bars_to_skip + 1
            )
            encoding += retrograde_encoding
        else:
            retrograde_encoding = self.reindex_bars(
                retrograde_encoding, start_onset, start_bar=retrograde_encoding[-1][0]
            )
            return retrograde_encoding

        return encoding

    def retrograde_melody_pitch_rhythm(
        self, midi_encoding, start_onset=None, bars_to_skip=1, include_original=False, **kwargs
    ):
        encoding = copy.deepcopy(midi_encoding)
        last_bar = encoding[-1][0]
        start_bar = encoding[0][0]
        diff_bar = last_bar - start_bar
        if start_onset is None:
            start_onset = encoding[0][1]
        reversed_encoding = [events for events in encoding[::-1]]

        retrograde_encoding = copy.deepcopy(reversed_encoding)
        map_dict = {}
        for i in range(len(reversed_encoding)):
            if retrograde_encoding[i][0] not in map_dict.keys():
                map_dict[retrograde_encoding[i][0]] = start_bar
                start_bar += 1

        retrograde_encoding = self.fix_bars(
            retrograde_encoding, start_onset, current_bar=retrograde_encoding[-1][0]
        )

        if include_original:
            # Increment fragment by last bar number of encoding + bars_to_skip + 1, leaving bars_to_skip for infilling
            # retrograde_encoding = self.reindex_bars(retrograde_encoding, start_onset, start_bar = last_bar + bars_to_skip + 1)
            retrograde_encoding = [
                [event[0] + diff_bar + bars_to_skip + 1, *event[1:]]
                for event in retrograde_encoding
            ]
            encoding += retrograde_encoding
        else:
            # retrograde_encoding = self.reindex_bars(retrograde_encoding, start_onset, start_bar = 0)
            return retrograde_encoding

        return encoding

    def invert_melody_strict(
        self, midi_encoding, start_onset=None, bars_to_skip=1, include_original=False, **kwargs
    ):
        encoding = copy.deepcopy(midi_encoding)
        invert_encoding = copy.deepcopy(midi_encoding)
        last_bar = encoding[-1][0]
        start_bar = encoding[0][0]
        diff_bar = last_bar - start_bar
        start_note = encoding[0][3]
        if start_onset is None:
            start_onset = encoding[0][1]

        for i in range(len(invert_encoding)):
            if i < len(invert_encoding) - 1 and i != 0:
                invert_encoding[i][3] = start_note + (start_note - encoding[i][3])

        if include_original:
            # Increment fragment by last bar number of encoding + bars_to_skip + 1, leaving bars_to_skip for infilling
            invert_encoding = self.reindex_bars(
                invert_encoding, start_onset, start_bar=last_bar + bars_to_skip + 1
            )
            encoding += invert_encoding
        else:
            invert_encoding = self.reindex_bars(
                invert_encoding, start_onset, start_bar=None
            )
            return invert_encoding

        return encoding
    
    def is_note_in_diatonic_scale(self, note, root_note, scale_intervals):
        """
        Check if a MIDI note is in the diatonic scale of another MIDI note.

        Parameters:
        - note: MIDI note number to check
        - root_note: MIDI note number representing the root of the diatonic scale
        - scale_intervals: List of intervals representing the diatonic scale (e.g., ['P1', 'M2', 'M3', ...])

        Returns:
        - True if the note is in the diatonic scale, False otherwise
        """
        # Define the interval values
        self.interval_values = {'P1': 0, 'm2': 1, 'M2': 2, 'm3': 3, 'M3': 4,
                        'P4': 5, 'A4': 6, 'd5': 6, 'P5': 7, 'm6': 8,
                        'M6': 9, 'm7': 10, 'M7': 11, 'P8': 12}

        # Calculate the relative note number with respect to the root
        relative_note = (note - root_note) % 12

        # Check if the relative note is in the specified diatonic scale intervals
        return relative_note in [self.interval_values.get(interval, -1) for interval in scale_intervals]
    
    def invert_melody_tonal(
        self, midi_encoding, start_onset=None, bars_to_skip=1, include_original=False, **kwargs
    ):
        encoding = copy.deepcopy(midi_encoding)
        invert_encoding = copy.deepcopy(midi_encoding)
        last_bar = encoding[-1][0]
        start_bar = encoding[0][0]
        diff_bar = last_bar - start_bar
        start_note = encoding[0][3]
        if start_onset is None:
            start_onset = encoding[0][1]

        # Normalize changed_encoding to c major or a minor
        _, is_major, pitch_shift = normalize_to_c_major(invert_encoding)
        
        # Define the number of octaves
        num_octaves = 3
        if is_major:
            # Define the diatonic scale pattern in semitones
            diatonic_scale_pattern = ['P1', 'M2', 'M3', 'P4', 'P5', 'M6', 'M7', 'P8']
        else:
            diatonic_scale_pattern = ['P1', 'M2', 'm3', 'P4', 'P5', 'm6', 'm7', 'P8']

        for i in range(len(invert_encoding)):
            if i < len(invert_encoding) - 1 and i != 0:
                inverted_pitch = start_note - encoding[i][3]
                # Check if the inverted pitch is in the diatonic scale
                in_diatonic_scale = self.is_note_in_diatonic_scale(inverted_pitch, start_note, diatonic_scale_pattern)
                if not in_diatonic_scale:
                    if is_major:
                        inverted_pitch += 1
                    else:
                        inverted_pitch -= 1

                invert_encoding[i][3] = start_note + inverted_pitch

        if include_original:
            # Increment fragment by last bar number of encoding + bars_to_skip + 1, leaving bars_to_skip for infilling
            invert_encoding = self.reindex_bars(
                invert_encoding, start_onset, start_bar=last_bar + bars_to_skip + 1
            )
            encoding += invert_encoding
        else:
            invert_encoding = self.reindex_bars(
                invert_encoding, start_onset, start_bar=None
            )
            return invert_encoding

        return encoding

    def expand_melody(
        self, midi_encoding, start_onset=None, bars_to_skip=1, include_original=False, **kwargs
    ):
        encoding = copy.deepcopy(midi_encoding)
        expand_encoding = copy.deepcopy(midi_encoding)
        last_bar = encoding[-1][0]
        start_bar = encoding[0][0]
        diff_bar = last_bar - start_bar
        if start_onset is None:
            start_onset = encoding[0][1]

        for i in range(len(expand_encoding)):
            if expand_encoding[i][4] <= 4:
                expand_encoding[i][4] = expand_encoding[i][4] * 2

        if include_original:
            # expand_encoding = self.reindex_bars(expand_encoding, start_onset, start_bar = last_bar + bars_to_skip + 1)
            expand_encoding = self.fix_bars(
                expand_encoding, start_onset, current_bar="None"
            )
            # Increment fragment by last bar number of encoding + bars_to_skip + 1, leaving bars_to_skip for infilling
            expand_encoding = [
                [event[0] + diff_bar + bars_to_skip + 1, *event[1:]]
                for event in expand_encoding
            ]
            encoding += expand_encoding
        else:
            # expand_encoding = self.reindex_bars(expand_encoding, start_onset, start_bar = 0)
            expand_encoding = self.fix_bars(expand_encoding, start_onset, current_bar=None)
            return expand_encoding

        return encoding

    def contract_melody(
        self, midi_encoding, start_onset=None, bars_to_skip=1, include_original=False, **kwargs
    ):
        encoding = copy.deepcopy(midi_encoding)
        contract_encoding = copy.deepcopy(midi_encoding)
        last_bar = encoding[-1][0]
        start_bar = encoding[0][0]
        diff_bar = last_bar - start_bar
        if start_onset is None:
            start_onset = encoding[0][1]

        for i in range(len(contract_encoding)):
            if contract_encoding[i][4] > 0.25:
                contract_encoding[i][4] = contract_encoding[i][4] / 2

        contract_encoding = self.fix_bars(
            contract_encoding, start_onset, current_bar="None"
        )

        if include_original:
            # Increment fragment by last bar number of encoding + bars_to_skip + 1, leaving bars_to_skip for infilling
            contract_encoding = self.reindex_bars(
                contract_encoding, start_onset, start_bar=last_bar + bars_to_skip + 1
            )
            encoding += contract_encoding
        else:
            contract_encoding = self.reindex_bars(
                contract_encoding, start_onset, start_bar=None
            )
            return contract_encoding

        return encoding

    def permute_melody_pitch(
        self, midi_encoding, start_onset=None, bars_to_skip=1, include_original=False, **kwargs
    ):
        encoding = copy.deepcopy(midi_encoding)
        permuted_encoding = copy.deepcopy(midi_encoding)
        last_bar = encoding[-1][0]
        start_bar = encoding[0][0]
        diff_bar = last_bar - start_bar
        if start_onset is None:
            start_onset = encoding[0][1]

        shuffle(permuted_encoding)

        for i in range(len(encoding)):
            permuted_encoding[i][0] = encoding[i][0]
            permuted_encoding[i][1] = encoding[i][1]
            permuted_encoding[i][2] = encoding[i][2]
            permuted_encoding[i][4:] = encoding[i][4:]

        if include_original:
            permuted_encoding = self.reindex_bars(
                permuted_encoding, start_onset, start_bar=last_bar + bars_to_skip + 1
            )
            # permuted_encoding = self.fix_bars(permuted_encoding, start_onset, current_bar = 'None')
            # Increment fragment by last bar number of encoding + bars_to_skip + 1, leaving bars_to_skip for infilling
            # permuted_encoding = [[event[0] + diff_bar + bars_to_skip + 1, *event[1:]] for event in permuted_encoding]
            encoding += permuted_encoding
        else:
            permuted_encoding = self.reindex_bars(
                permuted_encoding, start_onset, start_bar=None
            )
            # permuted_encoding = self.fix_bars(permuted_encoding, start_onset, current_bar = 0)
            return permuted_encoding

        return encoding

    def permute_melody_pitch_rhythm(
        self, midi_encoding, start_onset=None, bars_to_skip=1, include_original=False, **kwargs
    ):
        encoding = copy.deepcopy(midi_encoding)
        permuted_encoding = copy.deepcopy(midi_encoding)
        last_bar = encoding[-1][0]
        start_bar = encoding[0][0]
        diff_bar = last_bar - start_bar
        if start_onset is None:
            start_onset = encoding[0][1]

        shuffle(permuted_encoding)

        for i in range(len(encoding)):
            permuted_encoding[i][0] = encoding[i][0]
            permuted_encoding[i][1] = encoding[i][1]
            permuted_encoding[i][2] = encoding[i][2]
            permuted_encoding[i][5:] = encoding[i][5:]

        if include_original:
            # permuted_encoding = self.reindex_bars(permuted_encoding, start_onset, start_bar = last_bar + bars_to_skip + 1)
            permuted_encoding = self.fix_bars(
                permuted_encoding, start_onset, current_bar="None"
            )
            # Increment fragment by last bar number of encoding + bars_to_skip + 1, leaving bars_to_skip for infilling
            permuted_encoding = [
                [event[0] + diff_bar + bars_to_skip + 1, *event[1:]]
                for event in permuted_encoding
            ]
            encoding += permuted_encoding
        else:
            # permuted_encoding = self.reindex_bars(permuted_encoding, start_onset, start_bar = 0)
            permuted_encoding = self.fix_bars(
                permuted_encoding, start_onset, current_bar=start_bar
            )
            return permuted_encoding

        return encoding

    def permute_melody_rhythm(
        self, midi_encoding, start_onset=None, bars_to_skip=1, include_original=False, **kwargs
    ):
        encoding = copy.deepcopy(midi_encoding)
        permuted_encoding = copy.deepcopy(midi_encoding)
        last_bar = encoding[-1][0]
        start_bar = encoding[0][0]
        diff_bar = last_bar - start_bar
        if start_onset is None:
            start_onset = encoding[0][1]

        shuffle(permuted_encoding)

        for i in range(len(encoding)):
            permuted_encoding[i][0] = encoding[i][0]
            permuted_encoding[i][1] = encoding[i][1]
            permuted_encoding[i][2] = encoding[i][2]
            permuted_encoding[i][3] = encoding[i][3]
            permuted_encoding[i][5:] = encoding[i][5:]

        if include_original:
            # permuted_encoding = self.reindex_bars(permuted_encoding, start_onset, start_bar = last_bar + bars_to_skip + 1)
            permuted_encoding = self.fix_bars(
                permuted_encoding, start_onset, current_bar="None"
            )
            # Increment fragment by last bar number of encoding + bars_to_skip + 1, leaving bars_to_skip for infilling
            permuted_encoding = [
                [event[0] + diff_bar + bars_to_skip + 1, *event[1:]]
                for event in permuted_encoding
            ]
            encoding += permuted_encoding
        else:
            # permuted_encoding = self.reindex_bars(permuted_encoding, start_onset, start_bar = 0)
            permuted_encoding = self.fix_bars(
                permuted_encoding, start_onset, current_bar=start_bar
            )
            return permuted_encoding

        return encoding
    
    def permute_melody_new_pitch(self, midi_encoding, **kwargs):
        encoding = copy.deepcopy(midi_encoding)
        permuted_encoding = []
        # Choose a random pitch value for the new note that is between -5 and 7
        new_pitch = random.choice([i for i in range(-5, 8) if i not in [0]])
        pitch_values = [event[3] + new_pitch for event in encoding]
        for event in encoding:
            event[3] = random.choice(pitch_values)
            permuted_encoding.append(event)

        return permuted_encoding

    def get_note_sequence(self, encoded_note, major=True):
        note_pitch = encoded_note[3]
        note_duration = encoded_note[4]

        major_sequence = {
            "major_turn": [note_pitch, note_pitch + 2, note_pitch, note_pitch - 1],
            "major_mordent_up": [note_pitch, note_pitch + 2],
            "major_mordent_down": [note_pitch, note_pitch - 1],
            "major_trill": [note_pitch, note_pitch + 2, note_pitch, note_pitch + 2],
            "major_appogiatura_up": [note_pitch - 1],
            "major_appogiatura_down": [note_pitch + 2],
            "major_lower_neighbour_chord": [note_pitch, note_pitch - 1, note_pitch - 3],
            "major_upper_neighbour_chord": [note_pitch, note_pitch + 2, note_pitch + 4],
            "major_sequence_upper": [note_pitch, note_pitch + 2, note_pitch - 1],
            "major_sequence_lower": [note_pitch, note_pitch - 1, note_pitch + 2],
            "major_split": [note_pitch],
        }

        minor_sequence = {
            "minor_turn": [note_pitch, note_pitch + 1, note_pitch, note_pitch - 1],
            "minor_mordent_up": [note_pitch, note_pitch + 1],
            "minor_mordent_down": [note_pitch, note_pitch - 1],
            "minor_trill": [note_pitch, note_pitch + 1, note_pitch, note_pitch + 1],
            "minor_appogiatura_up": [note_pitch - 2],
            "minor_appogiatura_down": [note_pitch + 1],
            "minor_lower_neighbour_chord": [note_pitch, note_pitch - 1, note_pitch - 3],
            "minor_upper_neighbour_chord": [note_pitch, note_pitch + 2, note_pitch + 3],
            "minor_sequence_upper": [note_pitch, note_pitch + 1, note_pitch - 1],
            "minor_sequence_lower": [note_pitch, note_pitch - 2, note_pitch + 1],
            "minor_split": [note_pitch],
        }

        if major:
            sequence = major_sequence
        else:
            sequence = minor_sequence

        random_sequence_type, random_sequence_values = random.choice(
            list(sequence.items())
        )
        print(random_sequence_type)
        note_sequence = [encoded_note[:] for _ in range(len(random_sequence_values))]
        for i in range(len(note_sequence)):
            note_sequence[i][3] = random_sequence_values[i]
            note_sequence[i][4] = 0.25  # Minimum note duration is 0.25
            note_sequence[i][4] = 0.25  # Minimum note duration is 0.25

        return random_sequence_type, note_sequence

    def embellish_melody(
        self,
        midi_encoding: list,
        major=True,
        embellishment_rate=0.3,
        start_onset=None,
        bars_to_skip=1,
        include_original=False,
        **kwargs
    ):
        encoding = copy.deepcopy(midi_encoding)
        embellished_encoding = copy.deepcopy(midi_encoding)
        last_bar = encoding[-1][0]
        start_bar = encoding[0][0]
        diff_bar = last_bar - start_bar
        if start_onset is None:
            start_onset = encoding[0][1]

        i = 0
        while i < len(embellished_encoding):
            if embellished_encoding[i][4] >= 0.5:
                if random.random() < embellishment_rate:
                    selected_note = copy.deepcopy(embellished_encoding[i])
                    note_sequence = self.get_note_sequence(selected_note, major)
                    note_sequence_pitches = note_sequence[1]
                    embellished_encoding[i][4] = (
                        1
                        if embellished_encoding[i][4] - len(note_sequence_pitches) <= 1
                        else embellished_encoding[i][4] - len(note_sequence_pitches)
                    )
                    embellished_encoding[i:i] = note_sequence_pitches
                    # Reset i to 0 to account for the new notes added
                    i = 0
            i += 1

        if include_original:
            # embellished_encoding = self.reindex_bars(embellished_encoding, start_onset, start_bar = last_bar + bars_to_skip + 1)
            embellished_encoding = self.fix_bars(
                embellished_encoding, start_onset, current_bar="None"
            )
            # Increment fragment by last bar number of encoding + bars_to_skip + 1, leaving bars_to_skip for infilling
            embellished_encoding = [
                [event[0] + diff_bar + bars_to_skip + 1, *event[1:]]
                for event in embellished_encoding
            ]
            encoding += embellished_encoding
        else:
            # embellished_encoding = self.reindex_bars(embellished_encoding, start_onset, start_bar = 0)
            embellished_encoding = self.fix_bars(
                embellished_encoding, start_onset, current_bar=start_bar
            )
            return embellished_encoding

        return encoding

    def reduce_melody(
        self, midi_encoding, start_onset=None, bars_to_skip=1, include_original=False, **kwargs
    ):
        encoding = copy.deepcopy(midi_encoding)
        last_bar = encoding[-1][0]
        start_bar = encoding[0][0]
        diff_bar = last_bar - start_bar
        if start_onset is None:
            start_onset = encoding[0][1]

        reduced_encoding = []
        previous_note = None

        for event in midi_encoding:
            current_note = event[3]  # MIDI note number

            if current_note != previous_note:
                reduced_encoding.append(event)
                previous_note = current_note
            # else:
            # reduced_encoding[-1][4] += event[4]  # Add duration of current note to previous note

        reduced_encoding = self.fix_bars(
            reduced_encoding, start_onset, current_bar="None"
        )

        if include_original:
            # Increment fragment by last bar number of encoding + bars_to_skip + 1, leaving bars_to_skip for infilling
            reduced_encoding = self.reindex_bars(
                reduced_encoding, start_onset, start_bar=last_bar + bars_to_skip + 1
            )
            # reduced_encoding = [[event[0] + diff_bar + bars_to_skip + 1, *event[1:]] for event in reduced_encoding]
            encoding += reduced_encoding
        else:
            reduced_encoding = self.reindex_bars(
                reduced_encoding, start_onset, start_bar=start_bar
            )
            return reduced_encoding

        return encoding

    def change_major_minor(
        self, midi_encoding, major, start_onset=None, bars_to_skip=1, include_original=False, **kwargs
    ):
        encoding = copy.deepcopy(midi_encoding)
        changed_encoding = copy.deepcopy(midi_encoding)
        last_bar = encoding[-1][0]
        start_bar = encoding[0][0]
        diff_bar = last_bar - start_bar
        if start_onset is None:
            start_onset = encoding[0][1]

        # # Normalize changed_encoding to c major or a minor
        changed_encoding, _, pitch_shift = normalize_to_c_major(changed_encoding)
        # print("Is Major:", is_major, "Pitch Shift:", pitch_shift)

        # Root Note
        root_note = changed_encoding[0][3]

        def get_intervals(note_value):
            # Define the MIDI note numbers for E (third intervals) across octaves
            major_intervals = []
            minor_intervals = []

            # Calculate the note values once
            e_note_base = note_value + 4
            e_below_base = note_value - 8
            a_note_base = note_value + 9
            a_below_base = note_value - 3
            e_flat_note_base = note_value + 3
            e_flat_below_base = note_value - 9
            a_flat_note_base = note_value + 8
            a_flat_below_base = note_value - 4

            for octave in range(11):  # Assuming 11 octaves (C-1 to G9)
                octave_offset = octave * 12  # There are 12 semitones in an octave

                major_intervals.extend(
                    [
                        e_note_base + octave_offset,
                        e_below_base + octave_offset,
                        a_note_base + octave_offset,
                        a_below_base + octave_offset,
                    ]
                )

                minor_intervals.extend(
                    [
                        e_flat_note_base + octave_offset,
                        e_flat_below_base + octave_offset,
                        a_flat_note_base + octave_offset,
                        a_flat_below_base + octave_offset,
                    ]
                )

            return major_intervals, minor_intervals

        major_intervals, minor_intervals = get_intervals(root_note)

        if major:
            for i in range(len(changed_encoding)):
                # Check if note is 3rd or 6th of c major scale and convert to minor
                if changed_encoding[i][3] in major_intervals:
                    changed_encoding[i][3] = changed_encoding[i][3] - 1
        else:
            for i in range(len(changed_encoding)):
                # Check if note is 3rd or 6th of a minor scale and convert to major
                if changed_encoding[i][3] in minor_intervals:
                    changed_encoding[i][3] = changed_encoding[i][3] + 1

        # Revert back to original key
        changed_encoding = [[event[0], event[1], event[2], event[3] - pitch_shift, *event[4:]] for event in changed_encoding]

        if include_original:
            # Increment fragment by last bar number of encoding + bars_to_skip + 1, leaving bars_to_skip for infilling
            changed_encoding = self.reindex_bars(
                changed_encoding, start_onset, start_bar=last_bar + bars_to_skip + 1
            )
            encoding += changed_encoding
        else:
            changed_encoding = self.reindex_bars(
                changed_encoding, start_onset, start_bar=start_bar
            )
            return changed_encoding

        return encoding

    def harmonize_melody(
        self,
        midi_encoding,
        instruments_to_harmonize=[25, 80, 48, 32],
        start_onset=None,
        include_original=False,
        **kwargs
    ):
        encoding = copy.deepcopy(midi_encoding)
        if start_onset is None:
            start_onset = encoding[0][1]
        # group by bar
        instrument_bars = self.group_by_bar(encoding)

        harmonize_encoding = []
        for instrument in instruments_to_harmonize:
            # Choose a random note density between 1 and 8 based on higher probability for lower values
            # Define the population and weights
            population = list(range(1, 9))
            weights = [0.4, 0.3, 0.15, 0.10, 0.3, 0.1, 0.05, 0.05]
            # Choose a random note density with the given weights
            note_density = random.choices(population, weights, k=1)[0]
            # note_density = random.choice(range(1, 9))

            for i in range(len(instrument_bars) - 1):
                (
                    bar_number,
                    bar_onset,
                    bar_instrument,
                    bar_pitch,
                    bar_duration,
                    bar_velocity,
                    bar_tpc,
                    bar_extra,
                ) = instrument_bars[i][0]
                if instrument != bar_instrument:
                    new_duration = 9 // note_density
                    for j in range(note_density):
                        onset = bar_onset + j * new_duration
                        duration = (
                            new_duration
                            if j < note_density - 1
                            else 9 - j * new_duration
                        )
                        harmonize_encoding.append(
                            [
                                bar_number,
                                onset,
                                instrument,
                                bar_pitch,
                                duration,
                                bar_velocity,
                                bar_tpc,
                                bar_extra,
                            ]
                        )

        if include_original:
            # Increment fragment by last bar number of encoding + bars_to_skip + 1, leaving bars_to_skip for infilling
            # harmonize_encoding = self.reindex_bars(harmonize_encoding, start_onset, start_bar = last_bar + bars_to_skip + 1)
            encoding += harmonize_encoding
            # Sort list of lists by bar number, onset time and instrument number
            encoding.sort(key=lambda x: (x[0], x[1], x[2]))
        else:
            # harmonize_encoding = self.reindex_bars(harmonize_encoding, start_onset, start_bar = 0)
            return harmonize_encoding

        return encoding

    def metric_displacement(
        self, midi_encoding, start_onset=None, bars_to_skip=1, include_original=False, **kwargs
    ):
        encoding = copy.deepcopy(midi_encoding)
        displaced_encoding = copy.deepcopy(midi_encoding)
        last_bar = encoding[-1][0]
        start_bar = encoding[0][0]
        if start_onset is None:
            start_onset = encoding[0][1]

        # Identify the strong beats in the bar
        strong_beats = [0]

        # Displace the notes by a random number of beats
        displacement = random.choice([2, 3, 4])
        # displaced_encoding = self.fix_bars(displaced_encoding, displacement, current_bar = 'None')
        for event in displaced_encoding:
            if event[1] in strong_beats and event[4] > 1:
                event[4] = event[4] / displacement

        # fix the bars
        displaced_encoding = self.fix_bars(
            displaced_encoding, displacement, current_bar="None"
        )

        if include_original:
            # Increment fragment by last bar number of encoding + bars_to_skip + 1, leaving bars_to_skip for infilling
            displaced_encoding = self.reindex_bars(
                displaced_encoding, displacement, start_bar=last_bar + bars_to_skip + 1
            )
            encoding += displaced_encoding
        else:
            displaced_encoding = self.reindex_bars(
                displaced_encoding, displacement, start_bar=start_bar
            )
            return displaced_encoding



# Write phrase corruption code here
class Phrase_Corruption(Melodic_Development):
    def __init__(self, beats_in_bar=32):
        super().__init__(beats_in_bar)

    def masking(self, midi_encoding, mask_type="pitch", **kwargs):
        encoding = copy.deepcopy(midi_encoding)
        masked_encoding = []

        if mask_type == "bar":
            # Choose a random bar to mask all notes from and then append all bars to the masked encoding
            random_bar = random.choice(self.group_by_bar(encoding))
            for i in range(len(self.group_by_bar(encoding))):
                if self.group_by_bar(encoding)[i] == random_bar:
                    masked_encoding.append([self.group_by_bar(encoding)[i][0][0], "BAR_MASK", "BAR_MASK", "BAR_MASK", "BAR_MASK", "BAR_MASK"])
                else:
                    for event in self.group_by_bar(encoding)[i]:
                        masked_encoding.append(event)
        elif mask_type == "first_bar":
            # Mask the first bar of the encoding
            masked_encoding.append([self.group_by_bar(encoding)[0][0][0], "BAR_MASK", "BAR_MASK", "BAR_MASK", "BAR_MASK", "BAR_MASK"])
            for event in encoding:
                # Increment the bar number of the event by 1
                event[0] += 1
                masked_encoding.append(event)
        else:
            for event in encoding:
                if mask_type == "pitch":
                    if random.random() < 0.75:
                        event[3] = "PITCH_MASK"
                    else:
                        event[3] = event[3]
                elif mask_type == "duration":
                    event[1] = "DURATION_MASK"
                    event[4] = "DURATION_MASK"
                else:
                    raise ValueError("Invalid mask type. Please choose from 'pitch', 'duration' or 'bar'.")
                masked_encoding.append(event)

        return masked_encoding

    def incorrect_transposition(self, midi_encoding, **kwargs):
        encoding = copy.deepcopy(midi_encoding)
        # Find highest and lowest pitch values
        pitch_values = [event[3] for event in encoding]
        highest_pitch = max(pitch_values)
        lowest_pitch = min(pitch_values)
        # Choose a random pitch change value but ensure it is not 0 and within the midi pitch range of 0 to 127
        pitch_change = random.choice([i for i in range(-12,12) if i not in [0]])
        while highest_pitch + pitch_change > 127 or lowest_pitch + pitch_change < 0:
            if pitch_change < 0:
                pitch_change += 1
            else:
                pitch_change -= 1
        
        sequenced_encoding = [
            [event[0], event[1], event[2], event[3] + pitch_change, *event[4:]]
            for event in encoding
        ]

        return sequenced_encoding

    def incorrect_inversion(self, midi_encoding, **kwargs):
        encoding = copy.deepcopy(midi_encoding)
        inversion = random.choice([i for i in range(-4,4) if i not in [0]])

        for event in encoding:
            # random probability of inversion
            if random.random() < 0.3:
                event[3] = inversion + event[3]

        return encoding

    def melodic_stripping(self, midi_encoding, **kwargs):
        encoding = copy.deepcopy(midi_encoding)
        stripped_encoding = []

        i = 0
        while i < len(encoding) - 1:
            if random.random() < 0.5 and encoding[i][4] <= 2 and encoding[i+1][4] <= 2:
                stripped_encoding.append([
                    encoding[i][0],
                    encoding[i][1],
                    encoding[i][2],
                    encoding[i][3],
                    encoding[i][4] + encoding[i+1][4],
                    encoding[i][5]
                ])
                i += 2
            else:
                stripped_encoding.append(encoding[i])
                i += 1

        # If there is a remaining note at the end, append it to the stripped encoding
        if i == len(encoding) - 1:
            stripped_encoding.append(encoding[i])

        return stripped_encoding
    
    def get_random_midi_pitch(self, key_signature, is_major, pitch_value):
        major_scales = {
            "KS_A-": ["A", "B", "C", "D", "E", "F", "G"],
            "KS_A": ["A", "B", "C#", "D", "E", "F#", "G#"],
            "KS_B-": ["B", "C#", "D", "E", "F#", "G", "A"],
            "KS_B": ["B", "C#", "D#", "E", "F#", "G#", "A#"],
            "KS_C": ["C", "D", "E", "F", "G", "A", "B"],
            "KS_D-": ["D", "E", "F", "G", "A", "Bb", "C"],
            "KS_D": ["D", "E", "F#", "G", "A", "B", "C#"],
            "KS_E-": ["E", "F#", "G", "A", "B", "C", "D"],
            "KS_E": ["E", "F#", "G#", "A", "B", "C#", "D#"],
            "KS_F": ["F", "G", "A", "Bb", "C", "D", "E"],
            "KS_F#": ["F#", "G#", "A#", "B", "C#", "D#", "E#"],
            "KS_G": ["G", "A", "B", "C", "D", "E", "F#"]
        }
        minor_scales = {
        "KS_A-": ["A", "B", "C", "D", "E", "F", "G"],
        "KS_A": ["A", "B", "C#", "D", "E", "F#", "G#"],
        "KS_B-": ["B", "C#", "D", "E", "F#", "G", "A"],
        "KS_B": ["B", "C#", "D#", "E", "F#", "G#", "A#"],
        "KS_C": ["C", "D", "Eb", "F", "G", "Ab", "Bb"],
        "KS_D-": ["D", "E", "F", "G", "A", "Bb", "C"],
        "KS_D": ["D", "E", "F#", "G", "A", "B", "C#"],
        "KS_E-": ["E", "F#", "G", "A", "B", "C", "D"],
        "KS_E": ["E", "F#", "G#", "A", "B", "C#", "D#"],
        "KS_F": ["F", "G", "Ab", "Bb", "C", "Db", "Eb"],
        "KS_F#": ["F#", "G#", "A", "B", "C#", "D", "E"],
        "KS_G": ["G", "A", "Bb", "C", "D", "Eb", "F"]
        }
        
        if is_major:
            scale = major_scales[key_signature]
        else:
            scale = minor_scales[key_signature]

        pitch = random.choice(scale)

        # Convert note names to MIDI note values
        midi_pitch = self.get_nearest_midi_pitch(pitch_value, pitch)

        return midi_pitch
    
    def get_nearest_midi_pitch(self, midi_pitch, pitch_class):
        notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        enharmonic_notes = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
        if pitch_class in notes:
            pitch_index = notes.index(pitch_class)
        else:
            pitch_index = enharmonic_notes.index(pitch_class)
        pitch_class_octave = (midi_pitch // 12) - 1
        pitch_class_midi_pitch_lo = pitch_class_octave * 12 + pitch_index
        pitch_class_midi_pitch_ho = (pitch_class_octave + 1) * 12 + pitch_index
        
        diff_lo = abs(midi_pitch - pitch_class_midi_pitch_lo)
        diff_ho = abs(midi_pitch - pitch_class_midi_pitch_ho)

        if diff_lo < diff_ho:
            return pitch_class_midi_pitch_lo
        else:
            return pitch_class_midi_pitch_ho

    def melodic_addition(self, midi_encoding, key_signature, mode, **kwargs):
        if "major" in mode.lower():
            mode = True
        else:
            mode = False
        encoding = copy.deepcopy(midi_encoding)
        added_encoding = []

        i = 0
        while i < len(encoding) - 1:
            if random.random() < 0.5:
                added_encoding.append(encoding[i])
                # Get random pitch value for the new note
                nearest_pitch = self.get_random_midi_pitch(key_signature, mode, encoding[i][3])
                # Get random duration value for the new note from all possible durations in encoding
                duration_values = [event[4] for event in encoding]
                new_duration = random.choice(duration_values)
                # Add the new note to the added encoding
                added_encoding.append([
                    encoding[i][0],
                    encoding[i][1] + encoding[i][4],
                    encoding[i][2],
                    nearest_pitch,
                    new_duration,
                    encoding[i][5]
                ])
                i += 1
            else:
                added_encoding.append(encoding[i])
                i += 1

        # If there is a remaining note at the end, append it to the added encoding
        if i == len(encoding) - 1:
            added_encoding.append(encoding[i])

        # Fix bars of the added encoding
        added_encoding = self.fix_bars(added_encoding, start_onset=midi_encoding[0][1], current_bar=midi_encoding[0][0])

        return added_encoding
    
    def note_swapping(self, midi_encoding, **kwargs):
        encoding = copy.deepcopy(midi_encoding)
        swapped_encoding = []

        i = 0
        while i < len(encoding) - 1:
            # Check if offset + duration of note is less than or equal to beats in bar
            check = encoding[i+1][1] + encoding[i+1][4] <= self.beats_in_bar
            if random.random() < 0.5 and check:
                # Next note offset takes current note's offset with next note's duration as the notes will be swapped
                next_note_offset = encoding[i][1] + encoding[i+1][4]
                next_note_bar = encoding[i][0] + (next_note_offset // self.beats_in_bar)
                swapped_encoding.append([
                    encoding[i][0],
                    encoding[i][1],
                    encoding[i][2],
                    encoding[i+1][3],
                    encoding[i+1][4],
                    encoding[i+1][5],
                ])
                swapped_encoding.append([
                    next_note_bar,
                    next_note_offset,
                    encoding[i+1][2],
                    encoding[i][3],
                    encoding[i][4],
                    encoding[i][5],
                ])
                i += 2
            else:
                swapped_encoding.append(encoding[i])
                i += 1

        # If there is a remaining note at the end, append it to the swapped encoding
        if i == len(encoding) - 1:
            swapped_encoding.append(encoding[i])

        return swapped_encoding
    
    def same_note_modification(self, midi_encoding, **kwargs):
        encoding = copy.deepcopy(midi_encoding)
        modified_encoding = []
        note_ind_eliminated = []
        
        for i in range(len(encoding)):
            # Combine duration of two same notes into a single note 50% of the time and change their durations accordingly
            if random.random() < 0.5 and i != len(encoding) - 1 and encoding[i][3] == encoding[i+1][3] and i not in note_ind_eliminated and encoding[i][4] < self.beats_in_bar and encoding[i+1][4] < self.beats_in_bar:
                # Combine the two notes into a single note
                modified_encoding.append([
                    encoding[i][0],
                    encoding[i][1],
                    encoding[i][2],
                    encoding[i][3],
                    encoding[i][4] + encoding[i+1][4],
                    encoding[i][5],
                ])
                note_ind_eliminated.append(i+1)
            
            elif random.random() < 0.5 and i not in note_ind_eliminated and encoding[i][4] >= self.beats_in_bar/2:
                # Split the note into two notes
                split_duration = encoding[i][4] // 2 #random.randint(1, encoding[i][4] - 1)
                # Next note offset should be the remainder of 32 - current note's offset + split duration
                next_note_offset = (encoding[i][1] + split_duration) % self.beats_in_bar
                next_note_bar = encoding[i][0] + (encoding[i][1] + split_duration) // self.beats_in_bar
                
                modified_encoding.append([
                    encoding[i][0],
                    encoding[i][1],
                    encoding[i][2],
                    encoding[i][3],
                    split_duration,
                    encoding[i][5],
                ])
                
                modified_encoding.append([
                    next_note_bar,
                    next_note_offset,
                    encoding[i][2],
                    encoding[i][3],
                    encoding[i][4] - split_duration,
                    encoding[i][5],
                ])
            
            elif i not in note_ind_eliminated:
                modified_encoding.append(encoding[i])
        
        return modified_encoding
    
    def permute_note_pitch(self, midi_encoding, **kwargs):
        encoding = copy.deepcopy(midi_encoding)
        permuted_encoding = []
        pitch_values = [event[3] for event in encoding]
        for event in encoding:
            event[3] = random.choice(pitch_values)
            # Remove the pitch value from the list of pitch values
            pitch_values.remove(event[3])
            permuted_encoding.append(event)        

        return permuted_encoding
    
    def permute_note_duration(self, midi_encoding, **kwargs):
        encoding = copy.deepcopy(midi_encoding)
        permuted_encoding = []
        duration_values = [event[4] for event in encoding]
        for event in encoding:
            event[4] = random.choice(duration_values)
            # Remove the duration value from the list of duration values
            duration_values.remove(event[4])
            permuted_encoding.append(event)

        permuted_encoding = self.fix_bars(permuted_encoding, start_onset=midi_encoding[0][1], current_bar=midi_encoding[0][0])

        return permuted_encoding
    
    def permute_note_pitch_duration(self, midi_encoding, **kwargs):
        encoding = copy.deepcopy(midi_encoding)
        permuted_encoding = []
        pitch_values = [event[3] for event in encoding]
        duration_values = [event[4] for event in encoding]

        for i in range(len(encoding)):
            encoding[i][3] = random.choice(pitch_values)
            encoding[i][4] = random.choice(duration_values)
            # Remove the pitch and duration values from the list of pitch and duration values
            pitch_values.remove(encoding[i][3])
            duration_values.remove(encoding[i][4])
            permuted_encoding.append(encoding[i])

        permuted_encoding = self.fix_bars(permuted_encoding, start_onset=midi_encoding[0][1], current_bar=midi_encoding[0][0])

        return permuted_encoding
    
    def fragment_notes(
        self,
        midi_encoding: list,
        strategy="bar",
        **kwargs
    ):
        encoding = copy.deepcopy(midi_encoding)
        grouped_by_bar = self.group_by_bar(encoding)

        outermost_list_indices = list(range(len(grouped_by_bar)))

        if len(outermost_list_indices) < 2:
            return encoding

        if strategy == "bar":
            # fragments = random.choice([1, 2])
            # fragmented_encoding = random.sample(grouped_by_bar, min(fragments, len(grouped_by_bar)))
            fragmented_encoding = random.sample(grouped_by_bar, 1)
            fragmented_encoding = [
                item for sublist in fragmented_encoding for item in sublist
            ]

        elif strategy == "random_crop":
            # Select a random start and end index from encoding
            start_index, end_index = random.sample(range(len(encoding)), 2)

            # Ensure start_index is less than or equal to end_index
            start_index, end_index = min(start_index, end_index), max(
                start_index, end_index
            )

            fragmented_encoding = encoding[start_index : end_index + 1]

        start_onset = fragmented_encoding[0][1]
        start_bar = fragmented_encoding[0][0]

        fragmented_encoding = self.reindex_bars(
            fragmented_encoding, start_onset, start_bar=start_bar
        )
        return fragmented_encoding
    
    def apply_corruptions(self, midi_encoding, key_signature, mode):
        # Grouped function types
        function_type_1 = {
            "COR_incorrect_transposition": self.incorrect_transposition,
            "COR_incorrect_inversion": self.incorrect_inversion,
            "COR_note_swapping": self.note_swapping,
            "COR_permute_note_pitch": self.permute_note_pitch,
            "COR_permute_note_duration": self.permute_note_duration,
            "COR_permute_note_pitch_duration": self.permute_note_pitch_duration,
            "COR_melodic_stripping": self.melodic_stripping,
            "COR_melodic_addition": self.melodic_addition,
            "COR_same_note_modification": self.same_note_modification
        }

        function_type_2 = {
            "COR_PITCH_MASK": self.masking,
            "COR_DURATION_MASK": self.masking,
            "COR_BAR_MASK": self.masking
        }

        function_types = [function_type_1, function_type_2]

        # Group midi encoding by bars
        midi_bar_encoding = self.group_by_bar(midi_encoding)
        corruption_tracker = []
        if len(midi_bar_encoding) > 2 and random.random() < 0.5:
            # Apply fragmentation to the midi encoding
            # Choose between bar and random crop
            fragment_type = random.choice(["bar", "random_crop"])
            corrupted_phrase = self.fragment_notes(midi_encoding=midi_encoding, strategy=fragment_type)
            corruption_tracker.append("COR_FRAGMENT_NOTES")
            # Remove function_type_2 from the list of function_types
            function_types.remove(function_type_2)
        else:
            corrupted_phrase = midi_encoding

        if random.random() < 0.2 and function_type_2 in function_types:
            # Choose between masking pitch or duration
            mask_type = random.choice(["pitch", "duration", "bar"])
            if mask_type == "bar":
                corrupted_phrase = function_type_2.get("COR_BAR_MASK")(midi_encoding=corrupted_phrase, mask_type=mask_type)
                corruption_tracker.append("COR_BAR_MASK")
            elif mask_type == "pitch":
                corrupted_phrase = function_type_2.get("COR_PITCH_MASK")(midi_encoding=corrupted_phrase, mask_type=mask_type)
                corruption_tracker.append("COR_PITCH_MASK")
            else:
                corrupted_phrase = function_type_2.get("COR_DURATION_MASK")(midi_encoding=corrupted_phrase, mask_type=mask_type)
                corruption_tracker.append("COR_DURATION_MASK")
        else:
            if function_type_2 in function_types:
                function_types.remove(function_type_2)
            # Choose from a group of functions
            random_corruption = random.choice(function_types)
            
            # Choose a random corruption function to apply
            random_corruption_function = random.choice(list(random_corruption))
    
            # Find the function to apply
            corrupted_phrase = random_corruption.get(random_corruption_function)(midi_encoding=corrupted_phrase, key_signature=key_signature, mode=mode)
            corruption_tracker.append(random_corruption_function)

        return corrupted_phrase, corruption_tracker
    
    def apply_permutation_corruptions(self, midi_encoding):
        # Grouped function types
        function_type_1 = {
            "COR_permute_note_pitch": self.permute_note_pitch,
            "COR_permute_note_duration": self.permute_note_duration,
            "COR_permute_note_pitch_duration": self.permute_note_pitch_duration
        }

        corruption_tracker = []

        # Choose a random corruption function to apply
        random_corruption = random.choice(list(function_type_1))

        # Find the function to apply
        corrupted_phrase = function_type_1.get(random_corruption)(midi_encoding)
        corruption_tracker.append(random_corruption)

        return corrupted_phrase, corruption_tracker





# class NpEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         if isinstance(obj, np.floating):
#             return float(obj)
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return super(NpEncoder, self).default(obj)


# class Orchestrate(Melodic_Development):
#     def __init__(
#         self,
#         config_path="config.yaml",
#         instruments_to_orchestrate={
#             "piano": 0,
#             "strings": 48,
#             "lead": 80,
#             "contrabass": 32,
#             "guitar": 25,
#         },
#     ):
#         super().__init__()

#         with open(config_path, 'r') as file:
#             self.script_config = yaml.safe_load(file)
#         file_name = self.script_config['file_name']
#         midi_obj = miditoolkit.midi.parser.MidiFile(file_name)
#         self.x, self.is_major, self.pitch_shift, self.tpc = MIDI_to_encoding(midi_obj)
#         self.current_instrument = self.get_instrument(self.x)
#         self.instruments_to_orchestrate = instruments_to_orchestrate
#         self.assembler = []
#         # Nested JSON Object for logs
#         self.logs = {'Timestamp': str(datetime.now()), 'Pitch_Masking': dict(), 'Phrases': {}}

#     def get_instrument(self, x):
#         for event in x:
#             instruments = event[2]
#             break
#         return instruments
    
#     def get_other_instruments(self, instruments):
#         other_instruments = [i for i in self.instruments_to_orchestrate.values() if i != instruments]
#         return other_instruments
    
#     def change_range_of_instrument(self, midi_encoding, instrument_number, max_pitch):
#         correct_pitch = True
#         for event in midi_encoding:
#             if event[2] == instrument_number:
#                 if event[3] > max_pitch:
#                     correct_pitch = False
#                     break

#         if not correct_pitch:
#             # Randomly choose how many octaves to transpose the instrument
#             transpose = random.choice([12, 24])
#             for event in midi_encoding:
#                 if event[2] == instrument_number:
#                     event[3] = event[3] - transpose

#         return midi_encoding

#     def generate_harmony_logs(self, log, phrase, instruments):        
#         for ins in instruments:
#             if ins not in log['Pitch_Masking'].keys():
#                 log['Pitch_Masking'][ins] = list(set([bar[0] for bar in phrase]))
#             else:
#                 # Append to the list of bars
#                 log['Pitch_Masking'][ins] += list(set([bar[0] for bar in phrase]))        
#         return log
    
#     def generate_pitch_masking_logs(self, config, phrase):
#         if config.get('pitch_masking') == "first_bar":
#             if self.current_instrument not in self.logs['Pitch_Masking'].keys():
#                 self.logs['Pitch_Masking'][self.current_instrument] = [phrase[0][0]]
#             else:
#                 self.logs['Pitch_Masking'][self.current_instrument] += [phrase[0][0]]
#         elif config.get('pitch_masking') == "last_bar":
#             if self.current_instrument not in self.logs['Pitch_Masking'].keys():
#                 self.logs['Pitch_Masking'][self.current_instrument] = [phrase[-1][0]]
#             else:
#                 self.logs['Pitch_Masking'][self.current_instrument] += [phrase[-1][0]]
#         elif config.get('pitch_masking') == "all_bars":
#             if self.current_instrument not in self.logs['Pitch_Masking'].keys():
#                 self.logs['Pitch_Masking'][self.current_instrument] = list(set([bar[0] for bar in phrase]))
#             else:
#                 self.logs['Pitch_Masking'][self.current_instrument] += list(set([bar[0] for bar in phrase]))
#         else:
#             print("Pitch masking strategy not found. Choose between first_bar, last_bar or all_bars. Skipping pitch masking.")

#     def write_logs(self, file_path, data):
#         # Write the nested data to the JSON file
#         with open(file_path, 'w') as json_file:
#             json.dump(data, json_file, indent=4, cls=NpEncoder)

#     def execute_function(self, config):
#         # Import functions dynamically
#         function_name = config['name']
#         function = getattr(self, function_name, None)
        
#         if function:
#             # Get the transformed phrase based on the config
#             phrase_id = config.get('phrase')
#             if phrase_id > len(self.assembler) or phrase_id < 0:
#                 print("Phrase not found. Using previous phrase.")
#                 transformed_phrase = self.assembler[-1]
#             else:
#                 transformed_phrase = self.assembler[phrase_id]

#             # Get the fragment of the transformed phrase
#             if config.get('fragment'):
#                 transformed_phrase = self.fragment_melody(
#                     midi_encoding=transformed_phrase,
#                     strategy=config.get('fragment')['strategy'],
#                     start_onset=0,
#                     bars_to_skip=0,
#                     include_original=False)

#             # Transform the phrase
#             parameters = config.get('parameters', {})
#             transformed_phrase = function(transformed_phrase, **parameters)

#             # Change instrument
#             if config.get('change_instrument'):
#                 if config.get('change_instrument')['change_to'] is None:
#                     new_instrument = random.choice([ins for ins in self.instruments_to_orchestrate.values() if ins != self.current_instrument])
#                 else:
#                     new_instrument = config.get('change_instrument')['change_to']
#                 transformed_phrase = self.change_instrument(
#                     midi_encoding=transformed_phrase,
#                     change_from=self.current_instrument,
#                     instrument_number=new_instrument,
#                     bars_to_skip=0,
#                     include_original=False,
#                 )
#                 self.current_instrument = new_instrument

#             # Reindex bars
#             if self.assembler[-1][-1][4] > 6:
#                 last_bar_onset_off = self.assembler[-1][-1][1] + self.note_durations[self.assembler[-1][-1][4]]
#             else:
#                 last_bar_onset_off = self.assembler[-1][-1][1] + self.assembler[-1][-1][4]
#             last_bar = self.assembler[-1][-1][0]
#             if last_bar_onset_off % self.beats_in_bar == 0:
#                 additional_bars = last_bar_onset_off // self.beats_in_bar
#             else:
#                 additional_bars = last_bar_onset_off // self.beats_in_bar + 1
#             bars_to_skip = config.get('parameters')['bars_to_skip']
#             start_onset = 0 if config.get('parameters')['start_onset'] is None else config.get('parameters')['start_onset']
#             start_bar = last_bar + additional_bars + bars_to_skip
#             transformed_phrase = self.reindex_bars(transformed_phrase, start_onset=start_onset, start_bar=start_bar)
        
#         else:
#             print(f"Function {function_name} not found.")

#         return transformed_phrase, start_bar, start_onset

#     def orchestrate(self):
#         # Extract phrases for each instrument
#         phrase_1 = self.extract_phrases(self.x, self.current_instrument)
#         transformed_phrase = phrase_1[0]
#         self.assembler.append(transformed_phrase)
#         self.logs['Phrases']['Phrase_0'] = {'Instrument': self.current_instrument, 'Transform': []}
#         print("Number of notes in phrase:", len(transformed_phrase))

#         # # Harmonize the phrase
#         # transformed_phrase = self.harmonize_melody(
#         #     midi_encoding=transformed_phrase,
#         #     instruments_to_harmonize=[ins for ins in self.instruments_to_orchestrate.values() if ins != self.current_instrument],
#         #     start_onset=0,
#         #     bars_to_skip=0,
#         #     include_original=False,
#         # )
#         # self.assembler.append(transformed_phrase)
#         # # Generate logs for the current phrase
#         # self.logs = self.generate_harmony_logs(self.logs, transformed_phrase, [ins for ins in self.instruments_to_orchestrate.values() if ins != self.current_instrument])

#         for n, function_config in enumerate(self.script_config.get('functions', [])):
#             transformed_phrase, start_bar, start_onset = self.execute_function(function_config)
#             # Get current instrument
#             self.current_instrument = self.get_instrument(transformed_phrase)
#             self.assembler.append(transformed_phrase)
#             self.logs['Phrases'][f'Phrase_{n+1}'] = {'Instrument': self.current_instrument, 'Transform': [function_config['name']], 'Start_Bar': start_bar, 'Start_Onset': start_onset}            
#             if function_config.get('pitch_masking'):
#                 self.generate_pitch_masking_logs(function_config, transformed_phrase)

#         # Add all phrases of assembler together
#         self.assembler = [item for sublist in self.assembler for item in sublist]

#         # Change range of instrument
#         self.assembler = self.change_range_of_instrument(self.assembler, instrument_number=32, max_pitch=71)

#         new_midi_obj = encoding_to_MIDI(self.assembler, tpc=[], decode_chord=False)
#         new_midi_obj.dump("melodic_development_template.mid")
#         self.write_logs("instruction.json", self.logs)


if __name__ == "__main__":

    # orchestrate_obj = Orchestrate(config_path="templates/piano_to_piano.yaml") # template_1.yaml
    # orchestrate_obj.orchestrate()


# if __name__ == "__main__":
#     # Specify the path to the JSON file
    # Folder path
    folder_path = os.path.normpath("data/extracted_phrases")
    # Get random file from the folder
    file_path = random.choice([os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    # file_path = os.path.normpath("C:/Users/Keshav/Desktop/QMUL/Research/melodic-development/phrase_refinement/extracted_phrases/extracted_phrases_Wikifonia/_Louis Armstrong, kenny ball - Someday You'll Be Sorry_mono.mid.json")

    # Open the JSON file and load its contents
    with open(file_path, "r") as file:
        data = json.load(file)
    
    print(data["phrases"]['0'][0])
    print(data['metadata']["key_signature"])
    print(data['metadata']["major_or_minor"])
    print(data['metadata']["time_signature"])

    # Find beats in bar
    beats_in_bar = find_beats_in_bar(data['metadata']["time_signature"])

    # Load the phrase corruption class
    phrase_corruption = Phrase_Corruption(beats_in_bar)
    corrupted_phrase = phrase_corruption.melodic_addition(data["phrases"]['0'][0], data['metadata']["key_signature"], data['metadata']["major_or_minor"])
    print(corrupted_phrase)

#     print(data["phrases"]['3'])
#     tempo_location = data["metadata"]["tempo"]
#     data = data["phrases"]['3'][0]
#     corruption_name = ""
#     # corrupted_phrase = phrase_corruption.same_note_modification(data["phrases"]['2'])
#     # corrupted_phrase = phrase_corruption.note_swapping(data["phrases"]['2'])
#     # corrupted_phrase = phrase_corruption.melodic_stripping(data["phrases"]['2'])
#     # corrupted_phrase = phrase_corruption.incorrect_inversion(data["phrases"]['2'])
#     corrupted_phrase = phrase_corruption.incorrect_transposition(data)
#     # corrupted_phrase = phrase_corruption.masking(data["phrases"]['2'], mask_type="bar")
#     # corrupted_phrase = phrase_corruption.masking(data["phrases"]['2'], mask_type="pitch")
#     # corrupted_phrase = phrase_corruption.masking(data["phrases"]['2'], mask_type="duration")
#     # corrupted_phrase = phrase_corruption.permute_note_pitch(data["phrases"]['2'])
#     # corrupted_phrase = phrase_corruption.permute_note_duration(data["phrases"]['2'])
#     # corrupted_phrase = phrase_corruption.permute_note_pitch_duration(data["phrases"]['2'])
#     # corrupted_phrase, corruption_name = phrase_corruption.apply_corruptions(data["phrases"]['2'])
#     print(corrupted_phrase, corruption_name)
#     if corrupted_phrase == data:
#         print("Identical")

#     # # Get a flattened list of all the notes in the phrases
#     # notes = [note for phrase in data["phrases"].values() for note in phrase]

#     # duration_mapping = data["metadata"]["duration_mapping"]
#     reverse_duration_mapping = {v: k for k, v in duration_mapping.items()}
#     new_tokens = list_to_remi_encoding(corrupted_phrase, tempo_location, reverse_duration_mapping)
#     print(new_tokens)

#     # # Specify the path to the pickle file
#     # file_path = "phrase_refinement/tokenizer.pickle"

#     # # Open the pickle file in read mode
#     # with open(file_path, "rb") as file:
#     #     tokenizer = pickle.load(file)

#     # new_tokens = TokSequence(tokens=new_tokens)
#     # new_midi = tokenizer([new_tokens])
#     # new_midi.dump("writing_test.mid")