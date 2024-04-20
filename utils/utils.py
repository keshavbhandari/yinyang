import math
import pickle
import random
import copy
import numpy as np
import miditoolkit
from music21 import stream, meter, note, metadata, tempo, converter, instrument, chord

def find_beats_in_bar(time_signature):
    if time_signature == "null" or time_signature is None:
        time_signature = "4/4"
    numerator = int(time_signature.split("_")[1].split("/")[0])
    denominator = int(time_signature.split("_")[1].split("/")[1])
    if denominator == 4:
        beats_in_bar = numerator * (denominator / 8) * 2
    elif denominator == 8:
        beats_in_bar = numerator * (denominator / 8) / 2
    elif denominator == 2:
        beats_in_bar = numerator * (denominator / 8) * 8
    elif denominator == 1:
        beats_in_bar = numerator * (denominator / 8) * 32
    return beats_in_bar

def annotation_to_encoding(annotation_file):
    key_signature = annotation_file['features']['tonic'][0]
    major_or_minor = annotation_file['features']['mode'][0]

    time_signature = annotation_file['features']['timesignature'][0]
    if time_signature == "null" or time_signature is None:
        time_signature = "4/4"
    numerator = int(time_signature.split("/")[0])
    denominator = int(time_signature.split("/")[1])
    if denominator == 4:
        beats_in_bar = numerator * (denominator / 8) * 2
    elif denominator == 8:
        beats_in_bar = numerator * (denominator / 8) / 2
    elif denominator == 2:
        beats_in_bar = numerator * (denominator / 8) * 8
    elif denominator == 1:
        beats_in_bar = numerator * (denominator / 8) * 32
    else: 
        return [], time_signature, key_signature, major_or_minor
    
    pitches = annotation_file['features']['midipitch']
    durations = annotation_file['features']['duration']
    next_note_rest_value = annotation_file['features']['restduration_frac']

    encoding = []
    bar = 0
    onset = 0
    for idx, pitch_value in enumerate(pitches):
        note_info = []
        if idx == 0:
            note_info.append([bar, onset, 0, pitches[idx], durations[idx], 91])
        else:
            # Check if previous note was a rest
            prev_rest = next_note_rest_value[idx-1]
            if prev_rest is None:
                rest = 0
            else:            
                if "/" in prev_rest:
                    rest = float(int(prev_rest.split("/")[0]) / int(prev_rest.split("/")[1]))
                else:
                    rest = int(prev_rest)
            
            onset += durations[idx-1] + rest  
            
            if onset >= beats_in_bar:
                previous_onset = encoding[-1][1]
                onset = (previous_onset + durations[idx-1] + rest) % beats_in_bar
                bar += 1
            note_info.append([bar, onset, 0, pitches[idx], durations[idx], 91])
        encoding+=note_info
    
    return encoding, time_signature, key_signature, major_or_minor


def encoding_to_midi(encoding, tempo_dict, time_signature, midi_file_path="output.mid"):
    time_signature = time_signature.split("_")[1]

    # Create a Score
    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.title = "Your MIDI Score"

    # Create a Part for the instrument
    part = stream.Part()

    # Set the initial tempo
    initial_tempo = tempo_dict.get(0, 120)
    part.append(tempo.MetronomeMark(number=initial_tempo))

    # Set the time signature
    time_signature = meter.TimeSignature(time_signature)

    # Add the time signature to the Part
    part.append(time_signature)

    # Iterate through the MIDI data and create Note objects
    for entry in encoding:
        bar_number, onset_position, instrument_number, pitch, duration, velocity = entry[:6]

        # Create a Note
        n = note.Note(pitch, quarterLength=duration)
        n.volume.velocity = velocity

        # Calculate the offset position
        offset_position = bar_number * time_signature.barDuration.quarterLength + onset_position

        # Add the Note to the Part at the calculated offset position
        part.insert(offset_position, n)

        # Check if there is a tempo change for the next bar
        next_tempo = tempo_dict.get(bar_number + 1, None)
        if next_tempo is not None:
            part.append(tempo.MetronomeMark(number=next_tempo))

    # Add the Part to the Score
    score.append(part)

    # Write the Score to a MIDI file
    # midi_file_path = "output.mid"
    score.write('midi', fp=midi_file_path)



duration_mapping = {'0.1.8': 1, '0.2.8': 2, '0.3.8': 3, '0.4.8': 4, '0.5.8': 5, '0.6.8': 6, '0.7.8': 7, '1.0.8': 8,
                    '1.1.8': 9, '1.2.8': 10, '1.3.8': 11, '1.4.8': 12, '1.5.8': 13, '1.6.8': 14, '1.7.8': 15,
                    '2.0.8': 16, '2.1.8': 17, '2.2.8': 18, '2.3.8': 19, '2.4.8': 20, '2.5.8': 21, '2.6.8': 22,
                    '2.7.8': 23, '3.0.8': 24, '3.1.8': 25, '3.2.8': 26, '3.3.8': 27, '3.4.8': 28, '3.5.8': 29,
                    '3.6.8': 30, '3.7.8': 31, '4.0.4': 32, '4.1.4': 34, '4.2.4': 36, '4.3.4': 38, '5.0.4': 40,
                    '5.1.4': 42, '5.2.4': 44, '5.3.4': 46, '6.0.4': 48, '6.1.4': 50, '6.2.4': 52, '6.3.4': 54,
                    '7.0.4': 56, '7.1.4': 58, '7.2.4': 60, '7.3.4': 62, '8.0.4': 64, '8.1.4': 66, '8.2.4': 68,
                    '8.3.4': 70, '9.0.4': 72, '9.1.4': 74, '9.2.4': 76, '9.3.4': 78, '10.0.4': 80, '10.1.4': 82,
                    '10.2.4': 84, '10.3.4': 86, '11.0.4': 88, '11.1.4': 90, '11.2.4': 92, '11.3.4': 94, '12.0.4': 96}

reverse_duration_mapping = {v: k for k, v in duration_mapping.items()}

def find_key(duration_value):    
    for key in duration_mapping.keys():
        if duration_mapping[key] == duration_value:
            return key


pos_resolution = 4 # 16  # per beat (quarter note)
bar_max = 32
velocity_quant = 4
tempo_quant = 12  # 2 ** (1 / 12)
min_tempo = 16
max_tempo = 256
duration_max = 4  # 2 ** 8 * beat
max_ts_denominator = 6  # x/1 x/2 x/4 ... x/64
max_notes_per_bar = 1  # 1/64 ... 128/64 #
beat_note_factor = 4  # In MIDI format a note is always 4 beats
deduplicate = True
filter_symbolic = False
filter_symbolic_ppl = 16
trunc_pos = 2 ** 16  # approx 30 minutes (1024 measures)
sample_len_max = 1024  # window length max
sample_overlap_rate = 1.5
ts_filter = True
pool_num = 200
max_inst = 127
max_pitch = 127
max_velocity = 127
tracks_start = [16, 144, 997, 5366, 6921, 10489]
tracks_end = [143, 996, 5365, 6920, 10488, 11858]


inst_to_row = { '80':0, '32':1, '128':2,  '25':3, '0':4, '48':5, '129':6}
prog_to_abrv = {'0':'P','25':'G','32':'B','48':'S','80':'M','128':'D'}
track_name = ['lead', 'bass', 'drum', 'guitar', 'piano', 'string']

root_dict = {'C': 0, 'C#': 1, 'D': 2, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11}
kind_dict = {'null': 0, 'm': 1, '+': 2, 'dim': 3, 'seven': 4, 'maj7': 5, 'm7': 6, 'm7b5': 7}
root_list = list(root_dict.keys())
kind_list = list(kind_dict.keys())

_CHORD_KIND_PITCHES = {
    'null': [0, 4, 7],
    'm': [0, 3, 7],
    '+': [0, 4, 8],
    'dim': [0, 3, 6],
    'seven': [0, 4, 7, 10],
    'maj7': [0, 4, 7, 11],
    'm7': [0, 3, 7, 10],
    'm7b5': [0, 3, 6, 10],
}

ts_dict = dict()
ts_list = list()
for i in range(0, max_ts_denominator + 1):  # 1 ~ 64
    for j in range(1, ((2 ** i) * max_notes_per_bar) + 1):
        ts_dict[(j, 2 ** i)] = len(ts_dict)
        ts_list.append((j, 2 ** i))
dur_enc = list()
dur_dec = list()
for i in range(duration_max):
    for j in range(pos_resolution):
        dur_dec.append(len(dur_enc))
        for k in range(2 ** i):
            dur_enc.append(len(dur_dec) - 1)

tokens_to_ids = {}
ids_to_tokens = []
pad_index = None
empty_index = None


key_profile = pickle.load(open('key_profile.pickle', 'rb'))

pos_in_bar = beat_note_factor * max_notes_per_bar * pos_resolution


def normalize_to_c_major(e):
    def get_pitch_class_histogram(notes, use_duration=True, use_velocity=True, normalize=True):
        weights = np.ones(len(notes))
        if use_duration:
            weights *= [note[4] for note in notes]
        if use_velocity:
            weights *= [note[5] for note in notes]
        histogram, _ = np.histogram([note[3] % 12 for note in notes], bins=np.arange(
            13), weights=weights, density=normalize)
        if normalize:
            histogram /= (histogram.sum() + (histogram.sum() == 0))
        return histogram

    pitch_histogram = [i for i in e if i[2] < 128]
    if len(pitch_histogram) == 0:
        return e, True, 0

    histogram = get_pitch_class_histogram(pitch_histogram)
    key_candidate = np.dot(key_profile, histogram)
    key_temp = np.where(key_candidate == max(key_candidate))
    major_index = key_temp[0][0]
    minor_index = key_temp[0][1]
    major_count = histogram[major_index]
    minor_count = histogram[minor_index % 12]
    key_number = 0
    if major_count < minor_count:
        key_number = minor_index
        is_major = False
    else:
        key_number = major_index
        is_major = True
    real_key = key_number
    # transposite to C major or A minor
    if real_key <= 11:
        trans = 0 - real_key
    else:
        trans = 21 - real_key
    pitch_shift = trans

    e = [list(k + pitch_shift if j == 3 and i[2] != 128 else k for j, k in enumerate(i))
         for i in e]
    return e, is_major, pitch_shift

def t2e(x):
    assert x in ts_dict, 'unsupported time signature: ' + str(x)
    return ts_dict[x]

def e2t(x):
    return ts_list[x]

def d2e(x):
    return dur_enc[x] if x < len(dur_enc) else dur_enc[-1]

def e2d(x):
    return dur_dec[x] if x < len(dur_dec) else dur_dec[-1]

def v2e(x):
    return x // velocity_quant

def e2v(x):
    return (x * velocity_quant) + (velocity_quant // 2)

def b2e(x):
    x = max(x, min_tempo)
    x = min(x, max_tempo)
    x = x / min_tempo
    e = round(math.log2(x) * tempo_quant)
    return e

def e2b(x):
    return math.floor(2 ** (x / tempo_quant) * min_tempo)

def time_signature_reduce(numerator, denominator):
    while denominator > 2 ** max_ts_denominator and denominator % 2 == 0 and numerator % 2 == 0:
        denominator //= 2
        numerator //= 2
    while numerator > max_notes_per_bar * denominator:
        for i in range(2, numerator + 1):
            if numerator % i == 0:
                numerator //= i
                break
    return numerator, denominator


def MIDI_to_encoding(midi_obj):
    def time_to_pos(t):
        return round(t * pos_resolution / midi_obj.ticks_per_beat)
    notes_start_pos = [time_to_pos(j.start)
                       for i in midi_obj.instruments for j in i.notes]
    if len(notes_start_pos) == 0:
        return list()
    max_pos = max(notes_start_pos) + 1

    pos_to_info = [[None for _ in range(4)] for _ in range(max_pos)]
    tsc = midi_obj.time_signature_changes
    tpc = midi_obj.tempo_changes
    for i in range(len(tsc)):
        for j in range(time_to_pos(tsc[i].time), time_to_pos(tsc[i + 1].time) if i < len(tsc) - 1 else max_pos):
            if j < len(pos_to_info):
                pos_to_info[j][1] = t2e(time_signature_reduce(
                    tsc[i].numerator, tsc[i].denominator))
    for i in range(len(tpc)):
        for j in range(time_to_pos(tpc[i].time), time_to_pos(tpc[i + 1].time) if i < len(tpc) - 1 else max_pos):
            if j < len(pos_to_info):
                pos_to_info[j][3] = b2e(tpc[i].tempo)
    for j in range(len(pos_to_info)):
        if pos_to_info[j][1] is None:
            # MIDI default time signature
            pos_to_info[j][1] = t2e(time_signature_reduce(4, 4))
        if pos_to_info[j][3] is None:
            pos_to_info[j][3] = b2e(120.0)  # MIDI default tempo (BPM)
    cnt = 0
    bar = 0
    measure_length = None
    for j in range(len(pos_to_info)):
        ts = e2t(pos_to_info[j][1])
        if cnt == 0:
            measure_length = ts[0] * beat_note_factor * pos_resolution // ts[1]
        pos_to_info[j][0] = bar
        pos_to_info[j][2] = cnt
        cnt += 1
        if cnt >= measure_length:
            assert cnt == measure_length, 'invalid time signature change: pos = {}'.format(
                j)
            cnt -= measure_length
            bar += 1
    encoding = []

    for inst in midi_obj.instruments:
        for note in inst.notes:
            if time_to_pos(note.start) >= trunc_pos:
                continue

            info = pos_to_info[time_to_pos(note.start)]
            duration = d2e(time_to_pos(note.end) - time_to_pos(note.start))
            encoding.append([info[0], info[2], max_inst + 1 if inst.is_drum else inst.program, note.pitch + max_pitch +
                            1 if inst.is_drum else note.pitch, duration, v2e(note.velocity), info[1], info[3]])
    if len(encoding) == 0:
        return list()

    encoding.sort()
    _, is_major, pitch_shift = normalize_to_c_major(encoding)
    pitch_shift = 0

    return encoding, is_major, pitch_shift, tpc


def encoding_to_MIDI(encoding, tpc, decode_chord):
    
    bar_to_timesig = [list()
                      for _ in range(max(map(lambda x: x[0], encoding)) + 1)]
    for i in encoding:
        bar_to_timesig[i[0]].append(i[6])
    bar_to_timesig = [max(set(i), key=i.count) if len(
        i) > 0 else None for i in bar_to_timesig]
    for i in range(len(bar_to_timesig)):
        if bar_to_timesig[i] is None:
            bar_to_timesig[i] = t2e(time_signature_reduce(
                4, 4)) if i == 0 else bar_to_timesig[i - 1]
    bar_to_pos = [None] * len(bar_to_timesig)
    cur_pos = 0
    for i in range(len(bar_to_pos)):
        bar_to_pos[i] = cur_pos
        ts = e2t(bar_to_timesig[i])
        measure_length = ts[0] * beat_note_factor * pos_resolution // ts[1]
        cur_pos += measure_length
    pos_to_tempo = [list() for _ in range(
        cur_pos + max(map(lambda x: x[1], encoding)))]
    for i in encoding:
        pos_to_tempo[bar_to_pos[i[0]] + i[1]].append(i[7])
    pos_to_tempo = [round(sum(i) / len(i)) if len(i) >
                    0 else None for i in pos_to_tempo]
    for i in range(len(pos_to_tempo)):
        if pos_to_tempo[i] is None:
            pos_to_tempo[i] = b2e(120.0) if i == 0 else pos_to_tempo[i - 1]
 
    midi_obj = miditoolkit.midi.parser.MidiFile()
    midi_obj.tempo_changes = tpc

    def get_tick(bar, pos):
        return (bar_to_pos[bar] + pos) * midi_obj.ticks_per_beat // pos_resolution
    midi_obj.instruments = [miditoolkit.containers.Instrument(program=(
        0 if i == 128 else i), is_drum=(i == 128), name=str(i)) for i in range(128 + 1)]

    for i in encoding:
        start = get_tick(i[0], i[1])
        program = i[2]

        if program == 129 and decode_chord:
            root_name = root_list[i[3]]
            kind_name = kind_list[i[4]]
            root_pitch_shift = root_dict[root_name]
            end = start + get_tick(0, e2d(1))
            for kind_shift in _CHORD_KIND_PITCHES[kind_name]:
                pitch = 36 + root_pitch_shift + kind_shift
                midi_obj.instruments[1].notes.append(miditoolkit.containers.Note(
                start=start, end=end, pitch=pitch, velocity=e2v(20)))
        elif program != 129:
            pitch = (i[3] - 128 if program == 128 else i[3])
            if pitch < 0:
                continue
            duration = get_tick(0, e2d(i[4]))
            if duration == 0:
                duration = 1
            end = start + duration
            velocity = e2v(i[5])

            midi_obj.instruments[program].notes.append(miditoolkit.containers.Note(
                start=start, end=end, pitch=pitch, velocity=velocity))
    midi_obj.instruments = [
        i for i in midi_obj.instruments if len(i.notes) > 0]
    cur_ts = None
    for i in range(len(bar_to_timesig)):
        new_ts = bar_to_timesig[i]
        if new_ts != cur_ts:
            numerator, denominator = e2t(new_ts)
            midi_obj.time_signature_changes.append(miditoolkit.containers.TimeSignature(
                numerator=numerator, denominator=denominator, time=get_tick(i, 0)))
            cur_ts = new_ts
    cur_tp = None
    for i in range(len(pos_to_tempo)):
        new_tp = pos_to_tempo[i]
        if new_tp != cur_tp:
            tempo = e2b(new_tp)
            midi_obj.tempo_changes.append(
                miditoolkit.containers.TempoChange(tempo=tempo, time=get_tick(0, i)))
            cur_tp = new_tp
    return midi_obj


def remi_to_list_encoding(remi_encoding):
    tokens = copy.deepcopy(remi_encoding)
    midi_encoding = []
    tempo_location = {}

    bar_number = -1
    note_info = {}
    tokens[0].tokens.append("Bar_None")
    tokens[0].tokens.append("Position_0")
    tokens[0].tokens.append("Program_0")
    for token in tokens[0].tokens:
        if token == "Bar_None":
            bar_number += 1    
        elif token.startswith("Position_"):
            position = int(token.split("_")[1])
            if len(note_info.keys())>2:
                midi_encoding.append(note_info) 
                note_info = {} 
                note_info['bar'] = bar_number
                note_info['position'] = position
            else:        
                note_info['bar'] = bar_number
                note_info['position'] = position
        elif token.startswith("Tempo_"):
            tempo_location[bar_number] = token.split("_")[1]
        elif token.startswith("Program_"):
            program = int(token.split("_")[1])
            note_info['program'] = program
        elif token.startswith("Pitch_"):
            pitch = int(token.split("_")[1])
            note_info['pitch'] = pitch
        elif token.startswith("Velocity_"):
            velocity = int(token.split("_")[1])
            note_info['velocity'] = velocity
        elif token.startswith("Duration_"):
            # duration_value = duration_mapping[token.split("_")[1]]
            # note_info['duration'] = duration_value
            # note_info['duration_string'] = token.split("_")[1]
            note_info['duration'] = float(token.split("_")[1])

    encoding = []
    for n, dict in enumerate(midi_encoding):
        note_info = []
        note_info.append(dict['bar'])
        note_info.append(dict['position'])
        note_info.append(dict['program'])
        note_info.append(dict['pitch'])
        note_info.append(dict['duration'])
        note_info.append(dict['velocity'])
        # note_info.append(dict['duration_string'])

        encoding.append(note_info)
        
    return encoding, tempo_location

def string_to_list_encoding(remi_encoding):
    tokens = copy.deepcopy(remi_encoding)
    midi_encoding = []

    bar_number = -1
    note_info = {}
    for token in tokens:
        if token == "Bar_None":
            bar_number += 1    
        elif token.startswith("Position_"):
            position = float(token.split("_")[1])
            if len(note_info.keys())>2:
                midi_encoding.append(note_info) 
                note_info = {} 
                note_info['bar'] = bar_number
                note_info['position'] = position
            else:        
                note_info['bar'] = bar_number
                note_info['position'] = position
        elif token.startswith("Program_"):
            program = int(token.split("_")[1])
            note_info['program'] = program
        elif token.startswith("Pitch_"):
            pitch = int(token.split("_")[1])
            note_info['pitch'] = pitch
        elif token.startswith("Velocity_"):
            velocity = int(token.split("_")[1])
            note_info['velocity'] = velocity
        elif token.startswith("Duration_"):
            note_info['duration'] = float(token.split("_")[1])

    if len(note_info.keys())>0:
        midi_encoding.append(note_info)

    # Add duration and pitch to dictionary if not present. Duration should be next note's position - current note's position. Pitch should be next note's pitch or first note's pitch if it's the last note
    for n, dict in enumerate(midi_encoding):
        # See if duration is not present first
        if 'duration' not in dict.keys():
            if n < len(midi_encoding)-1:
                midi_encoding[n]['duration'] = midi_encoding[n+1]['position'] - midi_encoding[n]['position']
            else:
                midi_encoding[n]['duration'] = 1.0
        # See if pitch is not present first
        if 'pitch' not in dict.keys():
            if n < len(midi_encoding)-1:
                midi_encoding[n]['pitch'] = midi_encoding[n+1]['pitch']
            else:
                midi_encoding[n]['pitch'] = midi_encoding[0]['pitch']

    encoding = []
    for n, dict in enumerate(midi_encoding):
        note_info = []
        if 'bar' not in dict.keys():
            # Take the bar number from the previous note if it exists. Otherwise set it to 0
            if n > 0:
                dict['bar'] = midi_encoding[n-1]['bar']
            else:
                dict['bar'] = 0
        note_info.append(dict['bar'])
        note_info.append(dict['position'])
        note_info.append(0)
        note_info.append(dict['pitch'])
        note_info.append(dict['duration'])
        note_info.append(91)
        # note_info.append(dict['duration_string'])

        encoding.append(note_info)
        
    return encoding

def list_to_remi_encoding(encoding, tempo_location, time_signature):
    current_bar = encoding[0][0]
    reverse_encoding = []
    for n, note in enumerate(encoding):
        note_info = []
        if str(note) == "SEP" or any(str(note).startswith(prefix) for prefix in ["PL_", "PP_", "COR_", "TimeSig_", "KS_", "MM_", "CA_", "PR_"]):
            reverse_encoding += [note]
            continue
        elif str(note[1])=="BAR_MASK":
            reverse_encoding += ["BAR_MASK"]
            continue
        if n == 0:
            bar = 'Bar_None'
            note_info.append(bar)
            note_info.append(time_signature)
            current_bar = int(note[0])
        elif note[0] > current_bar:
            bar_diff = note[0] - current_bar
            for i in range(int(bar_diff)):
                bar = 'Bar_None'
                note_info.append(bar)
                note_info.append(time_signature)
                current_bar = note[0]

        note_info.append(f'Position_{note[1]}')
        if "Bar_None" in note_info and str(current_bar) in tempo_location.keys():
            note_info.append(f'Tempo_{tempo_location[str(current_bar)]}')
        note_info.append(f'Program_{note[2]}')
        if "PITCH_MASK" in str(note[3]):
            note_info.append("PITCH_MASK")
        else:
            note_info.append(f'Pitch_{note[3]}')
        note_info.append(f'Velocity_{note[5]}')
        if "DURATION_MASK" in str(note[4]):
            note_info.append("DURATION_MASK")
        else:
            note_info.append(f'Duration_{note[4]}')

        reverse_encoding += note_info
    return reverse_encoding

def list_to_cp_encoding(encoding, tempo_location, time_signature):
    current_bar = encoding[0][0]
    reverse_encoding = []
    for n, note in enumerate(encoding):
        if n == 0:
            note_info = []
            note_info.append("Family_Metric")
            note_info.append("Bar_None")
            note_info.append("Ignore_None")
            note_info.append("Ignore_None")
            note_info.append("Ignore_None")
            note_info.append("Ignore_None")
            note_info.append(time_signature)
            reverse_encoding.append(note_info)
            current_bar = int(note[0])
        elif note[0] > current_bar:
            bar_diff = note[0] - current_bar
            for i in range(int(bar_diff)):
                note_info = []
                note_info.append("Family_Metric")
                note_info.append("Bar_None")
                note_info.append("Ignore_None")
                note_info.append("Ignore_None")
                note_info.append("Ignore_None")
                note_info.append("Ignore_None")
                note_info.append(time_signature)
                reverse_encoding.append(note_info)
                current_bar = note[0]
        
        note_info = []
        note_info.append("Family_Metric")
        note_info.append(f'Position_{note[1]}')
        note_info.append("Ignore_None")
        note_info.append("Ignore_None")
        note_info.append("Ignore_None")        
        # if note_info[1].startswith("Position"):
        #     note_info.append(f'Tempo_{tempo_location[str(0)]}')
        # else:
        note_info.append("Ignore_None")
        note_info.append("Ignore_None")
        reverse_encoding.append(note_info)
        
        note_info = []
        note_info.append("Family_Note")
        note_info.append("Ignore_None")
        note_info.append(f'Pitch_{note[3]}')
        note_info.append(f'Velocity_{note[5]}')
        note_info.append(f'Duration_{note[4]}')
        note_info.append("Ignore_None")
        note_info.append("Ignore_None")
        reverse_encoding.append(note_info)

    return reverse_encoding

def cp_to_list_encoding(cp_encoding):
    encoding = copy.deepcopy(cp_encoding)

    midi_encoding = []
    bar_number = -1

    for n, token in enumerate(encoding):
        if token[1] == "Bar_None" and token[-1].startswith("TimeSig_"):
            bar_number += 1
            note_info = {}
        elif token[1].startswith("Position_") and token[0] == "Family_Metric":
            position = float(token[1].split("_")[1])
            # note_info = {}
            note_info['bar'] = bar_number
            note_info['position'] = position
        elif token[1] == "Ignore_None" and token[0].startswith("Family_Note"):
            note_info['program'] = 0
            note_info['pitch'] = int(token[2].split("_")[1])
            note_info['velocity'] = int(token[3].split("_")[1]) if token[3] != "Ignore_None" else 91
            note_info['duration'] = float(token[4].split("_")[1])
            midi_encoding.append(note_info)
            note_info = {}

    # Append position and program to the dictionary at the end if not present
    if len(note_info.keys())>0 and 'position' in note_info.keys():
        note_info['program'] = 0
        note_info['position'] = position
        midi_encoding.append(note_info)
    
    # Make sure length of each dict in midi_encoding is 6. If it isn't, then remove the dict
    filtered_encoding = []
    for dict in midi_encoding:
        if len(dict.keys()) > 5:
            filtered_encoding.append(dict)  

    list_encoding = []
    for n, dict in enumerate(filtered_encoding):
        note_info = []
        # If bar is not in dict then take the previous bar
        if 'bar' not in dict.keys():
            dict['bar'] = filtered_encoding[n-1]['bar']
        else:
            note_info.append(dict['bar'])
        note_info.append(dict['position'])
        note_info.append(dict['program'])
        note_info.append(dict['pitch'])
        note_info.append(dict['duration'])
        note_info.append(dict['velocity'])
        list_encoding.append(note_info)
        
    return list_encoding


def parse_midi(file_path):
    midi_data = []

    # Load MIDI file
    midi_stream = converter.parse(file_path)

    # Extract instrument parts
    parts = instrument.partitionByInstrument(midi_stream)

    if parts:
        # Iterate through each instrument part
        for part in parts.parts:
            # Reset bar number for each part
            current_bar_number = 0
            # Iterate through each note in the part
            for element in part.recurse():
                if isinstance(element, note.Note) or isinstance(element, chord.Chord):
                    # Calculate the bar number for each note/chord based on the time signature
                    offset_in_beats = element.getOffsetInHierarchy(part)  # Get offset in beats relative to the part
                    time_signature = part.getTimeSignatures()[0] if part.getTimeSignatures() else None
                    if time_signature:
                        time_sign = f"TimeSig_{time_signature.numerator}/{time_signature.denominator}"
                        beats_per_bar = find_beats_in_bar(time_sign)
                        bar_number = int(offset_in_beats / beats_per_bar)
                    else:
                        beats_per_bar = 4
                        bar_number = int(offset_in_beats / beats_per_bar)
                    current_bar_number = bar_number
                    # If it's a Note or Chord, extract relevant information and append to midi_data
                    if isinstance(element, note.Note):
                        midi_data.append([bar_number, offset_in_beats % beats_per_bar, 0, element.pitch.midi, element.duration.quarterLength, element.volume.velocity])
                    elif isinstance(element, chord.Chord):
                        continue
    return midi_data, time_sign
