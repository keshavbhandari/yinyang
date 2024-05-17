import os
import sys
from aria.tokenizer import AbsTokenizer
from aria.data.midi import MidiDict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.utils import flatten, skyline

generated_sequences = [('piano', 60, 15), ('onset', 0), ('dur', 470), ('piano', 48, 30), ('onset', 0), ('dur', 470), ('piano', 57, 15), ('onset', 20), ('dur', 460), ('piano', 65, 45), ('onset', 500), ('dur', 470), ('piano', 53, 30), ('onset', 520), ('dur', 450), ('piano', 60, 15), ('onset', 540), ('dur', 430), ('piano', 65, 45), ('onset', 1000), ('dur', 240), ('piano', 53, 30), ('onset', 1020), ('dur', 200), ('piano', 60, 45), ('onset', 1020), ('dur', 200), ('piano', 67, 60), ('onset', 1250), ('dur', 240), ('piano', 58, 30), ('onset', 1270), ('dur', 120), ('piano', 53, 30), ('onset', 1280), ('dur', 140), ('piano', 69, 60), ('onset', 1500), ('dur', 240), ('piano', 60, 45), ('onset', 1510), ('dur', 210), ('piano', 53, 30), ('onset', 1520), ('dur', 170), ('piano', 70, 60), ('onset', 1750), ('dur', 240), ('piano', 60, 30), ('onset', 1780), ('dur', 240), ('piano', 53, 45), ('onset', 1790), ('dur', 170), ('piano', 72, 60), ('onset', 2000), ('dur', 470), ('piano', 60, 45), ('onset', 2010), ('dur', 460), ('piano', 53, 30), ('onset', 2030), ('dur', 430), ('piano', 69, 45), ('onset', 2500), ('dur', 470), ('piano', 65, 45), ('onset', 2520), ('dur', 450), ('piano', 53, 30), ('onset', 2540), ('dur', 430), ('piano', 72, 60), ('onset', 3000), ('dur', 240), ('piano', 53, 30), ('onset', 3010), ('dur', 230), ('piano', 60, 45), ('onset', 3010), ('dur', 250), ('piano', 72, 60), ('onset', 3250), ('dur', 240), ('piano', 65, 45), ('onset', 3250), ('dur', 230), ('piano', 57, 45), ('onset', 3280), ('dur', 220), ('piano', 72, 60), ('onset', 3500), ('dur', 470), ('piano', 53, 30), ('onset', 3510), ('dur', 460), ('piano', 60, 30), ('onset', 3530), ('dur', 440), ('piano', 70, 60), ('onset', 4000), ('dur', 240), ('piano', 65, 30), ('onset', 4020), ('dur', 190), ('piano', 53, 30), ('onset', 4030), ('dur', 220), ('piano', 70, 60), ('onset', 4250), ('dur', 240), ('piano', 65, 60), ('onset', 4250), ('dur', 240), ('piano', 53, 45), ('onset', 4270), ('dur', 70), ('piano', 69, 45), ('onset', 4500), ('dur', 240), ('piano', 53, 30), ('onset', 4510), ('dur', 300), ('piano', 72, 60), ('onset', 4750), ('dur', 240), ('piano', 53, 30), ('onset', 4770), ('dur', 210), ('piano', 60, 30), ('onset', 4780), ('dur', 180), '<T>', ('piano', 72, 45), ('onset', 0), ('dur', 950), ('piano', 65, 45), ('onset', 30), ('dur', 910), ('piano', 57, 45), ('onset', 30), ('dur', 910), ('piano', 60, 30), ('onset', 1000), ('dur', 470), ('piano', 48, 30), ('onset', 1020), ('dur', 560), ('piano', 57, 30), ('onset', 1030), ('dur', 450), ('piano', 60, 30), ('onset', 1490), ('dur', 470), ('piano', 57, 30), ('onset', 1500), ('dur', 470), ('piano', 65, 45), ('onset', 1500), ('dur', 470), ('piano', 65, 60), ('onset', 2000), ('dur', 470), ('piano', 53, 45), ('onset', 2030), ('dur', 450), ('piano', 60, 60), ('onset', 2030), ('dur', 450), ('piano', 67, 75), ('onset', 2500), ('dur', 240), ('piano', 50, 45), ('onset', 2520), ('dur', 210), ('piano', 58, 45), ('onset', 2520), ('dur', 210), ('piano', 69, 60), ('onset', 2750), ('dur', 240), ('piano', 60, 45), ('onset', 2770), ('dur', 190), ('piano', 53, 45), ('onset', 2780), ('dur', 180), ('piano', 70, 75), ('onset', 3000), ('dur', 240), ('piano', 62, 60), ('onset', 3000), ('dur', 130), ('piano', 53, 45), ('onset', 3010), ('dur', 130), ('piano', 72, 75), ('onset', 3250), ('dur', 240), ('piano', 65, 45), ('onset', 3270), ('dur', 240), ('piano', 57, 45), ('onset', 3280), ('dur', 220), ('piano', 69, 75), ('onset', 3500), ('dur', 470), ('piano', 48, 45), ('onset', 3510), ('dur', 470), ('piano', 65, 60), ('onset', 4000), ('dur', 190), ('piano', 72, 75), ('onset', 4000), ('dur', 240), ('piano', 53, 45), ('onset', 4030), ('dur', 180), ('piano', 72, 75), ('onset', 4250), ('dur', 240), ('piano', 63, 60), ('onset', 4270), ('dur', 240), ('piano', 53, 45), ('onset', 4290), ('dur', 200), ('piano', 72, 60), ('onset', 4500), ('dur', 240), ('piano', 63, 60), ('onset', 4510), ('dur', 180), ('piano', 53, 45), ('onset', 4530), ('dur', 390), ('piano', 70, 60), ('onset', 4750), ('dur', 240), ('piano', 62, 45), ('onset', 4760), ('dur', 460), '<T>', ('piano', 74, 45), ('onset', 0), ('dur', 240), ('piano', 58, 45), ('onset', 0), ('dur', 250), ('piano', 70, 60), ('onset', 250), ('dur', 240), ('piano', 55, 45), ('onset', 260), ('dur', 230), ('piano', 67, 45), ('onset', 500), ('dur', 470), ('piano', 48, 45), ('onset', 520), ('dur', 450), ('piano', 60, 45), ('onset', 990), ('dur', 960), ('piano', 65, 45), ('onset', 1000), ('dur', 950), ('piano', 57, 30), ('onset', 1020), ('dur', 930), ('piano', 65, 45), ('onset', 2500), ('dur', 240), ('piano', 48, 30), ('onset', 2510), ('dur', 470), ('piano', 57, 30), ('onset', 2520), ('dur', 460), ('piano', 67, 60), ('onset', 2750), ('dur', 240), ('piano', 70, 60), ('onset', 3000), ('dur', 470), ('piano', 53, 45), ('onset', 3010), ('dur', 460), ('piano', 41, 45), ('onset', 3020), ('dur', 450), ('piano', 62, 45), ('onset', 3480), ('dur', 170), ('piano', 46, 30), ('onset', 3490), ('dur', 310), ('piano', 70, 60), ('onset', 3500), ('dur', 240), ('piano', 70, 60), ('onset', 3750), ('dur', 240), ('piano', 36, 45), ('onset', 3760), ('dur', 520), ('piano', 64, 45), ('onset', 3780), ('dur', 160), ('piano', 69, 60), ('onset', 4000), ('dur', 240), ('piano', 70, 60), ('onset', 4250), ('dur', 240), ('piano', 64, 45), ('onset', 4240), ('dur', 190), ('piano', 36, 45), ('onset', 4250), ('dur', 420), ('piano', 69, 60), ('onset', 4500), ('dur', 250), ('piano', 67, 60), ('onset', 4750), ('dur', 240), '<T>', ('piano', 69, 60), ('onset', 0), ('dur', 250), ('piano', 60, 45), ('onset', 10), ('dur', 230), ('piano', 65, 45), ('onset', 10), ('dur', 220), ('piano', 69, 45), ('onset', 500), ('dur', 240), ('piano', 65, 45), ('onset', 510), ('dur', 200), ('piano', 60, 30), ('onset', 520), ('dur', 230), ('piano', 70, 45), ('onset', 750), ('dur', 240), ('piano', 65, 30), ('onset', 760), ('dur', 210), ('piano', 62, 30), ('onset', 760), ('dur', 210), ('piano', 65, 45), ('onset', 990), ('dur', 190), ('piano', 69, 45), ('onset', 1000), ('dur', 240), ('piano', 60, 45), ('onset', 1030), ('dur', 190), ('piano', 70, 45), ('onset', 1250), ('dur', 240), ('piano', 64, 45), ('onset', 1260), ('dur', 200), ('piano', 60, 30), ('onset', 1270), ('dur', 180), ('piano', 69, 45), ('onset', 1500), ('dur', 240), ('piano', 65, 45), ('onset', 1520), ('dur', 210), ('piano', 60, 30), ('onset', 1530), ('dur', 180), ('piano', 67, 60), ('onset', 1750), ('dur', 240), ('piano', 58, 30), ('onset', 1760), ('dur', 240), ('piano', 65, 45), ('onset', 2000), ('dur', 710), ('piano', 48, 30), ('onset', 2030), ('dur', 680), ('piano', 60, 15), ('onset', 2030), ('dur', 210), ('piano', 57, 45), ('onset', 2030), ('dur', 190), ('piano', 41, 45), ('onset', 2740), ('dur', 250), ('piano', 53, 15), ('onset', 2750), ('dur', 240), ('piano', 65, 45), ('onset', 2750), ('dur', 240), ('piano', 67, 60), ('onset', 3000), ('dur', 240), ('piano', 69, 45), ('onset', 3250), ('dur', 240), ('piano', 60, 30), ('onset', 3260), ('dur', 220), ('piano', 41, 30), ('onset', 3260), ('dur', 220), ('piano', 65, 45), ('onset', 3260), ('dur', 220), ('piano', 70, 60), ('onset', 3500), ('dur', 470), ('piano', 60, 30), ('onset', 3520), ('dur', 240), ('piano', 65, 30), ('onset', 3530), ('dur', 300), ('piano', 53, 30), ('onset', 3530), ('dur', 270), ('piano', 70, 45), ('onset', 4000), ('dur', 470), ('piano', 61, 30), ('onset', 4010), ('dur', 470), ('piano', 53, 30), ('onset', 4030), ('dur', 450), ('piano', 70, 45), ('onset', 4500), ('dur', 470), ('piano', 65, 30), ('onset', 4520), ('dur', 450), ('piano', 48, 15), ('onset', 4530), ('dur', 460), '<T>', ('piano', 70, 45), ('onset', 0), ('dur', 470), ('piano', 60, 30), ('onset', 10), ('dur', 470), ('piano', 53, 30), ('onset', 20), ('dur', 470), ('piano', 65, 30), ('onset', 470), ('dur', 450), ('piano', 60, 30), ('onset', 490), ('dur', 440), ('piano', 69, 45), ('onset', 500), ('dur', 470), ('piano', 70, 45), ('onset', 1000), ('dur', 470), ('piano', 60, 30), ('onset', 1010), ('dur', 470), ('piano', 53, 30), ('onset', 1030), ('dur', 440), ('piano', 69, 60), ('onset', 1500), ('dur', 950), ('piano', 60, 30), ('onset', 1520), ('dur', 930), ('piano', 53, 30), ('onset', 1540), ('dur', 910), ('piano', 67, 45), ('onset', 2500), ('dur', 470), ('piano', 58, 30), ('onset', 2520), ('dur', 470), ('piano', 53, 30), ('onset', 2540), ('dur', 440), ('piano', 65, 60), ('onset', 3000), ('dur', 950), ('piano', 46, 30), ('onset', 3030), ('dur', 920), ('piano', 53, 30), ('onset', 3030), ('dur', 920), ('piano', 69, 60), ('onset', 4000), ('dur', 240), ('piano', 70, 60), ('onset', 4250), ('dur', 240), ('piano', 46, 30), ('onset', 4480), ('dur', 500), ('piano', 72, 75), ('onset', 4500), ('dur', 470), ('piano', 58, 30), ('onset', 4500), ('dur', 420), '<T>', ('piano', 72, 60), ('onset', 0), ('dur', 240), ('piano', 74, 60), ('onset', 250), ('dur', 240), ('piano', 72, 45), ('onset', 500), ('dur', 240), ('piano', 70, 45), ('onset', 750), ('dur', 240), ('piano', 69, 45), ('onset', 1000), ('dur', 950), ('piano', 53, 15), ('onset', 1020), ('dur', 930), ('piano', 60, 15), ('onset', 1030), ('dur', 920), ('piano', 72, 45), ('onset', 2000), ('dur', 470), ('piano', 70, 45), ('onset', 2500), ('dur', 470), ('piano', 67, 15), ('onset', 2510), ('dur', 470), ('piano', 53, 30), ('onset', 2520), ('dur', 990), ('piano', 60, 15), ('onset', 2530), ('dur', 510), ('piano', 67, 45), ('onset', 3500), ('dur', 470), ('piano', 65, 30), ('onset', 4000), ('dur', 950), ('piano', 53, 15), ('onset', 4020), ('dur', 1810), ('piano', 53, 15), ('onset', 4030), ('dur', 900), '<T>', ('piano', 72, 45), ('onset', 500), ('dur', 470), ('piano', 77, 45), ('onset', 1000), ('dur', 470), ('piano', 77, 60), ('onset', 1500), ('dur', 240), ('piano', 79, 60), ('onset', 1750), ('dur', 240), ('piano', 81, 60), ('onset', 2000), ('dur', 240), ('piano', 82, 60), ('onset', 2250), ('dur', 240), ('piano', 84, 60), ('onset', 2500), ('dur', 470), ('piano', 81, 45), ('onset', 3000), ('dur', 470), ('piano', 82, 60), ('onset', 4000), ('dur', 470), ('piano', 82, 60), ('onset', 4500), ('dur', 240), ('piano', 82, 60), ('onset', 4750), ('dur', 240), '<T>', ('piano', 82, 60), ('onset', 0), ('dur', 470), ('piano', 82, 60), ('onset', 500), ('dur', 470), ('piano', 82, 60), ('onset', 1000), ('dur', 240), ('piano', 82, 60), ('onset', 1250), ('dur', 240), ('piano', 82, 45), ('onset', 1500), ('dur', 470), ('piano', 72, 60), ('onset', 2000), ('dur', 470), ('piano', 77, 60), ('onset', 2500), ('dur', 470), ('piano', 77, 45), ('onset', 3000), ('dur', 470), ('piano', 79, 75), ('onset', 3500), ('dur', 240), ('piano', 81, 75), ('onset', 3750), ('dur', 240), ('piano', 82, 75), ('onset', 4000), ('dur', 240), ('piano', 84, 75), ('onset', 4250), ('dur', 240), ('piano', 81, 60), ('onset', 4500), ('dur', 710), '<T>', ('piano', 77, 60), ('onset', 250), ('dur', 240), ('piano', 76, 60), ('onset', 500), ('dur', 240), ('piano', 77, 60), ('onset', 750), ('dur', 240), ('piano', 79, 75), ('onset', 1000), ('dur', 470), ('piano', 84, 75), ('onset', 1500), ('dur', 470), ('piano', 82, 75), ('onset', 2000), ('dur', 240), ('piano', 81, 60), ('onset', 2250), ('dur', 240), ('piano', 79, 75), ('onset', 2500), ('dur', 950), ('piano', 81, 60), ('onset', 4000), ('dur', 470), ('piano', 72, 60), ('onset', 4020), ('dur', 450), ('piano', 63, 45), ('onset', 4020), ('dur', 440), ('piano', 58, 45), ('onset', 4020), ('dur', 440), ('piano', 75, 45), ('onset', 4020), ('dur', 440), ('piano', 84, 60), ('onset', 4500), ('dur', 470), ('piano', 72, 60), ('onset', 4520), ('dur', 460), ('piano', 63, 45), ('onset', 4520), ('dur', 460), ('piano', 69, 60), ('onset', 4520), ('dur', 450), ('piano', 79, 60), ('onset', 4520), ('dur', 450), '<T>', ('piano', 84, 75), ('onset', 0), ('dur', 470), ('piano', 77, 60), ('onset', 10), ('dur', 470), ('piano', 65, 45), ('onset', 20), ('dur', 460), ('piano', 62, 60), ('onset', 20), ('dur', 460), ('piano', 74, 75), ('onset', 20), ('dur', 460), ('piano', 82, 75), ('onset', 500), ('dur', 240), ('piano', 70, 60), ('onset', 530), ('dur', 240), ('piano', 79, 75), ('onset', 520), ('dur', 220), ('piano', 62, 45), ('onset', 530), ('dur', 240), ('piano', 81, 75), ('onset', 750), ('dur', 240), ('piano', 72, 60), ('onset', 760), ('dur', 240), ('piano', 65, 60), ('onset', 760), ('dur', 240), ('piano', 77, 75), ('onset', 770), ('dur', 240), ('piano', 79, 90), ('onset', 1000), ('dur', 240), ('piano', 70, 75), ('onset', 1010), ('dur', 230), ('piano', 64, 60), ('onset', 1010), ('dur', 250), ('piano', 76, 75), ('onset', 1020), ('dur', 190), ('piano', 77, 75), ('onset', 1250), ('dur', 240), ('piano', 65, 60), ('onset', 1260), ('dur', 250), ('piano', 69, 75), ('onset', 1270), ('dur', 240), ('piano', 74, 60), ('onset', 1280), ('dur', 220), ('piano', 81, 60), ('onset', 1500), ('dur', 470), ('piano', 72, 60), ('onset', 1510), ('dur', 470), ('piano', 77, 45), ('onset', 1520), ('dur', 450), ('piano', 65, 45), ('onset', 1540), ('dur', 440), ('piano', 72, 45), ('onset', 2000), ('dur', 950), ('piano', 69, 45), ('onset', 2000), ('dur', 840), ('piano', 63, 30), ('onset', 2010), ('dur', 1770), ('piano', 72, 45), ('onset', 3000), ('dur', 240), ('piano', 77, 60), ('onset', 3250), ('dur', 240), ('piano', 77, 60), ('onset', 3500), ('dur', 240), ('piano', 81, 60), ('onset', 3750), ('dur', 240), ('piano', 81, 60), ('onset', 4000), ('dur', 240), ('piano', 84, 60), ('onset', 4250), ('dur', 240), ('piano', 84, 60), ('onset', 4500), ('dur', 240), ('piano', 81, 60), ('onset', 4750), ('dur', 240), '<T>', ('piano', 81, 60), ('onset', 0), ('dur', 470), ('piano', 81, 45), ('onset', 750), ('dur', 240), ('piano', 81, 45), ('onset', 1000), ('dur', 240), ('piano', 84, 60), ('onset', 1250), ('dur', 240), ('piano', 79, 45), ('onset', 1500), ('dur', 240), ('piano', 84, 60), ('onset', 1750), ('dur', 240), ('piano', 76, 60), ('onset', 2000), ('dur', 240), ('piano', 79, 60), ('onset', 2250), ('dur', 240), ('piano', 76, 75), ('onset', 2500), ('dur', 240), ('piano', 77, 75), ('onset', 2750), ('dur', 240), ('piano', 77, 75), ('onset', 3000), ('dur', 240), ('piano', 81, 75), ('onset', 3250), ('dur', 240), ('piano', 84, 75), ('onset', 3500), ('dur', 240), ('piano', 81, 60), ('onset', 3750), ('dur', 240), ('piano', 84, 60), ('onset', 4000), ('dur', 240), ('piano', 81, 60), ('onset', 4250), ('dur', 240), ('piano', 70, 45), ('onset', 4500), ('dur', 290), ('piano', 86, 60), ('onset', 4500), ('dur', 240), ('piano', 65, 30), ('onset', 4510), ('dur', 290), ('piano', 74, 45), ('onset', 4520), ('dur', 220), ('piano', 84, 75), ('onset', 4750), ('dur', 240), ('piano', 70, 60), ('onset', 4760), ('dur', 230), ('piano', 67, 45), ('onset', 4780), ('dur', 190), ('piano', 70, 45), ('onset', 4780), ('dur', 120), '<T>', ('piano', 81, 60), ('onset', 0), ('dur', 950), ('piano', 72, 45), ('onset', 20), ('dur', 940), ('piano', 65, 45), ('onset', 30), ('dur', 1850), ('piano', 70, 45), ('onset', 1500), ('dur', 470), ('piano', 72, 30), ('onset', 1500), ('dur', 470), ('piano', 69, 45), ('onset', 1980), ('dur', 480), ('piano', 77, 30), ('onset', 2000), ('dur', 470), ('piano', 77, 60), ('onset', 2500), ('dur', 240), ('piano', 65, 30), ('onset', 2510), ('dur', 470), ('piano', 70, 30), ('onset', 2510), ('dur', 450), ('piano', 79, 60), ('onset', 2750), ('dur', 240), ('piano', 81, 60), ('onset', 3000), ('dur', 240), ('piano', 82, 75), ('onset', 3250), ('dur', 240), ('piano', 84, 75), ('onset', 3500), ('dur', 470), ('piano', 81, 60), ('onset', 4000), ('dur', 470), '<T>', ('piano', 72, 45), ('onset', 0), ('dur', 470), ('piano', 69, 45), ('onset', 10), ('dur', 470), ('piano', 77, 60), ('onset', 0), ('dur', 470), ('piano', 77, 75), ('onset', 500), ('dur', 240), ('piano', 65, 45), ('onset', 510), ('dur', 450), ('piano', 69, 45), ('onset', 520), ('dur', 440), ('piano', 81, 90), ('onset', 750), ('dur', 240), ('piano', 77, 75), ('onset', 1000), ('dur', 470), ('piano', 70, 60), ('onset', 1020), ('dur', 440), ('piano', 79, 75), ('onset', 1500), ('dur', 470), ('piano', 74, 60), ('onset', 1520), ('dur', 470), ('piano', 70, 45), ('onset', 1520), ('dur', 470), ('piano', 79, 90), ('onset', 2000), ('dur', 240), ('piano', 70, 60), ('onset', 2010), ('dur', 450), ('piano', 82, 90), ('onset', 2250), ('dur', 240), ('piano', 79, 75), ('onset', 2500), ('dur', 470), ('piano', 70, 60), ('onset', 2520), ('dur', 450), ('piano', 72, 60), ('onset', 3000), ('dur', 470), ('piano', 69, 60), ('onset', 3030), ('dur', 450), ('piano', 77, 75), ('onset', 3500), ('dur', 470), ('piano', 70, 60), ('onset', 3510), ('dur', 460), ('piano', 79, 75), ('onset', 4000), ('dur', 240), ('piano', 70, 60), ('onset', 4010), ('dur', 450), ('piano', 81, 75), ('onset', 4250), ('dur', 240), ('piano', 82, 75), ('onset', 4500), ('dur', 240), ('piano', 84, 75), ('onset', 4750), ('dur', 240), '<T>', ('piano', 81, 75), ('onset', 0), ('dur', 470), ('piano', 72, 60), ('onset', 500), ('dur', 240), ('piano', 72, 60), ('onset', 750), ('dur', 240), ('piano', 77, 75)]


flattened_sequence = flatten(generated_sequences, add_special_tokens=True)
print(len(flattened_sequence))
print(flattened_sequence, '\n')

def quantize_to_nearest_tenth(value):
    return round(value / 10) * 10

# Define the arpeggios for the queue
def get_arpeggios(queue, next_onset_time, arpeggiation_type="up"):
    """
    Arpeggiate the queue
    :param queue: The queue
    :param arpeggiation_type: The type of arpeggiation
    :return: The arpeggiated queue
    """
    if len(queue) == 1:
        return queue

    # Get minimum onset time in queue
    min_onset_time = min([note[2] for note in queue])
    onset_diff = next_onset_time - min_onset_time if next_onset_time > min_onset_time else 4990 - min_onset_time
    onset_delta = quantize_to_nearest_tenth(onset_diff / len(queue))
    if onset_delta < 100:
        return queue
    
    if arpeggiation_type == "up":
        queue = sorted(queue, key=lambda x: x[0])
    elif arpeggiation_type == "down":
        queue = sorted(queue, key=lambda x: x[0], reverse=True)
    
    arpeggiated_queue = []
    for i, note in enumerate(queue):
        note = list(note)
        note[2] = min_onset_time + i * onset_delta
        arpeggiated_queue.append(tuple(note))
    
    return arpeggiated_queue

# Define the arpeggiate function
def arpeggiate(flattened_sequence, arpeggiation_type="up"):
    """
    Arpeggiate the sequence
    :param flattened_sequence: The flattened sequence
    :param arpeggiation_type: The type of arpeggiation
    :return: The arpeggiated sequence
    """
    arpeggiated_sequence = []
    queue = []
    onset_in_queue = None
    threshold=30
    for n, note in enumerate(flattened_sequence):
        if isinstance(note, list):
            if len(queue) == 0:
                queue.append(note)
                onset_in_queue = note[2]
                continue
            else:
                # Check if the note's onset is within the threshold
                # If the note's onset is within the threshold, add it to the queue
                # If the note's onset is not within the threshold, arpeggiate the existing queue if there are any notes and append the notes to the arpeggiated sequence and initialize the queue again
                
                if abs(note[2] - onset_in_queue) < threshold:
                    queue.append(note)
                else:
                    queue = get_arpeggios(queue, note[2], "up")
                    # if arpeggiation_type == "up":
                    #     queue = sorted(queue, key=lambda x: x[0])

                    # elif arpeggiation_type == "down":
                    #     queue = sorted(queue, key=lambda x: x[0], reverse=True)
                    for chord_note in queue:
                        arpeggiated_sequence.append(chord_note)
                    queue = []
                    queue.append(note)
                    onset_in_queue = note[2]
        
        # If the last note is reached, arpeggiate the queue
        if len(queue) > 0 and n == len(flattened_sequence) - 1:
            if arpeggiation_type == "up":
                queue = sorted(queue, key=lambda x: x[0])

            elif arpeggiation_type == "down":
                queue = sorted(queue, key=lambda x: x[0], reverse=True)
            for chord_note in queue:
                arpeggiated_sequence.append(chord_note)

    return arpeggiated_sequence

# Reverse the flattened function
def unflatten(sequence):
    unflattened_sequence = []
    for i in range(len(sequence)):
        note_info = ("piano", sequence[i][0], sequence[i][1])
        unflattened_sequence.append(note_info)
        note_info = ("onset", sequence[i][2])
        unflattened_sequence.append(note_info)
        note_info = ("dur", sequence[i][3])
        unflattened_sequence.append(note_info)
        note_info = []
        
        if i < len(sequence)-1:
            if ((sequence[i][2] + sequence[i][3]) >= 5000 and (sequence[i+1][2] + sequence[i+1][3]) < 5000) or (sequence[i+1][2] < sequence[i][2] and sequence[i+1][2] == 0):
                unflattened_sequence.append("<T>")

    return unflattened_sequence

arpeggiated_sequence = arpeggiate(flattened_sequence, arpeggiation_type="up")
unflattened_sequence = unflatten(arpeggiated_sequence)
print(len(unflattened_sequence))
print(unflattened_sequence)


unflattened_sequence = [('prefix', 'instrument', 'piano'), "<S>"] + unflattened_sequence + ["<E>"]

# # Print the generated sequences
# print("Generated sequences:", unflattened_sequence)

# Write the generated sequences to a MIDI file
aria_tokenizer = AbsTokenizer()
mid_dict = aria_tokenizer.detokenize(unflattened_sequence)
mid = mid_dict.to_midi()
mid.save('test_file.mid')