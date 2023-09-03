import xml.etree.ElementTree as ET
import soundfile as sf
import numpy as np
import glob
from denet import get_denet
import random

from constants import *
from utils import *


def process_wav_file(filename, frame_size, step_size, sample_rate):
    audio_data, sr = sf.read(filename)
    if sr != sample_rate:
        raise ValueError(f"Sample rate mismatch. Expected {sample_rate} got {sr}")

    # Padding if necessary
    remainder = len(audio_data) % frame_size
    pad_length = frame_size - remainder
    audio_data = np.pad(audio_data, (0, pad_length), 'constant')

    # Reorganize frames
    frames = []
    for i in range(0, len(audio_data) - frame_size + 1, step_size):
        frames.append(audio_data[i:i + frame_size])

    frames = np.array(frames)
    return np.expand_dims(frames, axis=-1)  # Adding channel dimension



# Function to parse XML
def parse_xml(xml_file):
    events = []
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for item in root.find('events'):
        event_dict = {
            'CLASS_ID': float(item.find('CLASS_ID').text),
            'STARTSECOND': float(item.find('STARTSECOND').text),
            'ENDSECOND': float(item.find('ENDSECOND').text)
        }
        events.append(event_dict)
    return events

# Generate labels
def generate_labels(events, total_frames, frame_duration, n_classes):
    labels = np.zeros((total_frames, n_classes))  
    for event in events:
        start_frame = int(event['STARTSECOND'] // frame_duration)
        end_frame = int(event['ENDSECOND'] // frame_duration)
        labels[start_frame:end_frame, int(event['CLASS_ID'])-1] = 1.0  # One-hot encoding
    return labels

def main():
    # Parameters
    seq_len = 10  # Number of frames in a sequence
    n_classes = 4  # Number of classes
    sample_rate = 32000  # Sampling rate in Hz
    frame_duration = 0.05  # Frame duration in seconds
    frame_size = int(sample_rate * frame_duration)  # Frame size in samples
    step_size = frame_size  # Step size for frame slicing

    # Placeholder for loaded data and labels
    all_frames = []
    all_labels = []

    # print folders
    print(f"Training sounds folder: {TRAINING_SOUNDS_FOLDER}")
    print(f"Training XML folder: {TRAINING_FOLDER}")

    # Load sound and corresponding XML files
    training_wav_files = list_wav_files(TRAINING_SOUNDS_FOLDER)
    training_xml_files = list_xml_files(TRAINING_FOLDER)

    print(f"Number of training sound files: {len(training_wav_files)}")

    # Process each sound file and its corresponding XML
    for wav_file, xml_file in zip(training_wav_files, training_xml_files):
        # Process WAV file
        frames = process_wav_file(wav_file, frame_size, step_size, sample_rate)
        total_frames = frames.shape[0]

        # Process XML file
        events = parse_xml(xml_file)
        frame_labels = generate_labels(events, total_frames, frame_duration, n_classes)

        # Truncate frames and labels to fit into sequences
        valid_len = total_frames // seq_len * seq_len
        frames = frames[:valid_len].reshape((-1, seq_len, frame_size, 1))
        frame_labels = frame_labels[:valid_len].reshape((-1, seq_len, n_classes))  # Adjust to 4 classes

        all_frames.append(frames)
        all_labels.append(frame_labels)

    # Stack and shuffle
    all_frames = np.vstack(all_frames)
    all_labels = np.vstack(all_labels)
    indices = np.arange(all_frames.shape[0])
    random.shuffle(indices)

    # Save prepared data
    np.save('prepared_frames.npy', all_frames[indices])
    np.save('prepared_labels.npy', all_labels[indices])

    print("Data preparation complete. Shuffled and saved.")

if __name__ == '__main__':
    main()