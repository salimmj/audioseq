import xml.etree.ElementTree as ET
import soundfile as sf
import numpy as np
import glob
from denet import get_denet

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


def main():
    # Model Parameters
    batch_size = 25
    seq_len = 10  # Number of frames in the sequence
    sample_rate = 32000 # samples/sec 
    frame_duration = 0.05 # seconds (50ms)
    frame_size = int(sample_rate * frame_duration) # 1,600

    n_classes = 10

    step_size = frame_size # we move by a whole frame
    
    input_shape = (seq_len, frame_size, 1)
    model = get_denet(input_shape, n_classes, sr=sample_rate, before_pooling=False)
    
    training_wav_files = list_wav_files(TRAINING_SOUNDS_FOLDER)  # Assuming you've defined this function
    training_xml_files = list_xml_files(TRAINING_FOLDER)
    
    for wav_file in training_wav_files:
        frames = process_wav_file(wav_file, frame_size, step_size, sample_rate)

        print(f"Shape of frames before reshape for {wav_file}: {frames.shape}") 
        
        # Reshape to have the sequence dimension
        frames = frames[:len(frames) // seq_len * seq_len].reshape((-1, seq_len, frame_size, 1))
        
        print(f"Shape of frames for {wav_file}: {frames.shape}")  # Print shape for validation

        y_pred = model.predict(frames)

        print(f"y_pred shape: {y_pred.shape}")
        
        print(f"First 5 predictions: {y_pred[:5]}")  # Print a small portion of predictions for validation

        # Further processing on y_pred


    # for wav_file in testing_wav_files:
    #     frames = process_wav_file(wav_file, frame_size, step_size, sample_rate)
    #     frames_reshaped = frames.reshape(-1, seq_len, frame_size, 1)
    #     y_pred = model.predict(frames_reshaped)
    #     # Further processing on y_pred

if __name__ == '__main__':
    main()
