import xml.etree.ElementTree as ET
import soundfile as sf
import numpy as np
import glob
from denet import get_denet

TRAINING_FOLDER = "training"
TESTING_FOLDER = "testing"
SOUNDS_SUBFOLDER = "sounds"
TRAINING_SOUNDS_FOLDER = f"{TRAINING_FOLDER}/{SOUNDS_SUBFOLDER}"
TESTING_SOUNDS_FOLDER = f"{TESTING_FOLDER}/{SOUNDS_SUBFOLDER}"

def list_wav_files(folder_path):
    return glob.glob(f"{folder_path}/*.wav")

def list_xml_files(folder_path):
    return glob.glob(f"{folder_path}/*.xml")

def process_wav_file(filename, frame_size, step_size, sample_rate):
    audio_data, sr = sf.read(filename)
    if sr != sample_rate:
        raise ValueError(f"Sample rate mismatch. Expected {sample_rate} got {sr}")
    frames = [audio_data[i:i + frame_size] for i in range(0, len(audio_data), step_size)]
    return np.array(frames)

def main():
    # Model Parameters
    batch_size = 100
    seq_len = 10  # Number of frames in the sequence
    sample_rate = 32000 # samples/sec 
    frame_duration = 0.05 # seconds (50ms)
    frame_size = int(sample_rate * frame_duration) # 1,600


    n_classes = 10

    step_size = frame_size # we move by a whole frame
    
    input_shape = (seq_len, frame_size, 1)
    model = get_denet(input_shape, n_classes, sr=sample_rate, before_pooling=False)
    
    training_wav_files = list_wav_files(TRAINING_SOUNDS_FOLDER)
    # testing_wav_files = list_wav_files(TESTING_SOUNDS_FOLDER)
    
    training_xml_files = list_xml_files(TRAINING_FOLDER)
    # testing_xml_files = list_xml_files(TESTING_FOLDER)
    
    for wav_file in training_wav_files:
        frames = process_wav_file(wav_file, frame_size, step_size, sample_rate)
        frames_reshaped = frames.reshape(-1, seq_len, frame_size, 1)
        y_pred = model.predict(frames_reshaped)
        # Further processing on y_pred
    
    # for wav_file in testing_wav_files:
    #     frames = process_wav_file(wav_file, frame_size, step_size, sample_rate)
    #     frames_reshaped = frames.reshape(-1, seq_len, frame_size, 1)
    #     y_pred = model.predict(frames_reshaped)
    #     # Further processing on y_pred

if __name__ == '__main__':
    main()
