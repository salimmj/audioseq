from constants import *
from utils import *
import soundfile as sf

def inspect_wav_shapes(wav_files, sample_rate, frame_size, seq_len):
    report = []
    for wav_file in wav_files:
        audio_data, sr = sf.read(wav_file)
        if sr != sample_rate:
            report.append(f"{wav_file}: Sample rate mismatch. Expected {sample_rate} got {sr}")
        else:
            # Calculate padding
            remainder = len(audio_data) % frame_size
            pad_length = frame_size - remainder
            total_samples = len(audio_data) + pad_length
            
            total_seconds = total_samples / sr
            num_chunks = total_samples // frame_size

            # Number of samples in a sequence
            samples_in_sequence = seq_len * frame_size
            
            # Number of sequences
            n_sequences = (total_samples - samples_in_sequence) // frame_size + 1
            
            report.append(f"{wav_file}: Total samples {total_samples}, Total seconds {total_seconds}, Samples in sequence {samples_in_sequence}, " + 
                          f"Number of 50ms chunks {num_chunks}, Number of sequences {n_sequences}")
    return report

def main():
    # Model Parameters
    seq_len = 10
    sample_rate = 32000 # samples/sec 
    frame_duration = 0.05 # seconds (50ms)
    frame_size = int(sample_rate * frame_duration) # 1,600


    
    training_wav_files = list_wav_files(TRAINING_SOUNDS_FOLDER)
    # testing_wav_files = list_wav_files(TESTING_SOUNDS_FOLDER)
    
    # Inspect shapes of WAV files and print report
    training_shape_report = inspect_wav_shapes(training_wav_files, sample_rate, frame_size, seq_len)
    print("Training WAV Files Shape Report:")
    print(f"Frame size: {frame_size}")
    for line in training_shape_report:
        print(line)
        
    # Existing code ...
    
if __name__ == '__main__':
    main()
