import xml.etree.ElementTree as ET
import numpy as np
import glob
from denet import get_denet
import random

from constants import *
from utils import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy

import tensorflow as tf
tf.autograph.set_verbosity(0)


def main():
    # Model Parameters
    batch_size = 25
    seq_len = 10  # Number of frames in the sequence
    sample_rate = 32000 # samples/sec 
    frame_duration = 0.05 # seconds (50ms)
    frame_size = int(sample_rate * frame_duration) # 1,600

    n_classes = 4 
    # 1: background
    # 2: glass
    # 3: gunshots
    # 4: scream

    # Model Initialization
    input_shape = (seq_len, frame_size, 1)
    model = get_denet(input_shape, n_classes, sr=sample_rate, before_pooling=False)

    all_frames = np.load('prepared_frames.npy')
    all_labels = np.load('prepared_labels.npy')

    total_sequences = all_frames.shape[0]

    # Compile model
    loss = 'categorical_crossentropy'  # Suitable for multi-class classification


    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=[Accuracy()])


    # Inside the main loop where batches are processed
    for start_idx in range(0, total_sequences - batch_size + 1, batch_size):
        X_batch = all_frames[start_idx:start_idx + batch_size]
        y_batch = all_labels[start_idx:start_idx + batch_size]
        
        # Train model on this batch
        loss, acc = model.train_on_batch(X_batch, y_batch)
        
        print(f"Processed batch from index {start_idx} to {start_idx + batch_size}. Loss: {loss}, Accuracy: {acc}")



# Run main function
if __name__ == '__main__':
    main()
