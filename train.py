
import numpy as np
from denet import get_denet
from constants import *
from utils import *
import time
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.callbacks import ModelCheckpoint
import keras
import os


def main():
    # Model Parameters
    batch_size = 128//4
    seq_len = 10  # Number of frames in the sequence
    sample_rate = 32000 # samples/sec 
    frame_duration = 0.05 # seconds (50ms)
    frame_size = int(sample_rate * frame_duration) # 1,600

    max_epochs = 30  # Maximum number of epochs
    initial_epoch = 0  # Variable to keep track of initial epoch

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
    # RMSProp optimizer
    lr=1e-4
    optimizer = keras.optimizers.RMSprop(lr=lr)
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']


    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Initialize FileWriter for TensorBoard
    writer = tf.summary.FileWriter(f"logs/{time.strftime('%Y%m%d-%H%M%S')}")

    # Initialize Model Checkpoint
    # Model Checkpoint Initialization
    checkpoint_path = "best_model.h5"
    # checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
    # callbacks = [checkpoint]

    best_loss = float('inf')
    # Load latest checkpoint if exists
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
        print("Loaded model from last checkpoint")
        # Load initial epoch and best loss
        with open("epoch_count.txt", "r") as f:
            content = f.read().strip().replace('%', '')
            initial_epoch, best_loss = map(float, content.split(','))
            initial_epoch = int(initial_epoch)


    for epoch in range(initial_epoch, max_epochs):
        for start_idx in range(0, total_sequences - batch_size + 1, batch_size):
            X_batch = all_frames[start_idx:start_idx + batch_size]
            y_batch = all_labels[start_idx:start_idx + batch_size]

            loss, acc = model.train_on_batch(X_batch, y_batch)
    
            # TensorBoard Logging
            global_step = epoch * (total_sequences // batch_size) + (start_idx // batch_size)
            summary = tf.Summary(value=[tf.Summary.Value(tag="Loss/loss", simple_value=loss),
                                        tf.Summary.Value(tag="Metrics/accuracy", simple_value=acc)])
            writer.add_summary(summary, global_step=global_step)

            print(f"Processed batch from index {start_idx} to {start_idx + batch_size}. Loss: {loss}, Accuracy: {acc}")
        
        if loss < best_loss:
            model.save_weights(checkpoint_path)
            best_loss = loss
            print(f"Epoch {epoch + 1}: loss improved to {best_loss}, saving model to {checkpoint_path}")
            
        # Save the current epoch and best loss
        with open("epoch_count.txt", "w") as f:
            f.write(f"{epoch + 1},{best_loss:.6f}")

    writer.close()

# Run main function
if __name__ == '__main__':
    main()
