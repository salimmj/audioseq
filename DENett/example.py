import numpy as np
from denet import get_denet

# Settings
batch_size = 100

seq_len = 10 # number of frames in the sequence
samples = 400 # frame_size * sample_rate

input_shape = (seq_len, samples, 1)

sample_rate = 16000
n_classes = 10


# Get the model
model = get_denet(input_shape, n_classes, sr=sample_rate, before_pooling=False)

# Print the model 
model.summary()

# Predict random data
X = np.random.rand(batch_size, seq_len, samples, 1)
y = model.predict(X)

print(y.shape)