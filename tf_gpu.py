# import tensorflow as tf

# # List available GPUs
# gpus = tf.config.list_physical_devices('GPU')

# if len(gpus) > 0:
#     print(f'GPUs available: {gpus}')
# else:
#     print('No GPUs available.')

from tensorflow.python.client import device_lib
def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
print(get_available_devices()) 