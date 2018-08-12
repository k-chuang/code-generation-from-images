CONTEXT_LENGTH = 48
IMAGE_SIZE = (256, 256, 3,)
BATCH_SIZE = 1
EPOCHS = 50
VOCAB_SIZE = 18
# MAX_LENGTH = 150
IMAGE_FILE_FORMAT = 'channels_last'
VALIDATE = True
# KERNEL_INIT = 'glorot_uniform'
# KERNEL_INIT = 'he_uniform'
KERNEL_INIT = 'he_normal'
# Default kernel_initializer is 'glorot_uniform' or Xavier glorot uniform initialization
# Possibly try 'he_uniform' and'he_normal', seems to be better for ReLU activations
# xavier/glorot initializations seem to better for tanh activations
