import keras
from keras.layers import Conv2D, UpSampling2D, Input, MaxPooling2D, BatchNormalization, Reshape, Conv2DTranspose, Dropout
from keras.initializers import RandomNormal

def encode_text(activation_function=None, input_shape=(16, 300, 1, )):
    initial_weights = RandomNormal(mean=0.0, stddev=0.02)
    
    condition_input = Input(shape=input_shape)
    # encoder
    # size 16x300x1
    network = Conv2D(64, (5, 5), activation=activation_function, kernel_initializer=initial_weights, padding='same')(condition_input)
    network = BatchNormalization()(network)
    network = Conv2D(64, (5, 5), activation=activation_function, strides=(2, 2), kernel_initializer=initial_weights, padding='same')(network)
    network = BatchNormalization()(network)
    # size 8x150x64
    network = Conv2D(128, (3, 3), activation=activation_function, kernel_initializer=initial_weights, padding='same')(network)
    network = Reshape((32, 32, 150))(network)
    network = BatchNormalization()(network)
    # size 32x32x75
    network = Conv2D(128, (5, 5), activation=activation_function, strides=(2, 2), kernel_initializer=initial_weights, padding='same')(network)
    network = BatchNormalization()(network)
    # size 16x16x128
    network = Conv2D(256, (5, 5), activation=activation_function, kernel_initializer=initial_weights, padding='same')(network)
    network = BatchNormalization()(network)
    network = Conv2D(512, (3, 3), activation=activation_function, strides=(2, 2), kernel_initializer=initial_weights, padding='same')(network)
    network = BatchNormalization()(network)
    # size 8x8x512

    # decoder
    network = Conv2D(512, (3, 3), activation=activation_function, kernel_initializer=initial_weights, padding='same')(network)
    network = BatchNormalization()(network)
    network = Conv2DTranspose(256, (3, 3), activation=activation_function, strides=(2, 2), kernel_initializer=initial_weights, padding='same')(network)
    network = BatchNormalization()(network)
    # size 16x16x256

    network = Conv2D(128, (5, 5), activation=activation_function, kernel_initializer=initial_weights, padding='same')(network)
    network = BatchNormalization()(network)
    network = Conv2DTranspose(128, (5, 5), activation=activation_function, strides=(2, 2), kernel_initializer=initial_weights, padding='same')(network)
    network = BatchNormalization()(network)
    # size 32x32x128

    network = Conv2D(64, (5, 5), activation=activation_function, kernel_initializer=initial_weights, padding='same')(network)
    network = BatchNormalization()(network)
    network = Conv2D(32, (3, 3), activation=activation_function, kernel_initializer=initial_weights, padding='same')(network)
    network = BatchNormalization()(network)
    network = Conv2D(16, (3, 3), activation=activation_function, kernel_initializer=initial_weights, padding='same')(network)
    #network = BatchNormalization()(network)
    condition_output = Conv2D(3, (3, 3), activation=activation_function, kernel_initializer=initial_weights, padding='same')(network)

    return condition_input, condition_output