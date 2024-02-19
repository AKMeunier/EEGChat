import keras

from keras import optimizers
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Permute, Flatten, Dense, BatchNormalization, Activation, Dropout


def cvep_cnn_nagel2019(eeg_sampling_rate, window_size_samples, nr_eeg_channels, pretrained=False):
    if pretrained:
        assert window_size_samples == 150
        assert eeg_sampling_rate == 600
        pretrained_model_file = "../../data/models/cvep_cnn_nagel2019/VP1.hdf5"
        model = load_model(pretrained_model_file)
        return model

    learning_rate = 0.001  # the learning rate

    # define model
    model = Sequential()
    model.add(Permute((1, 2, 3), input_shape=(window_size_samples, nr_eeg_channels, 1)))
    # layer1
    model.add(Conv2D(16, kernel_size=(1, nr_eeg_channels), padding='valid', strides=(1, 1), data_format='channels_last',
                     activation='relu'))
    model.add(BatchNormalization(axis=1, scale=False, center=False))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    # layer2
    model.add(Conv2D(8, kernel_size=(64, 1), data_format='channels_last', padding='same'))
    model.add(BatchNormalization(axis=1, scale=False, center=False))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Dropout(0.5))
    # layer3
    model.add(Conv2D(4, kernel_size=(5, 5), data_format='channels_last', padding='same'))
    model.add(BatchNormalization(axis=1, scale=False, center=False))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last', padding='same'))
    model.add(Dropout(0.5))
    # layer4
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    # layer5
    model.add(Dense(2, activation='softmax'))

    adam = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


