# !pip install -q tensorflow-gpu==2.0.0-beta0

import tensorflow as tf
from tensorflow.keras import layers, regularizers
# from torch_two_sample.statistics_diff import MMDStatistic
import torch

import matplotlib.pyplot as plt
import numpy as np
import random

from IPython import display
from tqdm import tqdm
from shutil import copyfile
import pandas as pd

print(tf.__version__)

# from google.colab import drive
# drive.mount('/drive')


norm_value = 2173


#################################################################################
#          Helper functions
#################################################################################
def test_show(generator, discriminator):
    noise = tf.random.normal([1, 125, 50])
    generated_ecg = generator(noise, training=False)
    # print(generated_ecg.shape)
    plt.plot(generated_ecg[0, 0, :])
    plt.show()

    decision = discriminator(generated_ecg, training=False)
    print(decision)


def generate_and_save_ecg(model, epoch, test_input, save):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 3))
    plt.plot(predictions[0, 0, :] * norm_value)
    # plt.plot(predictions[0, 0, :])

    if save:
        plt.savefig('./ecg_at_epoch_{:04d}.png'.format(epoch))

    plt.show()


def prepare_data(dim):
    #     copyfile(f"/drive/My Drive/Colab Notebooks/data/fix_signals_400.npy", "./fix_signals.npy")

    # data = np.load('./data/fix_signals_400.npy')
    data = pd.read_csv('./normal_train.csv')  # .iloc[:,:187]
    data.iloc[:, 187] = 0
    data = np.array(data)
    data = np.reshape(data, (data.shape[0], 1, data.shape[1]))
    print('Data shape:', data.shape)

    data = data / norm_value  # Normalize
    data = np.array(data, dtype='float32')

    plt.figure(figsize=(4, 3))
    plt.plot(data[random.randint(0, data.shape[0])][0] * norm_value)
    plt.show()

    train_size = int(data.shape[0] * 0.9)
    test_size = data.shape[0] - train_size
    print(train_size, test_size)

    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(data[:train_size]).shuffle(train_size).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices(data[train_size:]).shuffle(test_size).batch(1)

    seed = tf.random.normal(dim)

    return seed, train_dataset, test_dataset


def make_generator_model():
    model = tf.keras.Sequential()

    model.add(layers.Input(shape=(47, 1)))  # Noise shape

    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))

    model.add(layers.Conv1D(filters=128, kernel_size=16, strides=1, padding='same'))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv1D(filters=64, kernel_size=16, strides=1, padding='same'))
    model.add(layers.LeakyReLU())

    model.add(layers.UpSampling1D(2))

    model.add(layers.Conv1D(filters=32, kernel_size=16, strides=1, padding='same'))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv1D(filters=16, kernel_size=16, strides=1, padding='same'))
    model.add(layers.LeakyReLU())

    model.add(layers.UpSampling1D(2))

    model.add(layers.Conv1D(filters=1, kernel_size=16, strides=1, padding='same', activation='tanh'))

    model.add(layers.Permute((2, 1)))

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()

    model.add(layers.Input(shape=(1, 188)))
    # model.add(layers.Input(shape=(1, 187)))
    model.add(layers.Permute((2, 1)))

    model.add(layers.Conv1D(filters=32, kernel_size=16, strides=1, padding='same'))
    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.1))  # COMMENT OUT MAYBE

    model.add(layers.Conv1D(filters=64, kernel_size=16, strides=1, padding='same'))
    model.add(layers.LeakyReLU())

    model.add(layers.MaxPool1D(pool_size=2))

    model.add(layers.Conv1D(filters=128, kernel_size=16, strides=1, padding='same'))
    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.1))  # COMMENT OUT MAYBE

    model.add(layers.Conv1D(filters=256, kernel_size=16, strides=1, padding='same'))
    model.add(layers.LeakyReLU())

    model.add(layers.MaxPool1D(pool_size=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
