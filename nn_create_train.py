# nn_create_train.py
# Created 2023-02-04
# contains functions for creating, training, and testing a neural network for identifying
# forest fires in images

import tensorflow as tf
import torch
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import sys
import os


def import_dataset(image_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        './Training/', label_mode='binary', class_names=['No_Fire', 'Fire'], 
        seed=123, shuffle=True, image_size=image_size, validation_split=0.2, subset='training'
        )

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        './Training/', label_mode='binary', class_names=['No_Fire', 'Fire'],
        seed=123, shuffle=True, image_size=image_size, validation_split=0.2, subset='validation'
        )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        './Test/', label_mode='binary', class_names=['No_Fire', 'Fire'], 
        seed=123, shuffle=True, image_size=image_size
        )

    return train_ds, validation_ds, test_ds


def reshape(data):
    # Reshaping the array to 4-dims so that it can work with the Keras API
    data = data.reshape(data.shape[0], 28, 28, 1)

    # Making sure that the values are float so that we can get decimal points after division
    data = data.astype('float32')

    # Normalizing the RGB codes by dividing it to the max RGB value.
    data /= 255

    return data


def visualize(data, label, title):
    figure = plt.figure(figsize=(10, 8))
    plt.title(title)
    
    cols, rows = 5, 5
    num_samples = cols * rows
    train_samples = data[:num_samples, :, :]

    print(f'{label} shape:', data.shape)
    print(f'Number of images in {label}', data.shape[0])

    for i in range(0, cols * rows):
        img = train_samples[i]
        figure.add_subplot(rows, cols, i + 1)
        plt.axis("off")
        plt.imshow(tf.squeeze(img), cmap="gray")
    plt.show()


def create_model(input_shape):
    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(2, 3, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))

    model.build()
    model.summary()
    return model


def train(model, train_ds, validation_ds):
    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
    model.fit(x=train_ds, validation_data=validation_ds, epochs=2)
    return model


def evaluate(model, dataset):
    model.evaluate(x=dataset)


def main():
    image_size = (254, 254)

    # get the data
    train_ds, validation_ds, test_ds = import_dataset(image_size)

    # reshape/preprocess
    # x_train = reshape(x_train)
    # x_test = reshape(x_test)

    # visualize the data
    # visualize(x_train, 'x_train', "Sample Training Data")
    # visualize(x_test, 'x_test', "Sample Test Data")

    # check GPU availability
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    model = create_model(image_size + (3, ))

    model = train(model, train_ds, validation_ds)

    evaluate(model, test_ds)

    return 0


if __name__ == '__main__':
    sys.exit(main())
