# nn_create_train.py
# Created 2023-02-04
# contains functions for creating, training, and testing a neural network for identifying
# forest fires in images

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import sys
import cnn_import_evaluate as cie


def import_dataset(image_size):
    # train_ds = tf.keras.utils.image_dataset_from_directory(
    #     './Training/', label_mode='binary', class_names=['No_Fire', 'Fire'], 
    #     seed=123, shuffle=True, image_size=image_size, validation_split=0.2, subset='training'
    #     )

    # validation_ds = tf.keras.utils.image_dataset_from_directory(
    #     './Training/', label_mode='binary', class_names=['No_Fire', 'Fire'],
    #     seed=123, shuffle=True, image_size=image_size, validation_split=0.2, subset='validation'
    #     )

    # test_ds = tf.keras.utils.image_dataset_from_directory(
    #     './Test/', label_mode='binary', class_names=['No_Fire', 'Fire'], 
    #     seed=123, shuffle=True, image_size=image_size
    #     )
    

    full_dataset = tf.keras.utils.image_dataset_from_directory(
        './Combined/', label_mode='binary', class_names=['No_Fire', 'Fire'], 
        seed=123, shuffle=True, image_size=image_size)
    
    dataset_size = full_dataset.cardinality().numpy()
    train_size = int(0.8 * dataset_size)
    val_size = int(0.10 * dataset_size)
    test_size = int(0.10 * dataset_size)

    # full_dataset = full_dataset.shuffle(500)
    train_ds = full_dataset.take(train_size)
    test_ds = full_dataset.skip(train_size)
    validation_ds = test_ds.skip(val_size)
    test_ds = test_ds.take(test_size)

    return train_ds, validation_ds, test_ds


def create_model(input_shape):
    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(2, 3, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation='sigmoid'))
    
    model.compile(optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=['accuracy'])
    return model


def train(model, train_ds, validation_ds):
    model.fit(x=train_ds, validation_data=validation_ds, epochs=2)
    return model


def save_model(model, path):
    model.save_weights(path)


def main():

    # setup
    image_size = (254, 254)
    train_ds, validation_ds, test_ds = import_dataset(image_size)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # create
    model = create_model(image_size + (3, ))
    model.summary()

    # train
    model = train(model, train_ds, validation_ds)

    # evaluate
    cie.evaluate(model, test_ds)

    # save
    model.save('forest_fire_cnn.h5')

    return 0


if __name__ == '__main__':
    sys.exit(main())
