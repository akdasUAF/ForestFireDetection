# nn_create_train.py
# Created 2023-02-04
# contains functions for creating, training, and testing a neural network for identifying
# forest fires in images

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, UpSampling2D, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.image import ssim
import sys
#import ae_import_evaluate as aeie


def import_no_fire_dataset(image_size):
    
    # Create an ImageDataGenerator to load the images
    image_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    test_image_datagen = ImageDataGenerator(rescale=1./255)
    
    # No Fire Datasets #
    no_fire_dataset_train = image_datagen.flow_from_directory(
        'No_Fire_Images\\Training\\',
        target_size=image_size,
        batch_size=32,
        class_mode='input',
        color_mode='rgb',
        subset='training')
    
    no_fire_dataset_val = image_datagen.flow_from_directory(
        'No_Fire_Images\\Training\\',
        target_size=image_size,
        batch_size=32,
        class_mode='input',
        color_mode='rgb',
        subset='validation')
    
    no_fire_dataset_test = test_image_datagen.flow_from_directory(
        'No_Fire_Images\\Test\\',
        target_size=image_size,
        batch_size=32,
        class_mode='input',
        color_mode='rgb')

    train_ds = no_fire_dataset_train
    validation_ds = no_fire_dataset_val
    test_ds = no_fire_dataset_test

    return train_ds, validation_ds, test_ds

def import_fire_dataset(image_size):

    # Create an ImageDataGenerator to load the images
    image_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    test_image_datagen = ImageDataGenerator(rescale=1./255)
    
    ### Fire Datasets ###
    fire_dataset_train = image_datagen.flow_from_directory(
        'Fire_Images\\Training\\',
        target_size=image_size,
        batch_size=32,
        class_mode='input',
        color_mode='rgb',
        subset='training')
    
    fire_dataset_val = image_datagen.flow_from_directory(
        'Fire_Images\\Training\\',
        target_size=image_size,
        batch_size=32,
        class_mode='input',
        color_mode='rgb',
        subset='validation')
    
    fire_dataset_test = test_image_datagen.flow_from_directory(
        'Fire_Images\\Test\\',
        target_size=image_size,
        batch_size=32,
        class_mode='input',
        color_mode='rgb')

    train_ds = fire_dataset_train
    validation_ds = fire_dataset_val
    test_ds = fire_dataset_test

    return train_ds, validation_ds, test_ds

def create_model(input_shape):
    # Define the encoder layers
    encoder = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape, name='Input'),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', name='Conv1'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='MaxPool1'),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding='same', name='Conv2'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='MaxPool2'),
        tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu', padding='same', name='Conv3'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='MaxPool3')
    ])

    # Define the decoder layers
    decoder = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 8)),
        tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D(size=(2, 2)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D(size=(2, 2)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D(size=(2, 2)),
        tf.keras.layers.Conv2D(filters=3, kernel_size=3, activation='sigmoid', padding='valid')
    ])

    '''tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=3, strides = (2, 2), activation='relu', padding='same', name='ConvT1'),
        tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, strides = (2, 2), activation='relu', padding='same', name='ConvT2'),
        tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides = (2, 2), activation='relu', padding='same', name='ConvT3'),
        tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, activation='sigmoid', padding='same', name='Output')
        '''

    # Define the full autoencoder model
    autoencoder = tf.keras.models.Sequential([encoder, decoder], name='Autoencoder')
    return autoencoder


def ssim_loss(y_true, y_pred):
    return 1 - ssim(y_true, y_pred, max_val=1.0)


def train(model, train_ds, validation_ds, epochs):
    model.fit(train_ds, validation_data=validation_ds, epochs=epochs)
    return model


def main():

    # setup
    image_size = (254, 254)
    train_ds, validation_ds, test_ds = import_no_fire_dataset(image_size)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # create and save architecture as png
    image_shape = image_size + (3, )
    model = create_model(image_shape)

    model.build((None, ) + image_shape)
    plot_dir = 'Models\\architectures\\forest_fire_ae.png'
    plot_model(model, to_file=plot_dir, show_shapes=True)

    # define hyperparameters
    optimizer = 'adam'
    loss_function_name = 'ssim'
    loss_function = ssim_loss
    epochs = 2
    
    # train
    model.compile(optimizer=optimizer, loss=loss_function)
    model = train(model, train_ds, validation_ds, epochs)

    # save
    model.save(f'Models\\weights\\forest_fire_ae_{optimizer}_{loss_function_name}_{epochs}.h5')

    # evaluate
    test_loss = model.evaluate(test_ds)
    print('Test Loss: {:.2f}'.format(test_loss))

    return 0


if __name__ == '__main__':
    sys.exit(main())
