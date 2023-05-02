# nn_create_train.py
# Created 2023-02-04
# contains functions for creating, training, and testing a neural network for identifying
# forest fires in images

import tensorflow as tf
from tensorflow import image
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, UpSampling2D, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys

def import_classification_dataset(image_size, batch_size):
    
    # Create an ImageDataGenerator to load the images
    image_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2)
    test_image_datagen = ImageDataGenerator(rescale=1/255)
    
    no_fire_train_dir   = './No_Fire_Images/Training/'
    no_fire_test_dir    = './No_Fire_Images/Test/'
    fire_train_dir      = './Fire_Images/Training/'
    fire_test_dir       = './Fire_Images/Test/'
    
    ### No Fire Datasets ###
    no_fire_train_ds = image_datagen.flow_from_directory(
        no_fire_train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='input',
        color_mode='rgb',
        subset='training')
    
    no_fire_validation_ds = image_datagen.flow_from_directory(
        no_fire_test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='input',
        color_mode='rgb',
        subset='validation')
    
    no_fire_test_ds = test_image_datagen.flow_from_directory(
        no_fire_test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='input',
        color_mode='rgb')
    
    ### Fire Datasets ###
    fire_train_ds = image_datagen.flow_from_directory(
        fire_train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='input',
        color_mode='rgb',
        subset='training')
    
    fire_validation_ds = image_datagen.flow_from_directory(
        fire_train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='input',
        color_mode='rgb',
        subset='validation')
    
    fire_test_ds = test_image_datagen.flow_from_directory(
        fire_test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='input',
        color_mode='rgb')

    return no_fire_train_ds, no_fire_validation_ds, no_fire_test_ds, fire_train_ds, fire_validation_ds, fire_test_ds


def create_ae_model(input_shape):
    
    # Encoder
    inputs = tf.keras.layers.Input(shape=input_shape, name="Input")
    enc1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', name='Conv1')(inputs)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='MaxPool1')(enc1)
    enc2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding='same', name='Conv2')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='MaxPool2')(enc2)
    enc3 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu', padding='same', name='Conv3')(pool2)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='MaxPool3')(enc3)
    
    # Decoder
    dec1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu', padding='same', name='Conv4')(pool3)
    up1 = tf.keras.layers.UpSampling2D(size=(2, 2), name='Up1')(dec1)
    dec2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding='same', name='Conv5')(up1)
    up2 = tf.keras.layers.UpSampling2D(size=(2, 2), name='Up2')(dec2)
    dec3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', name='Conv6')(up2)
    up3 = tf.keras.layers.UpSampling2D(size=(2, 2), name='Up3')(dec3)
    outputs = tf.keras.layers.Conv2D(filters=3, kernel_size=3, activation='sigmoid', padding='valid', name='Output')(up3)
    
    # In progress to convert above decoder with these layers - more efficient and cleaner
    #tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=3, strides = (2, 2), activation='relu', padding='same', name='ConvT1'),
    #tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, strides = (2, 2), activation='relu', padding='same', name='ConvT2'),
    #tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides = (2, 2), activation='relu', padding='same', name='ConvT3'),
    #tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, activation='sigmoid', padding='same', name='Output')
    
    # Autoencoder model using all layers 
    autoencoder = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='Autoencoder')
    
    return autoencoder


# Structural Similarity Index Measure loss function
def ssim_loss(y_true, y_pred):
    return 1 - image.ssim(y_true, y_pred, max_val=1.0)


def ae_train(model, train_ds, validation_ds, epochs):
    model.fit(train_ds, validation_data=validation_ds, epochs=epochs)
    return model


def main():

    # Train Setup
    image_size = (254, 254)
    batch_size = 32
    no_fire_train_ds, no_fire_validation_ds, no_fire_test_ds, fire_train_ds, fire_validation_ds, fire_test_ds = import_classification_dataset(image_size, batch_size)
    
    # Check for available GPUs
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(tf.config.list_physical_devices('GPU'))

    # Create and save architecture as figure
    image_shape = image_size + (3, )
    model = create_ae_model(image_shape)

    # Build model and print the summary
    model.build((None, ) + image_shape)
    plot_dir = f'C:/Users/Hunter/Desktop/Spring 2023/CS Capstone/GitHub/ForestFireDetection/Models/architectures/forest_fire_ae_{image_size[0]}x{image_size[1]}.png'
    plot_model(model, to_file=plot_dir, show_shapes=True)
    model.summary()

    # Define hyperparameters
    optimizer = 'adam'
    loss_function_name = 'ssim'
    loss_function = ssim_loss
    epochs = 10
    metrics = ['accuracy']
    
    # Train
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
    model = ae_train(model, no_fire_train_ds, no_fire_validation_ds, epochs)

    # Save
    model.save(f'C:/Users/Hunter/Desktop/Spring 2023/CS Capstone/GitHub/ForestFireDetection/Models/weights/forest_fire_ae_{image_size[0]}x{image_size[1]}_{optimizer}_{loss_function_name}_{epochs}.h5')

    # Evaluate
    test_loss = model.evaluate(no_fire_test_ds)
    print('Test Loss: {:.2f}'.format(test_loss[0]))
    print('Test Accuracy: {:.2f}'.format(test_loss[1]))

    return 0


if __name__ == '__main__':
    sys.exit(main())
