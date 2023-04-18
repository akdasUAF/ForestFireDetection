# unet_create_train.py
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

# Tensorflow configuration settings
#tf.config.experimental_run_functions_eagerly(True)
#gpu_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpu_devices[0], True)

def import_segmentation_dataset(image_size, batch_size):
    
    # Create an ImageDataGenerator to load the images
    image_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2)
    mask_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2)
    
    image_dir = 'D:/UAF/CS Capstone/Datasets/Segmentation/Images'
    mask_dir = 'D:/UAF/CS Capstone/Datasets/Segmentation/Masks'
    
    ### Image Datasets ###
    image_train_ds = image_datagen.flow_from_directory(
        image_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=True,
        subset='training')
    
    image_val_ds = image_datagen.flow_from_directory(
        image_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=True,
        subset='validation')
    
    ### Masks Datasets ###
    mask_train_ds = mask_datagen.flow_from_directory(
        mask_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=True,
        subset='training')
    
    mask_val_ds = mask_datagen.flow_from_directory(
        mask_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=True,
        subset='validation')

    train_ds = zip(image_train_ds, mask_train_ds)
    val_ds = zip(image_val_ds, mask_val_ds)

    return train_ds, val_ds


def create_unet_model(input_shape):
    
    # Encoder 
    inputs = tf.keras.layers.Input(shape=input_shape, name="Input")
    
    enc1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', name='Conv1')(inputs)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='MaxPool1')(enc1)
    enc2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', name='Conv2')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='MaxPool2')(enc2)
    enc3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', name='Conv3')(pool2)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='MaxPool3')(enc3)
    
    # Decoder
    dec1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', name='Conv5')(pool3)
    up1 = tf.keras.layers.UpSampling2D(size=(2, 2), name='Up1')(dec1)
    concat1 = tf.keras.layers.concatenate([enc3, up1], axis = 3, name='Concat1')
    dec2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', name='Conv6')(concat1)
    up2 = tf.keras.layers.UpSampling2D(size=(2, 2), name='Up2')(dec2)
    up2_cropped = tf.keras.layers.Cropping2D(cropping=((1, 0), (1, 0)), name='Crop1')(up2)
    concat2 = tf.keras.layers.concatenate([enc2, up2_cropped], axis = 3, name='Concat2')
    dec3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', name='Conv7')(concat2)
    up3 = tf.keras.layers.UpSampling2D(size=(2, 2), name='Up3')(dec3)
    concat3 = tf.keras.layers.concatenate([enc1, up3], axis = 3, name='Concat3')
    
    outputs = tf.keras.layers.Conv2D(filters=3, kernel_size=3, activation='sigmoid', padding='valid', name='Output')(concat3)
    
    # U-Net Model in total
    unet = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='U-Net')
    
    return unet

# Structural Similarity Index Measure loss function
def ssim_loss(y_true, y_pred):
    return 1.0 - tf.reduce_mean(image.ssim(y_true, y_pred, max_val=1.0))


def train(model, train_ds, validation_ds, epochs):
    model.fit(train_ds, validation_data=validation_ds, epochs=epochs)
    return model


def main():

    # Train Setup
    image_size = (254, 254)    #image_size = (3480, 2160) # Change padding on the last layer in the decoder to 'same' when doing 4K images
    batch_size = 32            #batch_size = 4 
    #no_fire_train_ds, no_fire_validation_ds, no_fire_test_ds, fire_train_ds, fire_validation_ds, fire_test_ds = import_classification_datasets(image_size, batch_size)
    train_ds, val_ds = import_segmentation_dataset(image_size, batch_size)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(tf.config.list_physical_devices('GPU'))

    # Create and save architecture as figure
    image_shape = image_size + (3, )
    model = create_unet_model(image_shape)

    # Build model and print the summary
    model.build((None, ) + image_shape)
    plot_dir = f'C:/Users/Hunter/Desktop/Spring 2023/CS Capstone/GitHub/ForestFireDetection/Models/architectures/forest_fire_unet_{image_size[0]}x{image_size[1]}.png'
    plot_model(model, to_file=plot_dir, show_shapes=True)
    model.summary()

    # Define hyperparameters
    optimizer = 'adam'
    loss_function_name = 'ssim'
    loss_function = ssim_loss
    metrics = ['accuracy']
    epochs = 10
    
    # Train
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
    model = train(model, train_ds, val_ds, epochs)

    # Save
    model.save(f'D:/UAF/CS Capstone/Models/weights/forest_fire_ae_{image_size[0]}x{image_size[1]}_{optimizer}_{loss_function_name}_{epochs}.h5')

    # Evaluate
    test_loss = model.evaluate(train_ds)
    print('Test Loss: {:.2f}'.format(test_loss))

    return 0


if __name__ == '__main__':
    sys.exit(main())
