# cnn_import_evaluate.py
# created 2023-02-16
# imports and evaluates the forest fire cnn


import sys
import tensorflow as tf
import cv2
import numpy as np
import unet_create_train as unetct
import glob as gb
import os


def import_unet_model(model, path):
    model.load_weights(path)
    return model

def evaluate(model, dataset):
    model.evaluate(x=dataset)


def main():
    # Setup
    image_size = (254, 254)
    batch_size = 32
    train_ds, val_ds = unetct.import_segmentation_dataset(image_size, batch_size)
    image_shape = image_size + (3, )
    model = unetct.create_ae_model(image_shape)
    model.build((None, ) + image_shape)
    
    # Import
    optimizer = 'adam'
    loss_function_name = 'ssim'
    loss_function = unetct.ssim_loss
    epochs = 10
    model.compile(optimizer=optimizer, loss=loss_function)
    model = import_unet_model(model, f'./Models/weights/forest_fire_ae_{image_size[0]}x{image_size[1]}_{optimizer}_{loss_function_name}_{epochs}.h5')

    # Prints the model summary
    model.summary()

    ### NEED TO IMPEMENT METRICS

if __name__ == '__main__':
    sys.exit(main())
