# cnn_import_evaluate.py
# created 2023-02-16
# imports and evaluates the forest fire cnn


import sys
import tensorflow as tf
import cv2
import numpy as np
import ae_create_train as aect


def import_model(model, path):
    model.load_weights(path)
    return model

def evaluate(model, dataset):
    loss = model.evaluate(dataset)
    print('Loss: {:.2f}'.format(loss))


def main():
    # setup
    image_size = (254, 254)
    train_ds, validation_ds, test_ds = aect.import_no_fire_dataset(image_size)
    fire_train_ds, fire_val_ds, fire_test_ds = aect.import_fire_dataset(image_size)
    image_shape = image_size + (3, )
    model = aect.create_model(image_shape)
    model.build((None, ) + image_shape)
    
    # import
    optimizer = 'adam'
    loss_function_name = 'ssim'
    loss_function = aect.ssim_loss
    epochs = 1
    model.compile(optimizer=optimizer, loss=loss_function)
    model = import_model(model, f'Models\\weights\\forest_fire_ae_{optimizer}_{loss_function_name}_{epochs}.h5')

    # evaluate
    print("Performance on Non-Fire Training Data:")
    evaluate(model, train_ds)
    print("Performance on Non-Fire Validation Data:")
    evaluate(model, validation_ds)
    print("Performance on Non-Fire Test Data:")
    evaluate(model, test_ds)
    
    print("Performance on Fire Training Data:")
    evaluate(model, fire_train_ds)
    print("Performance on Fire Validation Data:")
    evaluate(model, fire_val_ds)
    print("Performance on Fire Test Data:")
    evaluate(model, fire_test_ds)
    
    '''# test an image
    img = cv2.imread('test_image.jpg')
    img = cv2.resize(img, (254, 254))
    img = np.expand_dims(img, axis=0)
    # Generate a reconstruction of the input image using the autoencoder
    reconstructed_img = model.predict(img)
    # Calculate the mean squared error (MSE) between the original image and the reconstruction
    mse = np.mean(np.square(img - reconstructed_img))
    # Set a threshold for the MSE above which the image is considered to have an anomaly
    threshold = 0.01
    # Check if the MSE is above the threshold
    if mse > threshold:
        print('Anomaly detected in the image!')
    else:
        print('No anomaly detected in the image.')'''

if __name__ == '__main__':
    sys.exit(main())
