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
    #evaluate(model, train_ds)
    print("Performance on Non-Fire Validation Data:")
    #evaluate(model, validation_ds)
    print("Performance on Non-Fire Test Data:")
    #evaluate(model, test_ds)
    
    print("Performance on Fire Training Data:")
    #evaluate(model, fire_train_ds)
    print("Performance on Fire Validation Data:")
    #evaluate(model, fire_val_ds)
    print("Performance on Fire Test Data:")
    #evaluate(model, fire_test_ds)
    
    # test a no_fire and fire image
    fire_img = cv2.imread('resized_frame1.jpg')
    no_fire_img = cv2.imread('lake_resized_lake_frame2.jpg')
    fire_img = cv2.resize(fire_img, image_size)
    no_fire_img = cv2.resize(no_fire_img, image_size)
    fire_img = np.expand_dims(fire_img, axis=0)
    no_fire_img = np.expand_dims(no_fire_img, axis=0)
    
    reconstructed_fire_img = model.predict(fire_img)
    reconstructed_no_fire_img = model.predict(no_fire_img)

    mse1 = np.mean(np.square(fire_img - reconstructed_fire_img))
    print(mse1)
    mse2 = np.mean(np.square(no_fire_img - reconstructed_no_fire_img))
    print(mse2)

    threshold = 0.01
    if mse1 > threshold:
        print('Anomaly detected in the image!')
    else:
        print('No anomaly detected in the image.')

if __name__ == '__main__':
    sys.exit(main())
