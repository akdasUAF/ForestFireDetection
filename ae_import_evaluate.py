# cnn_import_evaluate.py
# created 2023-02-16
# imports and evaluates the forest fire cnn


import sys
import tensorflow as tf
import cv2
import numpy as np
import ae_create_train as aect
import glob as gb
import os


def import_model(model, path):
    model.load_weights(path)
    return model

def evaluate(model, dataset):
    model.evaluate(x=dataset)


def main():
    # Setup
    image_size = (254, 254)
    batch_size = 32
    no_fire_train_ds, no_fire_validation_ds, no_fire_test_ds, fire_train_ds, fire_validation_ds, fire_test_ds = aect.import_classification_datasets(image_size, batch_size)
    image_shape = image_size + (3, )
    model = aect.create_autoencoder_model(image_shape)
    model.build((None, ) + image_shape)
    
    # Import
    optimizer = 'adam'
    loss_function_name = 'ssim'
    loss_function = aect.ssim_loss
    epochs = 10
    model.compile(optimizer=optimizer, loss=loss_function)
    model = import_model(model, f'./Models/weights/forest_fire_ae_{image_size[0]}x{image_size[1]}_{optimizer}_{loss_function_name}_{epochs}.h5')

    # Evaluate
    print("Performance on Non-Fire Training Data:")
    evaluate(model, no_fire_train_ds)
    print("Performance on Non-Fire Validation Data:")
    evaluate(model, no_fire_validation_ds)
    print("Performance on Non-Fire Test Data:")
    evaluate(model, no_fire_test_ds)
    
    print("Performance on Fire Training Data:")
    evaluate(model, fire_train_ds)
    print("Performance on Fire Validation Data:")
    evaluate(model, fire_validation_ds)
    print("Performance on Fire Test Data:")
    evaluate(model, fire_test_ds)
    
    # Test a no_fire and fire image
    fire_database_dir = './Local_Testing/Sample Images/Fire/*.jpg'
    no_fire_database_dir = './Local_Testing/Sample Images/No_Fire/*.jpg'
    fire_database_dir_list = gb.glob(fire_database_dir)
    no_fire_database_dir_list = gb.glob(no_fire_database_dir)
    
    threshold = 66
    
    fire_and_nofire_test_image_lists = [fire_database_dir_list, no_fire_database_dir_list]
    list_names = ["fire", "no_fire"]
    
    for i, list in enumerate(fire_and_nofire_test_image_lists):
        print(f"Starting {list_names[i]} list: \n")
        for image in list:
            # Output dirs
            recon_output_file = os.path.join("./Local_Testing/reconstructed_and_circle/", image[35:-4] + f"_recon_{list_names[i]}.jpg")
            circle_output_file = os.path.join("./Local_Testing/reconstructed_and_circle/", image[35:-4] + f"_recon_circle_{list_names[i]}.jpg")
            # Read Image
            img = cv2.imread(image)
            print(f"{list_names[i]}_image shape: {img.shape}")
            img_normalized = img.astype('float32') / 255.0
            img_normalized = np.expand_dims(img_normalized, axis=0)
            print(f"{list_names[i]}_image_normalized shape: {img_normalized.shape}")
            # Use model to reconstruct and remove batch dimension
            reconstructed_img = model.predict(img_normalized)
            reconstructed_img = reconstructed_img[0]
            print(f"reconstructed_{list_names[i]}_img shape: {reconstructed_img.shape}")
            # Reconstruct and save image
            reconstructed_img_color = np.clip(reconstructed_img * 255.0, 0, 255).astype('uint8')
            reconstructed_img_color = cv2.cvtColor(reconstructed_img_color, cv2.COLOR_BGR2RGB)
            cv2.imwrite(recon_output_file, reconstructed_img_color)
            # Calculate Error
            mse_pix = np.mean(np.square(img - reconstructed_img_color), axis=-1)
            mse = np.mean(np.square(img - reconstructed_img_color))
            print(f"The mean squared error {mse}")
            if mse > threshold:
                print('Anomaly detected in the image!')
                max_mse_pixel = np.unravel_index(np.argmax(mse_pix), mse_pix.shape)
                print('Pixel with highest MSE:', max_mse_pixel)
                circle_image = cv2.circle(img, (max_mse_pixel[0], max_mse_pixel[1]), radius=5, color=(0, 0, 255), thickness=2)
                cv2.imwrite(circle_output_file, circle_image)
            else:
                print('No anomaly detected in the image.')

if __name__ == '__main__':
    sys.exit(main())
