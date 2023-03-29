from flask import Flask, render_template, request

import cv2

import tensorflow as tf
import numpy as np
from cnn_create_train import create_cnn_model
from cnn_import_evaluate import import_cnn_model
from ae_create_train import create_ae_model
from ae_import_evaluate import import_ae_model

app = Flask(__name__, template_folder='Templates')
image_size = (254, 254)

def autoEncoderPredict(image):
    threshold = 66
    img_normalized = image.astype('float32') / 255.0
    img_normalized = np.expand_dims(img_normalized, axis=0)
    
    # Use model to reconstruct and remove batch dimension
    reconstructed_img = model2.predict(img_normalized)
    reconstructed_img = reconstructed_img[0]
    
    # Reconstruct and save image
    reconstructed_img_color = np.clip(reconstructed_img * 255.0, 0, 255).astype('uint8')
    reconstructed_img_color = cv2.cvtColor(reconstructed_img_color, cv2.COLOR_BGR2RGB)
    
    # Calculate Error
    mse_pix = np.mean(np.square(image - reconstructed_img_color), axis=-1)
    mse = np.mean(np.square(image - reconstructed_img_color))
    print(f"The mean squared error {mse}")
    
    if mse > threshold:
        print('Anomaly detected in the image!')
        max_mse_pixel = np.unravel_index(np.argmax(mse_pix), mse_pix.shape)
        print('Pixel with highest MSE:', max_mse_pixel)
        circle_image = cv2.circle(image, (max_mse_pixel[0], max_mse_pixel[1]), radius=5, color=(0, 0, 255), thickness=2)
        #cv2.imwrite(circle_output_file, circle_image)
        return 1
    else:
        print('No anomaly detected in the image.')
        return 0
    
def cnnPredict(image):
    image = np.array(image, dtype=np.float32)
    pred = model.predict(np.expand_dims(image, axis=0))[0][0]

    return round(pred)

# Creating and importing CNN
model = create_cnn_model(image_size + (3, ))
model = import_cnn_model(model, f'Models/weights/forest_fire_cnn.h5')

# Creating and importing bad CNN
model2 = create_ae_model(image_size + (3, ))
model2 = import_ae_model(model2, f'Models/weights/forest_fire_ae_254x254_adam_ssim_10.h5')

listOfModels = [{'name': 'CNN 99%', 'model' : model}, {'name': 'Autoencoder', 'model' : model2}]

@app.route('/', methods = ['GET'])
def hello_world():
    return render_template('index.html', listOfModels = listOfModels)

@app.route('/', methods = ['POST'])
def predict():
    # Define a list of class labels
    class_labels = ['No Fire', 'Fire']
    class_label = 0
    imageFile = request.files['imageFile']
    image_path = "static/images/" + imageFile.filename
    imageFile.save(image_path)
    modelToUse = request.form.get('modelOptions')

    # Grab Image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Resize Image
    resizedImage = cv2.resize(img, image_size, cv2.INTER_AREA)

    if modelToUse == "CNN 99%":
        # Predicting with CNN
        class_idx = cnnPredict(resizedImage)
        # Assign Label
        class_label = class_labels[class_idx]
    elif modelToUse == "Autoencoder":
        # Predicting with Autoencoder
        class_idx = autoEncoderPredict(resizedImage)
        # Assign Label
        class_label = class_labels[class_idx]
    
    return render_template("index.html", prediction = class_label, img = image_path, listOfModels = listOfModels, modelToUse = modelToUse)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port = 8000, debug = True)
