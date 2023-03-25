from flask import Flask, render_template, request, jsonify

import cv2

import tensorflow as tf
import numpy as np
import io
from cnn_create_train import *
from cnn_import_evaluate import *
from ae_create_train import *
from ae_import_evaluate import *

app = Flask(__name__, template_folder='Templates')
image_size = (254, 254)

def autoEncoderPredict(image):
    threshold = 66
    img_normalized = image.astype('float32') / 255.0
    img_normalized = np.expand_dims(img_normalized, axis=0)
    
    # Use model to reconstruct and remove batch dimension
    reconstructed_img = model.predict(img_normalized)
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
    
model = create_model(image_size + (3, ))
# import
model = import_model(model, f'Models/weights/forest_fire_cnn.h5')

@app.route('/', methods = ['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def predict():
    # Define a list of class labels
    class_labels = ['No Fire', 'Fire']
    imageFile = request.files['imageFile']
    image_path = "static/images/" + imageFile.filename
    imageFile.save(image_path)

    # Grab Image
    img = Image.open(image_path)

    # Get and Print Width and Height
    width = img.width
    height = img.height

    img = img.resize((254, 254))
    img = np.array(img, dtype=np.float32)

    pred = model.predict(np.expand_dims(img, axis=0))[0][0]

    class_idx = round(pred)

    # Map the class index to a label
    class_label = class_labels[class_idx]

    # Print the predicted class label
    print('Predicted class label: {}'.format(class_label))
    print(pred)

    # Returning the main page to the user with variables to use on the front end
    return render_template("index.html", prediction = class_label, width = width, height = height, img = image_path)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port = 8000, debug = True)
