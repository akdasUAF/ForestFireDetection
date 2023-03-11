from flask import Flask, render_template, request, jsonify

# Import Python Imaging Library PIL(Pillow)
from PIL import Image, ImageDraw

import tensorflow as tf
import numpy as np
import io
from cnn_create_train import *
from cnn_import_evaluate import *

app = Flask(__name__)
image_size = (254, 254)
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
    return render_template("index.html", prediction = class_label, preds = pred, width = width, height = height)


if __name__ == '__main__':
    app.run(port = 3000, debug = True)
