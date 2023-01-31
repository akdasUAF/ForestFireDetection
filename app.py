from flask import Flask, render_template, request
import tensorflow as tf

from keras.utils import load_img
from keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

app = Flask(__name__)
model = VGG16()

@app.route('/', methods = ['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def predict():
    imageFile = request.files['imageFile']
    image_path = "./images/" + imageFile.filename
    imageFile.save(image_path)

    userImage = load_img(image_path, target_size=(224,224))
    userImage = img_to_array(userImage)
    userImage = userImage.reshape((1, userImage.shape[0], userImage.shape[1], userImage.shape[2]))
    userImage = preprocess_input(userImage)
    yhat = model.predict(userImage)
    label = decode_predictions(yhat)
    label = label[0][0]

    classification = '%s (%.2f%%)' % (label[1], label[2]*100)

    return render_template("index.html", prediction = classification)


if __name__ == '__main__':
    app.run(port = 3000, debug = True)