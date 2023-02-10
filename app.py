from flask import Flask, render_template, request

# Import Python Imaging Library PIL(Pillow)
from PIL import Image, ImageDraw

from keras.utils import load_img
from keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def predict():
    imageFile = request.files['imageFile']
    image_path = "static/images/" + imageFile.filename
    imageFile.save(image_path)
    # Grab Image
    img = Image.open(image_path)

    # Get and Print Width and Height
    width = img.width
    height = img.height

    # Re-sizing the image to fit into the models input
    userImage = load_img(image_path, target_size=(224,224))
    userImage = img_to_array(userImage)
    userImage = userImage.reshape((1, userImage.shape[0], userImage.shape[1], userImage.shape[2]))
    userImage = preprocess_input(userImage)
    label = label[0][0]

    classification = '%s (%.2f%%)' % (label[1], label[2]*100)

    # Draw the Circle and save as new image
    offset = 100
    position = [(width/2 - 2*offset, height/2 - 2*offset), (width/2 + offset, height/2 + offset)] # [(x0,y0), (x1,y1)]
    draw = ImageDraw.Draw(img)
    draw.arc(position, start=0, end=360, fill="red", width=10)
    img.save("static/imagesToSendBack/circle.jpg")

    # Returning the main page to the user with variables to use on the front end
    return render_template("index.html", prediction = classification, width = width, height = height, img = img)


if __name__ == '__main__':
    app.run(port = 3000, debug = True)