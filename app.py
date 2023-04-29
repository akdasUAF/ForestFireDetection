from flask import Flask, render_template, request

import cv2

import tensorflow as tf
import numpy as np
from cnn_create_train import *
from cnn_import_evaluate import *
from ae_create_train import *
from ae_import_evaluate import *
from db_import_evaluate import *
from unet_create_train import *
from unet_import_evaluate import *
import torch

app = Flask(__name__, template_folder='Templates')
image_size = (254, 254)

def segmentationPredict(model, image, image_path):
    ae_threshold = 66
    unet_threshold = 20
    img_normalized = image.astype('float32') / 255.0
    img_normalized = np.expand_dims(img_normalized, axis=0)
    
    # Use model to reconstruct and remove batch dimension
    if model == 'ae':
        reconstructed_img = aeModel.predict(img_normalized)
    elif model == 'unet':
        reconstructed_img = unetModel.predict(img_normalized)
    else:
        print("FATAL ERROR: segmentationPredict model does not exist")
    reconstructed_img = reconstructed_img[0]
    
    # Reconstruct and save image
    reconstructed_img_color = np.clip(reconstructed_img * 255.0, 0, 255).astype('uint8')
    #recon_output_file_path = image_path[:-4] + '_reconstructed.png'
    #cv2.imwrite(recon_output_file_path, reconstructed_img_color)
    
    # Calculate Error
    mse_pix = np.mean(np.square(image - reconstructed_img_color), axis=-1)
    mse = np.mean(np.square(image - reconstructed_img_color))
    print(f"The mean squared error {mse}")
    
    # Checks if there is an anomoly detected, if so, saves an image with the largest anomoly with it boxed
    if mse > ae_threshold or mse > unet_threshold:
        print('Anomaly detected in the image!')
        max_mse_pixel = np.unravel_index(np.argmax(mse_pix), mse_pix.shape)
        print('Pixel with highest MSE:', max_mse_pixel)
        square_buffer_size = 10
        square_image = cv2.rectangle(image, (max_mse_pixel[0]-square_buffer_size, max_mse_pixel[1]+square_buffer_size), (max_mse_pixel[0]+square_buffer_size, max_mse_pixel[1]-square_buffer_size), color=(0, 0, 255), thickness=2)
        square_image_path = image_path[:-4] + '_sq.png'
        cv2.imwrite(square_image_path, square_image)
        return 1, square_image_path
    else:
        print('No anomaly detected in the image.')
        return 0, image_path
    
def cnnPredict(image):
    image = np.array(image, dtype=np.float32)
    pred = cnnModel.predict(np.expand_dims(image, axis=0))[0][0]

    return round(pred)

def dbnPredict(image):
    image = np.array(image, dtype=np.float32)
    image = image.reshape(-1, 193548) / 255.0
    pred = dbModel.predict(image)
    return pred[0]

def yoloPredict(image):
    result = yoloModel(image)
    result.save()
    path = './runs/detect/exp/image0.jpg'
    return int(len(result.xyxy[0]) > 0), path

# Creating and importing CNN
cnnModel = CNN_create_model(image_size + (3, ))
cnnModel = CNN_import_model(cnnModel, f'Models/weights/forest_fire_cnn.h5')

# Creating and importing Autoencoder
aeModel = create_ae_model(image_size + (3, ))
aeModel = import_ae_model(aeModel, f'Models/weights/forest_fire_ae_254x254_adam_ssim_5.h5')

# Creating and importing Deep Belief Network
dbModel = DBN_import_model('Models/weights/dbn_pipeline_model.joblib')

# Creating and importing U-Net
unetModel = create_unet_model(image_size + (3, ))
unetModel = import_unet_model(unetModel, f'Models/weights/forest_fire_unet_254x254_adam_ssim_5.h5')

# Importing yolo
yoloModel = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp9/weights/best.pt', force_reload=True)

listOfModels = [{'name': 'CNN 99%', 'model' : cnnModel}, 
                {'name': 'Autoencoder', 'model' : aeModel}, 
                {'name': 'Deep Belief', 'model' : dbModel}, 
                {'name': 'U-Net', 'model' : unetModel},
                {'name': 'YOLO', 'model' : yoloModel}]

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
        return render_template("index.html", prediction = class_label, img = image_path, listOfModels = listOfModels, modelToUse = modelToUse)
    elif modelToUse == "Autoencoder":
        # Predicting with Autoencoder
        class_idx, square_image_path = segmentationPredict('ae', resizedImage, image_path)
        # Assign Label
        class_label = class_labels[class_idx]
        return render_template("index.html", prediction = class_label, img = image_path, listOfModels = listOfModels, modelToUse = modelToUse)
    elif modelToUse == "Deep Belief":
        # Predicting with CNN
        class_idx = dbnPredict(resizedImage)
        # Assign Label
        class_label = class_labels[class_idx]
        return render_template("index.html", prediction = class_label, img = image_path, listOfModels = listOfModels, modelToUse = modelToUse)
    elif modelToUse == "U-Net":
        # Predicting with Autoencoder
        class_idx, square_image_path = segmentationPredict('unet', resizedImage, image_path)
        # Assign Label
        class_label = class_labels[class_idx]
        return render_template("index.html", prediction = class_label, img = square_image_path, listOfModels = listOfModels, modelToUse = modelToUse)
    elif modelToUse == 'YOLO':
        class_idx, yolo_image_path = yoloPredict(resizedImage)
        class_label = class_labels[class_idx]
        return render_template("index.html", prediction = class_label, img = yolo_image_path, listOfModels = listOfModels, modelToUse = modelToUse)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port = 8001, debug = True)
