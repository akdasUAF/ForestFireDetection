import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump, load
import db_create_train

def DBN_import_model(model_filename):
    loaded_pipeline = load(model_filename)
    return loaded_pipeline
    print(f'Model loaded from {model_filename}')

def DBN_evaluate_model(pipeline, x_test, y_test):
    predictions = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy = {accuracy}")

def main():
    image_width_in_pixels = 254
    image_height_in_pixels = 254
    image_size = (image_width_in_pixels, image_height_in_pixels)

    _, _, x_test, y_test = db_create_train.DBN_import_dataset(image_size)
    pipeline = DBN_import_model('./Models/weights/dbn_pipeline_model.joblib')
    DBN_evaluate_model(pipeline, x_test, y_test)
    return 0

if __name__ == '__main__':
    sys.exit(main())