import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump, load
import db_import_evaluate


def DBN_import_dataset(image_size):
    train_dir = './Training'
    val_dir = './Test'

    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)
    train_dataset = data_gen.flow_from_directory(
        train_dir, class_mode='categorical', classes=['No_Fire', 'Fire'], 
        seed=123, shuffle=True, target_size=image_size, subset='training')
        
    val_dataset = data_gen.flow_from_directory(
            val_dir, class_mode = 'categorical', classes=['No_Fire', 'Fire'], 
            seed=123, shuffle=True, target_size=image_size, subset = 'validation')

    train_image, train_label = next(train_dataset)
    train_image = np.array(train_image)
    train_label = np.array(train_label)
    train_label = np.argmax(train_label, axis=1)

    val_image, val_label = next(val_dataset)
    val_image = np.array(val_image)
    val_label = np.array(val_label)
    val_label = np.argmax(val_label, axis=1)

    train_image = train_image.reshape(-1, 193548) / 255.0
    val_image = val_image.reshape(-1, 193548) / 255.0

    return train_image, train_label, val_image, val_label


def DBN_create_model(batch_size):
    # Define RBMs and logistic regression
    rbm1 = BernoulliRBM(n_components=256, learning_rate=0.001, n_iter=1, batch_size=batch_size, verbose=True)
    rbm2 = BernoulliRBM(n_components=256, learning_rate=0.001, n_iter=1, batch_size=batch_size, verbose=True)
    # can add more unsupervised RBMs layers here

    # Supervised logistic regression model to be layered on top
    logistic = LogisticRegression(solver='lbfgs', max_iter=10000, multi_class='multinomial')

    # Create a pipeline (stacking the layers)
    pipeline = Pipeline(steps=[('rbm1', rbm1), ('rbm2', rbm2), ('logistic', logistic)])

    return pipeline


def DBN_train_model(pipeline, x_train, y_train):
    # Train the model
    print('Start Deep Belief Network training...')
    pipeline.fit(x_train, y_train)
    return pipeline


def DBN_save_model(pipeline, model_filename):
    # Save the model to a file
    dump(pipeline, model_filename)
    print(f'Model saved to {model_filename}')


def main():
    image_width_in_pixels = 254
    image_height_in_pixels = 254
    image_size = (image_width_in_pixels, image_height_in_pixels)

    x_train, y_train, x_test, y_test = DBN_import_dataset(image_size)

    batch_size = 32
    pipeline = DBN_create_model(batch_size)

    pipeline = DBN_train_model(pipeline, x_train, y_train)

    db_import_evaluate.DBN_evaluate_model(pipeline, x_test, y_test)

    DBN_save_model(pipeline, './Models/weights/dbn_pipeline_model.joblib')

if __name__ == '__main__':
    sys.exit(main())
