# cnn_import_evaluate.py
# created 2023-02-16
# imports and evaluates the forest fire cnn


import sys
import tensorflow as tf
import cnn_create_train as cct


def CNN_import_model(model, path):
    model.load_weights(path)
    return model


def CNN_evaluate(model, dataset):
    evaluation_metrics = model.evaluate(x=dataset)
    print(f'Loss: {evaluation_metrics[0]}')
    print(f'Accuracy: {evaluation_metrics[1]}')

def main():
    # setup
    image_size = (254, 254)
    train_ds, validation_ds, test_ds = cct.CNN_import_dataset(image_size)
    model = cct.CNN_create_model(image_size + (3, ))

    # import
    model = CNN_import_model(model, './Models/weights/cnn_m2.h5')

    # evaluate
    print("Performance on Training Data:")
    CNN_evaluate(model, train_ds)
    print("Performance on Validation Data:")
    CNN_evaluate(model, validation_ds)
    print("Performance on Test Data:")
    CNN_evaluate(model, test_ds)

if __name__ == '__main__':
    sys.exit(main())
