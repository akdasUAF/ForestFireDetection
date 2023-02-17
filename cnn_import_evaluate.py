# cnn_import_evaluate.py
# created 2023-02-16
# imports and evaluates the forest fire cnn


import sys
import tensorflow as tf
import cnn_create_train as cct


def import_model(model, path):
    model.load_weights(path)
    return model


def evaluate(model, dataset):
    model.evaluate(x=dataset)


def main():
    # setup
    image_size = (254, 254)
    train_ds, validation_ds, test_ds = cct.import_dataset(image_size)
    model = cct.create_model(image_size + (3, ))

    # import
    model = import_model(model, './forest_fire_cnn')

    # evaluate
    print("Performance on Training Data:")
    evaluate(model, train_ds)
    print("Performance on Validation Data:")
    evaluate(model, validation_ds)
    print("Performance on Test Data:")
    evaluate(model, test_ds)

if __name__ == '__main__':
    sys.exit(main())
