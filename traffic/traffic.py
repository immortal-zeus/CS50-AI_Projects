import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images=[]
    labels=[]
    for sub_dir in range(NUM_CATEGORIES):
        path_sub_dir = os.path.join(data_dir,str(sub_dir))
        for files in os.listdir(path_sub_dir):
            path_file=os.path.join(path_sub_dir,files)
            img = cv2.resize(cv2.imread(path_file),(IMG_WIDTH,IMG_HEIGHT))
            images.append(img)
            labels.append(sub_dir)
    return (images,labels)



def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    my_model = tf.keras.models.Sequential()
    my_model.add(tf.keras.layers.Conv2D(32,(3,3),activation=tf.keras.activations.tanh,input_shape=(IMG_WIDTH,IMG_HEIGHT,3)))
    my_model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    my_model.add(tf.keras.layers.Conv2D(32,(2,2),activation=tf.keras.activations.selu))
    my_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    my_model.add(tf.keras.layers.Conv2D(64,(3,3),activation=tf.keras.activations.relu))
    my_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    my_model.add(tf.keras.layers.Flatten())
    my_model.add(tf.keras.layers.Dropout(0.375))
    my_model.add(tf.keras.layers.Dense(NUM_CATEGORIES,activation=tf.keras.activations.softmax))
    my_model.compile(loss=tf.keras.losses.poisson , optimizer=tf.keras.optimizers.Adam(learning_rate=0.0009),metrics=['accuracy'])
    return my_model



if __name__ == "__main__":
    main()
