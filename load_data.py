"""From our images, make data loader into keras compatible format
   for transfer learning on densenet-121"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import glob
import scipy.misc
import re

import cv2
from keras.datasets import cifar10
from keras import backend as K
from keras.utils import np_utils

import matplotlib.pyplot as plt

VAL_SIZE = .1
TEST_SIZE = .05

def get_train_valid_test(train_dir, img_rows, img_cols):
    """
    train_dir: directory where training data is held.
    img_rows, img_cols: resolution of image (e.g. 64, 64)

    Example train_dir format:
    train_dir
       -- water_0
          -- img0.jpg
          -- img1.jpg
       -- forest_0
          -- img27.jpg
          -- img28.jpg
    Returns:
    x_train, x_test, y_train, y_test
    x_train, x_test: uint8 array of RGB image data with shape (num_samples, 3, img_rows, img_cols).
    y_train, y_test: uint8 array of category labels (integers in range 0-9) with shape (num_samples,)
    """

    data_dir = train_dir + '/**/*.jpg'
    file_list = glob.glob(data_dir)

    label_names = glob.glob(train_dir + '/*')
    label_names = list(map(lambda name: (re.match(re.compile(r".*/(.*)"), name)).group(1), label_names))

    labels = list(map(lambda filename: filename.split(os.sep)[1], file_list))
    labels = np.array(labels)

    _, labels = np.unique(labels, return_inverse=True)
    labels = np.array(labels, dtype='uint8')

    imgs = list(map(plt.imread, file_list))

    # Blur all images
    imgs = list(map(lambda img: cv2.GaussianBlur(img, (3,3), 0),imgs))

    # Resize images
    imgs = list(map(lambda x: cv2.resize(x, (img_rows, img_cols)), imgs))


    # Normalize all images the same way the ImageNet pretrained model did
    # (from densenet fine tune github: https://github.com/flyyufelix/cnn_finetune)
    # Switch RGB to BGR order

    imgs = imgs[:, :, :, ::-1]

    imgs = np.array(imgs, dtype='float64')

    # Subtract ImageNet mean pixel
    imgs[:, :, :, 0] -= 103.939
    imgs[:, :, :, 1] -= 116.779
    imgs[:, :, :, 2] -= 123.68

    imgs = np.array(imgs, dtype='uint8')

    # Split first into train + (valid and test), then (valid and test) into valid + test
    X_train, X_valid_test, Y_train, y_valid_test = train_test_split(
        imgs, labels, test_size=(VAL_SIZE+TEST_SIZE), random_state=42, shuffle=True)

    # For each image in train set, get horizontal, vertical, horizontal + vertical flips
    X_train, Y_train = generate_flips(X_train, Y_train)

    X_valid, X_test, Y_valid, Y_test = train_test_split(
        X_valid_test, y_valid_test, test_size=TEST_SIZE/(VAL_SIZE+TEST_SIZE), random_state=42, shuffle=True)


    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test



def generate_flips(imgs, labels):
    """ imgs: numpy array of 3-channel images.
    labels: numpy array of integer labels corresponding to images
    Returns:
      - list of original images, along with horizontal flips, vertical flips,
        horizontal and vertical flips
      - list of integer labels corresponding to returned images, in the order:
       [ all original labels, all horizontal labels, all vertical, all horz + vertical ]
    """
    horz = imgs[:,::-1,:,:]
    vert = imgs[:,:,::-1,:]
    horz_vert = imgs[:,::-1,::-1,:]

    all_imgs = np.concatenate((imgs, horz,vert, horz_vert),axis=0)
    all_labels = np.concatenate((labels, labels, labels, labels),axis=0)

    return np.array(all_imgs), np.array(all_labels)


def load_data(train_dir, img_rows, img_cols, num_classes):

    # Load custom training and validation set
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = get_train_valid_test(train_dir, img_rows, img_cols)

    # Transform targets to keras compatible format
    Y_train = np_utils.to_categorical(Y_train, num_classes)
    Y_valid = np_utils.to_categorical(Y_valid, num_classes)
    Y_test = np_utils.to_categorical(Y_test, num_classes)

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test


def load_test_data(test_dir, img_rows, img_cols):

    # Load custom training and validation sets
    data_dir = test_dir + '/**/*.jpg'
    file_list = glob.glob(data_dir)

    labels = list(map(lambda filename: filename.split(os.sep)[1], file_list))
    labels = np.array(labels)

    _, labels = np.unique(labels, return_inverse=True)
    labels = np.array(labels, dtype='uint8')

    imgs = list(map(plt.imread, file_list))

    # Resize images to have 3 channels
    imgs = list(map(lambda x: np.resize(x, (img_rows, img_cols, 3)), imgs))
    imgs = np.array(imgs, dtype='uint8')

    num_classes = NUM_CLASSES # TODO: hard-coded

    # Transform targets to keras compatible format
    Y_test = np_utils.to_categorical(labels, num_classes)

    X_test, Y_test = shuffle(imgs, Y_test, random_state=42)

    return X_test, Y_test


def plot_class_distribution(labels, label_dict, which_set, plot_filename, class_weight_dict=None):
    """ Generate bar plot of number of images per class.
    labels:             array of integer labels corresponding to training/validation instances
    label_dict:         dictionary mapping integer labels to class name
    which_set:          one of "train", "validation", or "test" describing which set labels came from
    class_weight_dict:  dictionary mapping integer labels to class weight. If not None, generates
                        a second bar plot showing class weight distribution.
    plot_filename:      name for chart to be saved as. If class_weight_dict is not None,
                        'class_weight' is added to the filename before the extension for the
                        class weight plot.
    """

    _, counts = np.unique(labels, return_counts=True)
    plot_dict = {label_dict[i] : counts[i] for i in range(len(counts))}
    print(plot_dict)

    plt.bar(range(len(plot_dict)), plot_dict.values(), align='center')
    plt.xticks(range(len(plot_dict)), plot_dict.keys())
    plt.savefig(plot_filename)

