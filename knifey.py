########################################################################
#
# Functions for downloading the Knifey-Spoony data-set from the internet
# and loading it into memory. Note that this only loads the file-names
# for the images in the data-set and does not load the actual images.
#
# Implemented in Python 3.5
#
########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
########################################################################

import os
import hops.hdfs as hdfs
import pydoop.hdfs.path as hpath


########################################################################

# Directory where you want to download and save the data-set.
# Set this before you start calling any of the functions below.
data_dir = hdfs.project_path() + "Resources/knifey-spoony/"

# Directory for the training-set after copying the files using copy_files().
train_dir = data_dir + "/train"

# Directory for the test-set after copying the files using copy_files().
test_dir = data_dir + "/test"

# URL for the data-set on the internet.
#data_url = "https://github.com/Hvass-Labs/knifey-spoony/raw/master/knifey-spoony.tar.gz"

########################################################################
# Various constants for the size of the images.
# Use these constants in your own program.

# Width and height of each image.
img_size = 200

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Shape of the numpy-array for an image.
img_shape = [img_size, img_size, num_channels]

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = 3

########################################################################

if __name__ == '__main__':
    # Download and extract the data-set if it doesn't already exist.
    maybe_download_and_extract()

    hdfs_path = hdfs.project_path() + "Resources/knifey-spoony"
    # Load the data-set.
    dataset = Dataset(hdfs_path)

    # Get the file-paths for the images and their associated class-numbers
    # and class-labels. This is for the training-set.
    image_paths_train, cls_train, labels_train = dataset.get_training_set()

    # Get the file-paths for the images and their associated class-numbers
    # and class-labels. This is for the test-set.
    image_paths_test, cls_test, labels_test = dataset.get_test_set()

    # Check if the training-set looks OK.

    # Print some of the file-paths for the training-set.
    for path in image_paths_train[0:5]:
        print(path)

    # Print the associated class-numbers.
    print(cls_train[0:5])

    # Print the class-numbers as one-hot encoded arrays.
    print(labels_train[0:5])

    # Check if the test-set looks OK.

    # Print some of the file-paths for the test-set.
    for path in image_paths_test[0:5]:
        print(path)

    # Print the associated class-numbers.
    print(cls_test[0:5])

    # Print the class-numbers as one-hot encoded arrays.
    print(labels_test[0:5])

########################################################################
