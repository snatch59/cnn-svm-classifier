# cnn-svm-classifier

This example uses a sub set of 48 labelled images from the Caltech image
set (http://www.vision.caltech.edu/Image_Datasets/Caltech101/),
limited to between 40 and 80 images per label. The images are fed to a
TensorFlow implementation of Inception V3 with the classification layer
removed in order to produce a set of labelled feature vectors.

Next dimensionality reduction is carried out on the 2048-d features using
t-distributed stochastic neighbor embedding (t-SNE) to transform them
to a 2-d feature which is easy to visualize. Note that the t-SNE is used
as an informative step. If the same color/label points are mostly
clustered together there is a high chance that we could use the features
to train a classifier with high accuracy.

These labelled features are to present to a Support Vector Machine (SVM)
to train it in order to classify the images. For comparision Random
Forest, Extra Trees, and K-Nearest Neighbor classifies are also trained.

Training and validation time, and the accuracy of each classifier is
displayed.

## Quick Start

1. Unzip the curated image set caltech_101_images.zip. You should then
have a directory called caltech_101_images in the same directory as
inception3_svm_classifier.py

2. The imagenet directory already has classify_image_graph_def.pb. If I've
removed it to save space on my github account, then download it from
http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz,
un-zip it, and place classify_image_graph_def.pb in a directory called 'imagenet'.

3. Run inception3_svm_classifier.py using Python 3. The following packages
are required: tensorflow, sklearn (scikit-learn), numpy, matplotlib.
Run time (from scratch) was 28 minutes on my dual core i7 Skylake laptop.



