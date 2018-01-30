import tensorflow as tf
from tensorflow.python.platform import gfile
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
import pickle
import collections
import itertools
import time
import os
import re

# what and where
model_dir = 'imagenet'
images_dir = 'caltech_101_images/'

# TensorFlow inception-v3 feature extraction


def create_graph():
    """Create the CNN graph"""
    with gfile.FastGFile(os.path.join(model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def extract_features(list_images):
    """Extract bottleneck features"""
    nb_features = 2048
    features = np.empty((len(list_images), nb_features))
    labels = []

    create_graph()

    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048 float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG encoding of the image.

    with tf.Session() as sess:
        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        for ind, image in enumerate(list_images):
            imlabel = image.split('/')[1]

            # rough indication of progress
            if ind % 100 == 0:
                print('Processing', image, imlabel)
            if not gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)

            image_data = gfile.FastGFile(image, 'rb').read()
            predictions = sess.run(next_to_last_tensor, {'DecodeJpeg/contents:0': image_data})
            features[ind, :] = np.squeeze(predictions)
            labels.append(imlabel)

    return features, labels


# Graphics


def plot_features(feature_labels, t_sne_features):
    """feature plot"""
    plt.figure(figsize=(9, 9), dpi=100)

    uniques = {x: labels.count(x) for x in feature_labels}
    od = collections.OrderedDict(sorted(uniques.items()))

    colors = itertools.cycle(["r", "b", "g", "c", "m", "y",
                              "slategray", "plum", "cornflowerblue",
                              "hotpink", "darkorange", "forestgreen",
                              "tan", "firebrick", "sandybrown"])
    n = 0
    for label in od:
        count = od[label]
        m = n + count
        plt.scatter(t_sne_features[n:m, 0], t_sne_features[n:m, 1], c=next(colors), s=10, edgecolors='none')
        c = (m + n) // 2
        plt.annotate(label, (t_sne_features[c, 0], t_sne_features[c, 1]))
        n = m

    plt.show()


def plot_confusion_matrix(y_true, y_pred, matrix_title):
    """confusion matrix computation and display"""
    plt.figure(figsize=(9, 9), dpi=100)

    # use sklearn confusion matrix
    cm_array = confusion_matrix(y_true, y_pred)
    plt.imshow(cm_array[:-1, :-1], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(matrix_title, fontsize=16)

    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Number of images', rotation=270, labelpad=30, fontsize=12)

    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    xtick_marks = np.arange(len(true_labels))
    ytick_marks = np.arange(len(pred_labels))

    plt.xticks(xtick_marks, true_labels, rotation=90)
    plt.yticks(ytick_marks, pred_labels)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.tight_layout()

    plt.show()


# Classifier performance


def run_classifier(clfr, x_train_data, y_train_data, x_test_data, y_test_data, acc_str, matrix_header_str):
    """run chosen classifier and display results"""
    start_time = time.time()
    clfr.fit(x_train_data, y_train_data)
    y_pred = clfr.predict(x_test_data)
    print("%f seconds" % (time.time() - start_time))

    # confusion matrix computation and display
    print(acc_str.format(accuracy_score(y_test_data, y_pred) * 100))
    plot_confusion_matrix(y_test_data, y_pred, matrix_header_str)


# Read in images and extract features


# get images - labels are from the subdirectory names
if os.path.exists('features'):
    print('Pre-extracted features and labels found. Loading them ...')
    features = pickle.load(open('features', 'rb'))
    labels = pickle.load(open('labels', 'rb'))
else:
    print('No pre-extracted features - extracting features ...')
    # get the images and the labels from the sub-directory names
    dir_list = [x[0] for x in os.walk(images_dir)]
    dir_list = dir_list[1:]
    list_images = []
    for image_sub_dir in dir_list:
        sub_dir_images = [image_sub_dir + '/' + f for f in os.listdir(image_sub_dir) if re.search('jpg|JPG', f)]
        list_images.extend(sub_dir_images)

    # extract features
    features, labels = extract_features(list_images)

    # save, so they can be used without re-running the last step which can be quite long
    pickle.dump(features, open('features', 'wb'))
    pickle.dump(labels, open('labels', 'wb'))
    print('CNN features obtained and saved.')


# Classification


# TSNE defaults:
# n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000,
# n_iter_without_progress=300, min_grad_norm=1e-07, metric=’euclidean’, init=’random’, verbose=0,
# random_state=None, method=’barnes_hut’, angle=0.5

# t-sne feature plot
if os.path.exists('tsne_features.npz'):
    print('t-sne features found. Loading ...')
    tsne_features = np.load('tsne_features.npz')['tsne_features']
else:
    print('No t-sne features found. Obtaining ...')
    tsne_features = TSNE().fit_transform(features)
    np.savez('tsne_features', tsne_features=tsne_features)
    print('t-sne features obtained and saved.')

plot_features(labels, tsne_features)

# prepare training and test datasets
X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.2, random_state=42)


# LinearSVC defaults:
# penalty=’l2’, loss=’squared_hinge’, dual=True, tol=0.0001, C=1.0, multi_class=’ovr’, fit_intercept=True,
# intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000

# classify the images with a Linear Support Vector Machine (SVM)
print('Support Vector Machine starting ...')
clf = LinearSVC()
run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-SVM Accuracy: {0:0.1f}%", "SVM Confusion matrix")


# RandomForestClassifier/ExtraTreesClassifier defaults:
# (n_estimators=10, criterion='gini’, max_depth=None, min_samples_split=2, min_samples_leaf=1,
# min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0,
# min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False,
# class_weight=None)

# classify the images with a Extra Trees Classifier
print('Extra Trees Classifier starting ...')
clf = ExtraTreesClassifier(n_jobs=4,  n_estimators=100, criterion='gini', min_samples_split=10,
                           max_features=50, max_depth=40, min_samples_leaf=4)
run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-ET Accuracy: {0:0.1f}%", "Extra Trees Confusion matrix")

# classify the images with a Random Forest Classifier
print('Random Forest Classifier starting ...')
clf = RandomForestClassifier(n_jobs=4, criterion='entropy', n_estimators=70, min_samples_split=5)
run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-RF Accuracy: {0:0.1f}%", "Random Forest Confusion matrix")


# KNeighborsClassifier defaults:
# n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’, metric_params=None,
# n_jobs=1, **kwargs

# classify the images with a k-Nearest Neighbors Classifier
print('K-Nearest Neighbours Classifier starting ...')
clf = KNeighborsClassifier(n_neighbors=1, n_jobs=4)
run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-KNN Accuracy: {0:0.1f}%",
               "K-Nearest Neighbor Confusion matrix")


# MPLClassifier defaults:
# hidden_layer_sizes=(100, ), activation=’relu’, solver=’adam’, alpha=0.0001, batch_size=’auto’,
# learning_rate=’constant’, learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None,
# tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False,
# validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08

# classify the image with a Multi-layer Perceptron Classifier
print('Multi-layer Perceptron Classifier starting ...')
clf = MLPClassifier()
run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-MLP Accuracy: {0:0.1f}%",
               "Multi-layer Perceptron Confusion matrix")
