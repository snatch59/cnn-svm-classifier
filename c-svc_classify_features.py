import numpy as np
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pickle
from sklearn.externals import joblib
import time


def plot_confusion_matrix(y_true, y_pred, matrix_title):
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

    plt.show()


def train_svm_classifier(features, labels):
    # prepare training and test datasets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels,
                                                                        test_size=0.2, random_state=42)

    # train and then classify the images with C-Support Vector Classification
    param = [
        {
            "kernel": ["linear"],
            "C": [1, 10, 100, 1000]
        },
        {
            "kernel": ["rbf"],
            "C": [1, 10, 100, 1000],
            "gamma": [1e-2, 1e-3, 1e-4, 1e-5]
        }
    ]

    print('C-Support Vector Classification starting ...')
    start_time = time.time()

    # request probability estimation
    svm_c = SVC(probability=True)

    # 10-fold cross validation, use 4 thread as each fold and each parameter set can be train in parallel
    clf = model_selection.GridSearchCV(svm_c, param, cv=10, n_jobs=4, verbose=3)

    clf.fit(X_train, y_train)

    # let us know the training outcome - so we don't have to do it again!
    print("\nBest parameters set:")
    print(clf.best_params_)

    y_pred = clf.predict(X_test)
    print("%f seconds" % (time.time() - start_time))

    # save the C-SVC training results for future use
    joblib.dump(clf.best_estimator_, 'svc_estimator.pkl')
    joblib.dump(clf, 'svc_clf.pkl')

    # confusion matrix computation and display
    print("CNN-C-SVC Accuracy: {0:0.1f}%".format(accuracy_score(y_test, y_pred) * 100))
    plot_confusion_matrix(y_test, y_pred, "C-SVC Confusion matrix")


def c_svc_classify():
    # Classification

    # features and labels from feature extraction
    loaded_features = pickle.load(open('features', 'rb'))
    loaded_labels = pickle.load(open('labels', 'rb'))

    train_svm_classifier(loaded_features, loaded_labels)


if __name__ == "__main__":
    c_svc_classify()
