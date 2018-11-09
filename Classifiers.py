import numpy as np
from sklearn import svm
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from DataPreProcessing import obtainXY,plot_confusion_matrix
from random import randint

def dumpClassifiers(KNN_n, KNN_w, KNN_algo, SVM_C, SVM_gamma):
    x, y = obtainXY()

    # shuffling of the data
    x_sparse = coo_matrix(x)
    count = 0
    max = randint(1,15)
    while count<max:
        x, x_sparse, y = shuffle(x, x_sparse, y, random_state=0)
        count = count + 1
    x_train = x
    y_train = y


    meanX = np.mean(x_train, axis=0)
    stdX = np.std(x_train, axis=0)

    np.savetxt('normalValues.txt', (meanX, stdX))

    count = 0
    for entry in x_train:
        x_train[count] = (entry - meanX) / stdX
        count = count + 1


    KNN(x_train, y_train, KNN_n, KNN_w, KNN_algo)
    SVM(x_train, y_train, SVM_C, SVM_gamma)
    MLP(x_train, y_train)

def MLP(x_train,y_train):
    clf = MLPClassifier()
    clf = clf.fit(x_train, y_train)
    joblib.dump(clf, 'mlpDump.pkl')

def KNN(x_train,y_train, KNN_n, KNN_w, KNN_algo):
    clf = neighbors.KNeighborsClassifier(n_neighbors=KNN_n, weights=KNN_w, algorithm=KNN_algo)
    clf = clf.fit(x_train, y_train)
    joblib.dump(clf, 'knnDump.pkl')

def SVM(x_train, y_train, SVM_C, SVM_gamma):
    clf = svm.SVC(gamma=SVM_gamma, C=SVM_C)
    clf = clf.fit(x_train, y_train)
    joblib.dump(clf, 'svmDump.pkl')