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
from Classifiers import dumpClassifiers


global KNN_n, KNN_w, KNN_algo, SVM_C, SVM_gamma

def makeClassifiers():
    class_names = ['(1)', '(2)', '(3)', '(4)', '(5)', '(6)', '(7)']
    x, y = obtainXY()

    # Set the limits for the training and test data
    dsSize = x.shape[0]
    tsSize = int(dsSize * .8)

    # shuffling of the data
    x_sparse = coo_matrix(x)
    count = 0
    while count<10:
        x, x_sparse, y = shuffle(x, x_sparse, y, random_state=0)
        count = count + 1
    x_sparse = x

    # create training set
    x_train = x_sparse[0:tsSize - 1, :]
    y_train = y[0:tsSize - 1]
    # create test set
    x_test = x_sparse[tsSize:dsSize, :]
    y_test = y[tsSize:dsSize]


    meanX = np.mean(x_train, axis=0)
    stdX = np.std(x_train, axis=0)

    count = 0
    for entry in x_train:
        x_train[count] = (entry - meanX) / stdX
        count = count + 1

    count = 0
    for entry in x_test:
        x_test[count] = (entry - meanX) / stdX
        count = count + 1


    KNN(class_names, x_train, x_test, y_train, y_test)
    SVM(class_names, x_train, x_test, y_train, y_test)
    MLP(class_names, x_train, x_test, y_train, y_test)

    dumpClassifiers(KNN_n, KNN_w, KNN_algo, SVM_C, SVM_gamma)

def MLP(class_names,x_train,x_test,y_train,y_test):
    print('--------------------------------------------------MLP----------------------------------------------------')
    clf = MLPClassifier()
    clf = clf.fit(x_train, y_train)
    print("Training completed")

    # checking the classifier accuracy
    prediction = clf.predict(x_test)
    score = accuracy_score(y_test, prediction)
    print("Test Set accuracy score: " + str(score * 100))
    print('---------------------------------------------------------------------------------------------------------')

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, prediction)
    np.set_printoptions(precision=3)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')

def KNN(class_names,x_train,x_test,y_train,y_test):
    print('--------------------------------------------------KNN----------------------------------------------------')
    neighs = [3, 5, 9, 11, 13, 15, 17, 19, 23, 25, 27, 29]
    weighs = {'uniform', 'distance'}
    algos = {'ball_tree', 'kd_tree', 'brute'}
    acc_sel = 0

    # Select best parameters
    for neigh in neighs:
        for weigh in weighs:
            for algo in algos:
                clf = neighbors.KNeighborsClassifier(n_neighbors=neigh, weights=weigh, algorithm=algo)
                clf = clf.fit(x_train, y_train)
                prediction = clf.predict(x_test)
                score = accuracy_score(y_test, prediction) * 100
                # print("Neighbours = " + str(neigh) + ", Weighting : " + weigh + ",Algorthim: " + algo + ", Accuracy = " + str(score))
                if (score > acc_sel):
                    acc_sel = score
                    n_sel = neigh
                    w_sel = weigh
                    a_sel = algo

    print("Best values are: Neighbours = " + str(n_sel) + ", Weighting : " + w_sel + ",Algorthim: " + a_sel + ", Accuracy = " + str(acc_sel))

    global KNN_n, KNN_w, KNN_algo
    KNN_n = n_sel
    KNN_w = w_sel
    KNN_algo = a_sel

    clf = neighbors.KNeighborsClassifier(n_neighbors=n_sel, weights=w_sel, algorithm=a_sel)
    clf = clf.fit(x_train, y_train)
    print("Training completed")

    # checking the classifier accuracy
    prediction = clf.predict(x_test)
    score = accuracy_score(y_test, prediction)
    print("Test Set accuracy score: " + str(score * 100))
    print('---------------------------------------------------------------------------------------------------------')

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, prediction)
    np.set_printoptions(precision=3)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')

def SVM(class_names, x_train, x_test, y_train, y_test):
    print('--------------------------------------------------SVM----------------------------------------------------')
    C_parm = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    G_parm = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    C_sel = 10
    G_sel = 0.3
    Acc_sel = 0

    # Select best C and gamma values
    for C in C_parm:
        for Gam in G_parm:
            clf = svm.SVC (gamma=Gam, C=C)
            clf = clf.fit(x_train, y_train)
            prediction = clf.predict(x_test)
            score = accuracy_score(y_test, prediction) * 100
            # print("C = " + str(C) + ", Gamma = " + str(Gam) + ", Accuracy = " + str(score))
            if (score > Acc_sel):
                Acc_sel = score
                C_sel = C
                G_sel = Gam

    print("Best parameter values are: C= " + str(C_sel) + " Gamma: " + str(G_sel))
    global  SVM_C, SVM_gamma
    SVM_C = C_sel
    SVM_gamma = G_sel

    # training svm classifer with gaussian kernel
    clf = svm.SVC(gamma=G_sel, C=C_sel)
    clf = clf.fit(x_train, y_train)
    print("Training completed")

    # checking the classifier accuracy
    prediction = clf.predict(x_test)
    score = accuracy_score(y_test, prediction)
    print("Test Set accuracy score: " + str(score * 100))
    print('---------------------------------------------------------------------------------------------------------')

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, prediction)
    np.set_printoptions(precision=3)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')