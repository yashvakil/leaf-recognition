import numpy as np
import cv2
from sklearn.externals import joblib
import os.path
from DataPreProcessing import obtainXY,getAllFeatures,getTreeData
from TestingClassifiers import makeClassifiers

def imageX(image_path):
    meanX, stdX = np.loadtxt('normalValues.txt')
    length, width, area, perimeter, aspect_ratio, form_factor, rectangularity, hu, hist = getAllFeatures(image_path)
    x = np.array([])
    # x = np.append(x, length)
    # x = np.append(x, width)
    x = np.append(x, area)
    x = np.append(x, perimeter)
    x = np.append(x, aspect_ratio)
    x = np.append(x, form_factor)
    x = np.append(x, rectangularity)
    x = np.append(x, hu)
    x = np.append(x, hist)
    x = x.reshape(1, x.shape[0])
    x = (x - meanX)/stdX
    return(x)

def Execute(image_path):
    if os.path.exists('./normalValues.txt')==False:
        makeClassifiers()

    clf1 = joblib.load('knnDump.pkl')
    clf2 = joblib.load('svmDump.pkl')
    clf3 = joblib.load('mlpDump.pkl')

    x = imageX(image_path)

    print("Data Read In")

    prediction = np.array([clf1.predict(x)[0], clf2.predict(x)[0], clf3.predict(x)[0]])
    print(prediction)
    counts = np.bincount(prediction)
    finalPre = np.argmax(counts)
    treeJson = getTreeData(str(finalPre))
    print(treeJson)

Execute('./Test Leaves/Acer Palmatum/1318.jpg')
# Execute('./Test Leaves/Cedrus Deodara/2397.jpg')
# Execute('./Test Leaves/Cercis Chinensis/1173.jpg')
# Execute('./Test Leaves/Citrus Reticulata Blanco/3616.jpg')
# Execute('./Test Leaves/Ginkgo Biloba/2483.jpg')
# Execute('./Test Leaves/Liriodendron Chinense/3561.jpg')
# Execute('./Test Leaves/Nerium Oleander/2597.jpg')

cv2.waitKey(0)
cv2.destroyAllWindows()