from mnist import MNIST
from sklearn import preprocessing
import numpy as np

def loadTrain():
    mndataSet = 'Dataset'
    mndata = MNIST()
    #mndata.gz =True
    mndata = MNIST(mndataSet)
    images,labels = mndata.load_training()
    return(np.array(images),np.array(labels))

def loadTest():
    mndataSet = 'Dataset'
    mndata = MNIST()
    #mndata.gz =True
    mndata = MNIST(mndataSet)
    images_test, labels_test = mndata.load_testing()
    return(np.array(images_test),np.array(labels_test))


def OneHotTransform(input):
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(input)
    oneHotInput= encoder.transform(input).toarray()
    return(oneHotInput)

def getSamples(X,Y):
    samples = []
    n_training = Y.shape[0]
    for i in range(0, Y.shape[0]):
        samples.append([X[i], Y[i]])
    return(samples)
#print(mndata.display(images[1]))

#%%