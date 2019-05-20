#import numpy, of course
import numpy as np


#sign function, returns the sign of a certain value
def sign(x):
    if x>=0:
        return 1
    else:
        return -1

#the main function that trains a perceptor with given data
#rate stands for the learning rate
def perceptor(x,labels,rate=1):
    p=list()
    for i in range(0,len(x)):
        p.append([1])
        
    #adjust x, so that each x is given a perception value
    p=np.array(p)
    x=np.hstack((p,x))

    beta=np.zeros(len(x[0]))

    #train the model with gradient descent method
    for k in range(0,len(x)):
        if sign(np.dot(beta,x[k])) != labels[k]:
            if sign(np.dot(beta,x[k]))>=0:
                beta=beta-rate*x[k]
            else:
                beta=beta+rate*x[k]
        else:
            beta=beta

    #we return the beta vector as the trained model
    #hence, for a given new data y, we can calculate its
    #predicted value with sign(np.dot(beta,y))
    return beta


#given a test set, we use the model trained to classify it
def predict(beta,test):
    classification=np.zeros(len(test))

    p=list()
    for i in range(0,len(test)):
        p.append([1])

    p=np.array(p)
    test=np.hstack((p,test))

    for k in range(0,len(test)):
        classification[k]=sign(np.dot(beta,test[k]))

    #return the array with each element stands for its
    #predicted classification
    return classification


#testing data
x=np.array([[2,1],[3,0],[3,2],[-4,-5],[2,3],[3,4],[-7,-6],[1,1.5]])
labels=np.array([1,1,1,1,-1,-1,-1,-1])
test=np.array([[1.5,1.6],[10,1],[0,-1]])
                 
