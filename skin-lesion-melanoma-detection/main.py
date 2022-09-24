#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 22:15:40 2021

@author: mbaye diongue
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import preprocessing
import numpy as np
from numpy.random import seed
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier as rf
#from math import exp, log


def calculate_cost_LogReg(y, y_hat):
    """
    Calculates the cost of the OUTPUT OF JUST ONE pattern from the logistic
    regression classifier (i.e. the result of applying the h function) and
    its real class.
    
    Parameters
        ----------
        y: float
            Real class.
        y_hat: float
            Output of the h function (i.e. the hypothesis of the logistic
             regression classifier.
         ----------
    Returns
        -------
        cost_i: float
            Value of the cost of the estimated output y_hat.
        -------
    """

    if y_hat>=1:
        y_hat=0.999
    elif y_hat<=0:
        y_hat=0.001
    cost_i=-y*np.log(y_hat)-(1-y)*np.log(1-y_hat)

    return cost_i


def fun_sigmoid(theta, x):
    """
    This function calculates the sigmoid function g(z), where z is a linear
    combination of the parameters theta and the feature vector X's components
    
    Parameters
        ----------
        theta: numpy vector
            Parameters of the h function of the logistic regression classifier.
        x: numpy vector
            Vector containing the data of one pattern.
         ----------
    
    Returns
        -------
        g: float
            Result of applying the sigmoid function using the linear combination 
            of theta and X.
        -------
    """

    # ====================== YOUR CODE HERE ======================
    _,n=theta.shape
    #print(x.shape)
    lin_combi=0 # linear combination
    for i in range(n):
        lin_combi=lin_combi+theta[0,i]*x[0,i]
        
    g= 1/(1+np.exp(-lin_combi))
    # ============================================================

    return g


def train_logistic_regression(X_train, Y_train, alpha):
    """
    This function implements the training of a logistic regression classifier
    using the training data (X_train) and its classes (Y_train).

    Parameters
        ----------
        X_train: Numpy array
            Matrix with dimensions (m x n) with the training data, where m is
            the number of training patterns (i.e. elements) and n is the number
            of features (i.e. the length of the feature vector which characterizes
             the object).
        Y_train: Numpy vector
            Vector that contains the classes of the training patterns. Its length is n.

    Returns
        -------
        theta: numpy vector
            Vector with length n (i.e, the same length as the number of features
            on each pattern). It contains the parameters theta of the hypothesis
            function obtained after the training.

    """
    # CONSTANTS
    # =================
    verbose = True
    max_iter = 300 # You can try with a different number of iterations
    # =================

    # Number of training patterns.
    m = np.shape(X_train)[0]

    # Allocate space for the outputs of the hypothesis function for each training pattern
    h_train = np.zeros(shape=(1, m))

    # Allocate spaces for the values of the cost function on each iteration
    J = np.zeros(shape=(1, 1 + max_iter))

    # Initialize the vector to store the parameters of the hypothesis function
    theta = np.zeros(shape=(1, 1 + np.shape(X_train)[1]))

    # -------------
    # CALCULATE THE VALUE OF THE COST FUNCTION FOR THE INITIAL THETAS
    # -------------
    # a. Intermediate result: Get the error for each element to sum it up.
    total_cost = 0
    for i in range(m):

        # Add a 1 (i.e., the value for x0) at the beginning of each pattern
        x_i = np.insert(np.array([X_train[i]]), 0, 1, axis=1)

        # Expected output (i.e. result of the sigmoid function) for i-th pattern
        # ====================== YOUR CODE HERE ======================
        y_hat_i=fun_sigmoid(theta, x_i)
        # ============================================================

        # Calculate the cost for the i-the pattern and add it to the cost of 
        # the last patterns
        # ====================== YOUR CODE HERE ======================
        cost_pattern_i=calculate_cost_LogReg(Y_train[i], y_hat_i)
        total_cost = total_cost + cost_pattern_i   
        # ============================================================
        

    # b. Calculate the total cost
    # ====================== YOUR CODE HERE ======================
    J[0, 0] = total_cost # initial valye of the cost function
    print( "\n Initial cost :", total_cost)
    # ============================================================

    # -------------
    # GRADIENT DESCENT ALGORITHM TO UPDATE THE THETAS
    # -------------
    # Iterative method carried out during a maximum number (max_iter) of iterations
    for num_iter in range(max_iter):

        # ------
        # STEP 1. Calculate the value of the h function with the current theta values
        # FOR EACH SAMPLE OF THE TRAINING SET
        for i in range(m):
            # Add a 1 (i.e., the value for x0) at the beginning of each pattern
            x_i = np.insert(np.array([X_train[i]]), 0, 1, axis=1)

            # Expected output (i.e. result of the sigmoid function) for i-th pattern
            # ====================== YOUR CODE HERE ======================
            h_i=fun_sigmoid(theta, x_i)
            # ============================================================

            # Store h_i for future use
            h_train[0,i] = h_i

        # ------
        # STEP 2. Update the theta values. To do it, follow the update
        # equations that you studied in the theoretical session
        # a. Intermediate result: Calculate the (h_i-y_i)*x for EACH element from the training set
        # ====================== YOUR CODE HERE ======================
        delta= np.zeros(shape=(1, 1 + np.shape(X_train)[1]))
        for i in range(m):
            x_i = np.insert(np.array([X_train[i]]), 0, 1, axis=1)
            delta=delta+( h_train[0,i]-Y_train[i])*x_i
            
        #print( "theta.shape, delta", theta.shape, delta.shape)
        theta=theta - alpha*1/m*delta
        # ============================================================

        # ------
        # STEP 3: Calculate the cost on this iteration and store it on vector J.
        # ====================== YOUR CODE HERE ======================
        total_cost=0
        for i in range(m):
            # Add a 1 (i.e., the value for x0) at the beginning of each pattern
            x_i = np.insert(np.array([X_train[i]]), 0, 1, axis=1)
    
            # Expected output (i.e. result of the sigmoid function) for i-th pattern
            # ====================== YOUR CODE HERE ======================
            y_hat_i=fun_sigmoid(theta, x_i)
            # ============================================================
    
            # Calculate the cost for the i-the pattern and add it to the cost of 
            # the last patterns
            # ====================== YOUR CODE HERE ======================
            cost_pattern_i=calculate_cost_LogReg(Y_train[i], y_hat_i)
            total_cost = total_cost + cost_pattern_i  
        J[0, num_iter+1] = total_cost
        # ============================================================

        
    
        
    # If verbose is True, plot the cost as a function of the iteration number
    if verbose:
        plt.plot(J[0], color='red')
        plt.xlabel('Number of iterations')
        plt.ylabel('Cost J')
        plt.show()

    return theta


def classify_logistic_regression(X_test, theta):
    """
    This function returns the probability for each pattern of the test set to
    belong to the positive class using the logistic regression classifier.

    Parameters
        ----------
        X_test: Numpy array
            Matrix with dimension (m_t x n) with the test data, where m_t
            is the number of test patterns and n is the number of features (i.e.
            the length of the feature vector that define each element).
        theta: numpy vector
            Parameters of the h function of the logistic regression classifier.

    Returns
        -------
        y_hat: numpy vector
            Vector of length m_t with the estimations made for each test
            element by means of the logistic regression classifier. These
            estimations corredspond to the probabilities that these elements belong
            to the positive class.
    """

    num_elem_test = np.shape(X_test)[0]
    y_hat = np.zeros(shape=(1, num_elem_test))

    for i in range(num_elem_test):
        # Add a 1 (value for x0) at the beginning of each pattern
        x_test_i = np.insert(np.array([X_test[i]]), 0, 1, axis=1)
        # ====================== YOUR CODE HERE ======================

        y_hat[0, i] = fun_sigmoid(theta, x_test_i)
        # ============================================================

    return y_hat

# %%
# -------------
# MAIN PROGRAM
# -------------

dir_output ="Output_intensi"
features_path = dir_output + "/features.h5"
labels_path = dir_output + "/labels_high-low.h5"
test_size = 0.3

seed(1234) # Pour la reproductibilité
# -------------
# PRELIMINARY: LOAD DATASET AND PARTITION TRAIN-TEST SETS (NO NEED TO CHANGE ANYTHING)
# -------------

# import features and labels
h5f_data = h5py.File(features_path, 'r')
h5f_label = h5py.File(labels_path, 'r')

features_string = h5f_data['dataset_skin_lesion'] # '
labels_string = h5f_label['dataset_skin_lesion']

X = np.array(features_string)
Y = np.array(labels_string)

h5f_data.close()
h5f_label.close()

#X=[ x[0:14] for x in X]
#Y=Y

# SPLIT DATA INTO TRAINING AND TEST SETS
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=test_size, random_state=None)

# STANDARDIZE DATA
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# print("Mean of the training set: {}".format(X_train.mean(axis=0)))
# print("Std of the training set: {}".format(X_train.std(axis=0)))
# print("Mean of the test set: {}".format(X_test.mean(axis=0)))
# print("Std of the test set: {}".format(X_test.std(axis=0)))


# -------------
# PART 2.1: TRAINING OF THE CLASSIFIER AND CLASSIFICATION OF THE TEST SET
# -------------

# TRAINING
# Choice of the classifier
method=2


if method==0:
    # The function fTrain_LogisticReg implements the logistic regression
    # classifier. Open it and complete the code.
    alpha = 0.75 # learning rate
    theta = train_logistic_regression(X_train, Y_train, alpha)
    # -------------
    # CLASSIFICATION OF THE TEST SET
    # -------------
    Y_test_hat = classify_logistic_regression(X_test, theta)[0]
    classifier=" Logisti regression "

elif method==1:
    #------------------ LogisticRegression with sklearn ----------------
    clf=LogisticRegression()
    clf.fit(X_train, Y_train)
    Y_test_hat=clf.predict_proba(X_test)
    Y_test_hat= Y_test_hat[:,1]
    classifier=" Logisti regression "

    
elif method==2:
    # -------------------SVM------------------------------------ -----------
    clf = svm.SVC(kernel="rbf", random_state=123, gamma=0.09, C=1, probability=True)
    clf.fit(X_train, Y_train)
    Y_test_hat=clf.predict_proba(X_test)
    Y_test_hat= Y_test_hat[:,1]
    classifier=" SVM "

    
elif method==3:
    #----------------------------- Neural Network -------------------------
    #tf.random.set_seed(1234)
    # build a model
    model = Sequential()
    model.add(Dense(10, input_shape=(X_train.shape[1],), activation='relu')) # Add an input shape! (features,)
    #model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary() 
    
    # compile the model
    model.compile(optimizer='Adam', 
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # now we just update our model fit call
    history = model.fit(X_train, Y_train,
                        epochs=200, # you can set this to a big number!
                        batch_size=10,
                        validation_data = (X_test, Y_test),
                        verbose=0)
    
    Y_test_hat=model.predict(X_test) # probabilities
    Y_test_hat=np.array( [ x[0] for x in Y_test_hat])
    classifier=" Neural Network "


elif method==4:
    #------------------ Random Forest --------------------------------
    clf=rf( max_depth=10, random_state=0, criterion='entropy')
    clf.fit(X_train, Y_train)
    Y_test_hat=clf.predict_proba(X_test)
    Y_test_hat=np.array( [ x[1] for x in Y_test_hat])

    classifier=" Random Forest "


# ============================================================


##---------------------CHOSE OF THE THREHOLD AND CLASS ASIGNATION-------------
# Since all classification algoritms give a probabilty Y_test_hat of belonging in
# the classe "0" or "1". Thus we calculate the best threhold to asign class
# by using the ROC curve  [ we could use 0.5 as default threhold by it is not always optimal]

test_fpr, test_tpr, threholds = roc_curve(Y_test, Y_test_hat)
# The best threhold is given by the point of the ROC curve that is nearest of the point of the
# UP-left corner with coodinate (0,1)
best_threhold=threholds[np.argmin((1 - test_tpr) ** 2 + test_fpr ** 2)]

# Assignation of the class: If the probability is higher than or equal 
# to "best_threhold", then assign it to class 1
Y_test_asig= Y_test_hat > best_threhold


# -------------
# PART 2.2: PERFORMANCE OF THE CLASSIFIER: CALCULATION OF THE ACCURACY AND FSCORE
# -------------

# Show confusion matrix
confm = confusion_matrix(Y_test.T, Y_test_asig.T)
print(confm)
disp = ConfusionMatrixDisplay(confusion_matrix=confm)
disp.plot()
plt.show()


# -------------
# ACCURACY AND F-SCORE
# -------------
# ====================== YOUR CODE HERE ======================
accuracy_= ( confm[0,0]+confm[1,1])/np.sum(confm)
print("***************")
print("The accuracy of the"+classifier+"classifier is {}".format( round(accuracy_,3)))
print("***************")

print("")
precision_= confm[0,0]/( confm[0,0]+confm[1,0])
recall_=confm[0,0]/( confm[0,0]+confm[0,1])
f_score=2*precision_*recall_/(precision_+recall_)
print("***************")
print("The F1-score of the"+classifier+"classifier is {}".format( round(f_score, 3)))
print("***************")
# ============================================================


##---------------------ROC curve-------------------------------------
test_fpr, test_tpr, seuils = roc_curve(Y_test, Y_test_hat)
plt.plot(test_fpr, test_tpr, marker='.')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title(" ROC curve")
plt.show()

#--------------------------- AUC --------------------
test_auc = roc_auc_score(Y_test, Y_test_hat)
print(' AUC=%.3f' % (test_auc))