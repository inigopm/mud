from sklearn.naive_bayes import MultinomialNB
from utils import toNumpyArray

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import sys

# You may add more classifier methods replicating this function
def applyNaiveBayes(X_train, y_train, X_test):
    '''
    Task: Given some features train a Naive Bayes classifier
          and return its predictions over a test set
    Input; X_train -> Train features
           y_train -> Train_labels
           X_test -> Test features 
    Output: y_predict -> Predictions over the test set
    '''
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    clf = MultinomialNB()
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict

def applyKNN(X_train, y_train, X_test, n_neighbors=5):
    '''
    Task: Given some features train a K-Nearest Neighbors classifier
          and return its predictions over a test set
    Input: X_train -> Train features
           y_train -> Train labels
           X_test -> Test features
           n_neighbors -> Number of neighbors to use by default for kNN queries
    Output: y_predict -> Predictions over the test set
    '''
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict

def applyDecisionTree(X_train, y_train, X_test, max_depth=None):
    '''
    Task: Given some features train a Decision Tree classifier
          and return its predictions over a test set
    Input: X_train -> Train features
           y_train -> Train labels
           X_test -> Test features
           max_depth -> The maximum depth of the tree
    Output: y_predict -> Predictions over the test set
    '''
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict

def applySVM(X_train, y_train, X_test, kernel='linear'):
    '''
    Task: Given some features train a Support Vector Machine classifier
          and return its predictions over a test set
    Input: X_train -> Train features
           y_train -> Train labels
           X_test -> Test features
           kernel -> Specifies the kernel type to be used in the algorithm
    Output: y_predict -> Predictions over the test set
    '''
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    clf = SVC(kernel=kernel)
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict
