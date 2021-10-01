"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.svm import SVR
from copy import deepcopy
from pandas.plotting import scatter_matrix
####################
""" Auxiliary functions """
def normalize(X): # max-min normalizing
    N, m = X.shape
    Y = np.zeros([N, m])
    
    for i in range(m):
        mX = min(X[:,i])
        Y[:,i] = (X[:,i] - mX) / (max(X[:,i]) - mX)
    
    return Y

####################
def split_dataset(data, r): # split a dataset in matrix format, using a given ratio for the testing set
	N = len(data)	
	X = []
	Y = []
	
	if r >= 1: 
		print ("Parameter r needs to be smaller than 1!")
		return
	elif r <= 0:
		print ("Parameter r needs to be larger than 0!")
		return

	n = int(round(N*r)) # number of elements in testing sample
	nt = N - n # number of elements in training sample
	ind = -np.ones(n,int) # indexes for testing sample
	R = np.random.randint(N) # some random index from the whole dataset
	
	for i in range(n):
		while R in ind: R = np.random.randint(N) # ensure that the random index hasn't been used before
		ind[i] = R

	ind_ = list(set(range(N)).difference(ind)) # remaining indexes	
	X = data[ind_,:-1] # training features
	XX = data[ind,:-1] # testing features
	Y = data[ind_,-1] # training targets
	YY = data[ind,-1] # testing targests
	return X, XX, Y, YY
#####################
	
""" Setting up the data for classification problem """
r = 0.2 # ratio of test data over all data (this can be changed to any number between 0.0 and 1.0 (not inclusive)
dataset = np.genfromtxt ('https://library.startlearninglabs.uw.edu/DATASCI400/Datasets/iris.csv', delimiter=",")
all_inputs = normalize(dataset[:,:4]) # inputs (features)
normalized_data = deepcopy(dataset)
normalized_data[:,:4] = all_inputs
X, XX, Y, YY = split_dataset(normalized_data, r)


""" CLASSIFICATION MODELS """
# Logistic regression classifier
print ('\n\n\nLogistic regression classifier\n')
C_parameter = 50. / len(X) # parameter for regularization of the model
class_parameter = 'ovr' # parameter for dealing with multiple classes
penalty_parameter = 'l1' # parameter for the optimizer (solver) in the function
solver_parameter = 'saga' # optimization system used
tolerance_parameter = 0.1 # termination parameter
#####################

#Training the Model
clf = LogisticRegression(C=C_parameter, multi_class=class_parameter, penalty=penalty_parameter, solver=solver_parameter, tol=tolerance_parameter)
clf.fit(X, Y) 
print ('coefficients:')
print (clf.coef_) # each row of this matrix corresponds to each one of the classes of the dataset
print ('intercept:')
print (clf.intercept_) # each element of this vector corresponds to each one of the classes of the dataset

# Apply the Model
print ('predictions for test set:')
print (clf.predict(XX))
print ('actual class values:')
print (YY)
#####################

# Naive Bayes classifier
print ('\n\nNaive Bayes classifier\n')
nbc = GaussianNB() # default parameters are fine
nbc.fit(X, Y)
print ("predictions for test set:")
print (nbc.predict(XX))
print ('actual class values:')
print (YY)
####################

# k Nearest Neighbors classifier
print ('\n\nK nearest neighbors classifier\n')
k = 5 # number of neighbors
distance_metric = 'euclidean'
knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
knn.fit(X, Y)
print ("predictions for test set:")
print (knn.predict(XX))
print ('actual class values:')
print (YY)
###################

# Support vector machine classifier
t = 0.001 # tolerance parameter
kp = 'rbf' # kernel parameter
print ('\n\nSupport Vector Machine classifier\n')
clf = SVC(kernel=kp, tol=t)
clf.fit(X, Y)
print ("predictions for test set:")
print (clf.predict(XX))
print ('actual class values:')
print (YY)
####################

# Decision Tree classifier
print ('\n\nDecision Tree classifier\n')
clf = DecisionTreeClassifier() # default parameters are fine
clf.fit(X, Y)
print ("predictions for test set:")
print (clf.predict(XX))
print ('actual class values:')
print (YY)
####################

# Random Forest classifier
estimators = 10 # number of trees parameter
mss = 2 # mininum samples split parameter
print ('\n\nRandom Forest classifier\n')
clf = RandomForestClassifier(n_estimators=estimators, min_samples_split=mss) # default parameters are fine
clf.fit(X, Y)
print ("predictions for test set:")
print (clf.predict(XX))
print ('actual class values:')
print (YY)
####################

""" REGRESSION MODELS """
# Setting up the data for regression problem
r = 0.1 # ratio of test data over all data (this can be changed to any number between 0.0 and 1.0 (not inclusive)
dataset = np.genfromtxt ('https://library.startlearninglabs.uw.edu/DATASCI400/Datasets/OnlineNewsPopularity.csv', delimiter=",")
dataset = dataset[1:,39:] # get rid of headers and first 38 columns
all_inputs = normalize(dataset[:,:-1]) # inputs (features)
normalized_data = deepcopy(dataset)
normalized_data[:,:-1] = all_inputs
X, XX, Y, YY = split_dataset(normalized_data, r)
#####################

# Linear regression 
print ("\n\n\nBasic Linear Regression\n")
regr = LinearRegression()
regr.fit(X, Y)
print ("Coefficients:")
print (regr.coef_)
print ("Intercept:")
print (regr.intercept_)
print ("predictions for test set:")
print (regr.predict(XX))
print ('actual target values:')
print (YY)
#####################

# Polynomial regression 
print ("\n\nPolynomial Regression\n")
pd = 2 # maximum degree for polynomials of original features to be used
PF = PolynomialFeatures(degree=pd) # polynomial features molds
PX = PF.fit_transform(X) # actual polynomial features (training)
PXX = PF.fit_transform(XX) # actual polynomial features (testing)
regr = LinearRegression()
regr.fit(PX, Y)
print ("Coefficients:")
print (regr.coef_)
print ("Intercept:")
print (regr.intercept_)
print ("predictions for test set:")
print (regr.predict(PXX))
print ('actual target values:')
print (YY)
#######################

# Ridge regression
print ("\n\nRidge Regression\n")
a = 0.5 # alpha parameter for regularization
t = 0.001 # tolerance parameter
regr = Ridge(alpha=a, tol=t)
regr.fit(X, Y)
print ("Coefficients:")
print (regr.coef_)
print ("Intercept:")
print (regr.intercept_)
print ("predictions for test set:")
print (regr.predict(XX))
print ('actual target values:')
print (YY)
########################

# Lasso regression
print ("\n\nLasso Regression\n")
a = 0.5 # alpha parameter for regularization
t = 0.0001 # tolerance parameter
regr = Lasso(alpha=a, tol=t)
regr.fit(X, Y)
print ("Coefficients:")
print (regr.coef_)
print ("Intercept:")
print (regr.intercept_)
print ("predictions for test set:")
print (regr.predict(XX))
print ('actual target values:')
print (YY)
########################

# kNN regression
print ("\n\nK Nearest Neighbors Regression\n")
k = 10 # number of neighbors to be used
distance_metric = 'euclidean'
regr = KNeighborsRegressor(n_neighbors=k, metric=distance_metric)
regr.fit(X, Y)
print ("predictions for test set:")
print (regr.predict(XX))
print ('actual target values:')
print (YY)
#########################

# SVM regression
print ("\n\nSupport Vector Machine Regression\n")
t = 0.001 # tolerance parameter
kp = 'rbf'
regr = SVR(kernel=kp, tol=t)
regr.fit(X, Y)
print ("predictions for test set:")
print (regr.predict(XX))
print ('actual target values:')
print (YY)

#################