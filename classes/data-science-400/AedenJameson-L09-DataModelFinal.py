#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print(
'''
Author: Aeden Jameson

Project: Assignment 7

Data Description: Hepatitis Data Set 

  I. Source: https://archive.ics.uci.edu/ml/datasets/hepatitis
    
  
  II. Characteristics:
      
    Number of Attributes: 20
    Number of Observations: 155
    
    
  III. Attributes - Name, Type, Distribution
     class               int64  die: 32 live: 123
     age                 int64  Normal Distribution
     sex                 int64  male: 139, female: 16
     steroid            object  no: 76, yes: 78, ?: 1
     antivirals          int64  no: 24, yes: 131
     fatigue            object  no: 100, yes: 54, ?: 1
     malaise            object  no: 93, yes: 61, ?: 1
     anorexia           object  no: 61, yes, 93, ?: 1
     liver big          object  no: 25, yes: 120, ?: 10
     liver firm         object  no: 60, yes:  84, ?: 11
     spleen palpable    object  no: 30, yes: 120, ?: 5
     spiders            object  no: 51, yes: 99, ?: 5
     ascites            object  no: 20, yes: 130, ?: 5
     varices            object  no: 18, yes: 132, ?: 5
     bilirubin          object  bi-modal distribution skewed right
     alk phosphate      object  skewed right
     sgot               object  bi-modal distribution skewed right
     albumin            object  skewed-right
     protime            object  skewed-right
     histology           int64  no: 84, yes: 70

  III. Missing & Outlier Values

    All of the following variables had both missing values, indicated by '?',
    and outliers,
    
        protime
        sgot  
        
  Values were considered outliers if they fell outside two standard deviations of 
  the mean. All missing values were replaced because it appeared that all columns
  had a sensible median. 
    

  IV. Removed Attributes

    None


  V. Removed Rows

    No rows were removed. After sifting throught the data I didn't see obvious
    candidates, but I'm probably wrong.
'''
)
    
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data'
patients = pd.read_csv(url, header=None)
patients.columns = ["class","age", "sex", "steroid", "antivirals", "fatigue", "malaise", 
"anorexia", "liver big", "liver firm", "spleen palpable", "spiders", "ascites","varices",
"bilirubin", "alk phosphate", "sgot", "albumin", "protime", "histology"]


# Returns the data set with the missing values replaced with the median. 
def replace_missing(data, column, dtype):
    data.loc[:, column] = pd.to_numeric(data.loc[:, column], errors='coerce')
    HasNan = np.isnan(data.loc[:, column])
    data.loc[HasNan, column] = np.nanmedian(data.loc[:,column])
    data.loc[:,column] = data.loc[:,column].astype(int)
    return data

# Returns the data with the outliers replaced with the mean. Values that 
# lie two standard deviations from the mean are replaced.
def replace_outliers(data, column):
    flag_good = gaussian_dist_filter(data, column)
    data.loc[~flag_good, column] = np.mean(data.loc[flag_good, column])

    return data

# replaces missing data and outliers
def cleanse(data, column, dtype): 
  missing_fixed = replace_missing(data, column, dtype)
  return replace_outliers(missing_fixed, column)

# a helper method that returns a boolean array that can be
# to filter values that fall within two standard
# deviations 
def gaussian_dist_filter(data, column):
    hi = np.mean(data.loc[:,column]) + 2*np.std(data.loc[:,column])
    low = np.mean(data.loc[:,column]) - 2*np.std(data.loc[:,column])
    flag_good = (data.loc[:,column] >= low) & (data.loc[:,column] <= hi)
    
    return flag_good



print("Hepatitis Data Set " )
print(f'Column Names: {list(patients.columns)}')
print('\n')
columns = [
    "protime",
    "sgot",
    "age"
]
for c in columns:   
  print(f'*********** Replace Missing: {c} Column *******************')
  print(f'# Values Missing: {len(list(patients.loc[patients.loc[:, c] == "?", c]))}\n')
  print(f'Original Values: {list(patients.loc[:, c])}\n')
  patients = replace_missing(patients, c, int)
  print(f'# Values After: {len(list(patients.loc[patients.loc[:, c] == "?", c]))}\n')
  print(f'After Replace: {list(patients.loc[0:9, c])}')
  print('\n')

for c in columns: 
  flag_good = gaussian_dist_filter(patients, c)
  print(f'*********** Replace Outliers: {c} Column *******************')
  print(f'Original Values: {list(patients.loc[0:9, c])}\n')
  print(f'Outliers: {list(patients.loc[~flag_good, c])}\n')
  patients = replace_outliers(patients, c)
  print(f'After Replace: {list(patients.loc[0:9, c])}\n')
  print('\n')


for c in columns: 
  print(f'*********** Z-Normalize: {c} Column *******************')
  print(f'Values Before: {list(patients.loc[0:9,c])}\n')
  offset = np.mean(list(patients.loc[:, c]))
  spread = np.std(list(patients.loc[:, c]))
  patients[f'z-norm {c}'] = (patients.loc[:, c] - offset)/spread
  print(f'Z-Norm Values: {list(patients.loc[0:9, "z-norm " + c])}')
  print('\n')

print (" ***************  Encode target to 0, 1 for ease of use *************************")
patients.loc[patients.loc[:,"class"] == 1, "class"] = 0
patients.loc[patients.loc[:,"class"] == 2, "class"] = 1
print ("\n")

print(
'''
QUESTION: Do hepatitis patients 40-50 with lower prothrombin times die?

Expert Label: Class

Decision Comments: The column labelled class contains whether the patients
lived or died
'''      
)



print(f'********* Perform K-Means on age vs protime - Not Normalized ************************')
X = patients.filter(["age", "protime"])
kmeans = KMeans(n_clusters=7, random_state=0).fit(X.values)
plt.scatter(X.loc[:, "age"], X.loc[:, "protime"], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.title('Age (yrs) vs Prothrombin Time in Hepatitis Patients (secs)')
plt.xlabel('Age (yrs)')
plt.ylabel('Prothrombin Time (secs)')
plt.show()


print(f'********* Perform K-Means on age vs protime - Z-Normalized ************************')
xNorm = patients.filter(["z-norm age", "z-norm protime"])
normKmeans = KMeans(n_clusters=7, random_state=0).fit(xNorm.values)
plt.scatter(xNorm.loc[:, "z-norm age"], xNorm.loc[:, "z-norm protime"], c=normKmeans.labels_.astype(float), s=50, alpha=0.5)
normCentroids = normKmeans.cluster_centers_
plt.scatter(normCentroids[:, 0], normCentroids[:, 1], c='red', s=50)
plt.title('Age vs Prothrombin Time in Hepatitis Patients')
plt.xlabel('Age')
plt.ylabel('Prothrombin Time')
plt.show()

print("***********   Add labels to the dataset  *****************************")
xNorm["class"] = patients["class"]
xNorm["labels"] = kmeans.labels_
print(xNorm.head())


print("***********   Create Train & Test *****************************")
print("Explanation: Just use the good ole 80/20 rule cause reasons")
TrainSet, TestSet=train_test_split(xNorm, test_size=0.2)

# Logistic regression classifier
print ('\n\n\n************** Logistic regression classifier ***********************\n')
C_parameter = 50. / len(xNorm) # parameter for regularization of the model
class_parameter = 'ovr' # parameter for dealing with multiple classes
penalty_parameter = 'l1' # parameter for the optimizer (solver) in the function
solver_parameter = 'saga' # optimization system used
tolerance_parameter = 0.1 # termination parameter
#####################

#Training the Model
clf = LogisticRegression(C=C_parameter, multi_class=class_parameter, penalty=penalty_parameter, solver=solver_parameter, tol=tolerance_parameter)
clf.fit(TrainSet.loc[:,["z-norm age", "z-norm protime", "labels"]], TrainSet.loc[:,"class"]) 
print ('coefficients:')
print (clf.coef_) # each row of this matrix corresponds to each one of the classes of the dataset
print ('intercept:')
print (clf.intercept_) # each element of this vector corresponds to each one of the classes of the dataset


# Apply the Model
print ('predictions for test set using logistic regression:')
predictions = clf.predict(TestSet.loc[:,["z-norm age", "z-norm protime", "labels"]])
TestSet["prediction"] = predictions


print("\n\n*****************  All The Test Data w/ Predictions for Logistic Regression ****************************")
print(TestSet.to_string())



print ('\n****************  Confusion Matrix and Metrics for Logistic Regression **************************')
Threshold = 0.6 # Some number between 0 and 1
print ("\nTHRESHOLD :", Threshold)
print ("\nEXPLANATION: Perhaps we don't mind false positives since we are talking about possible death")

BothProbabilities = clf.predict_proba(TestSet.loc[:,["z-norm age", "z-norm protime", "labels"]])
probabilities = BothProbabilities[:,1]
predictions = (probabilities > Threshold).astype(int)
print(predictions)
CM = confusion_matrix(TestSet.loc[:,"class"], predictions)
tn, fp, fn, tp = CM.ravel()
print ("TP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR = accuracy_score(TestSet.loc[:,"class"], predictions)
print ("Accuracy rate:", np.round(AR, 2))
P = precision_score(TestSet.loc[:,"class"], predictions)
print ("Precision:", np.round(P, 2))
R = recall_score(TestSet.loc[:,"class"], predictions)
print ("Recall:", np.round(R, 2))

# False Positive Rate, True Positive Rate, probability thresholds
fpr, tpr, th = roc_curve(TestSet.loc[:,"class"], probabilities)
AUC = auc(fpr, tpr)

plt.rcParams["figure.figsize"] = [10, 10] # Square
font = {'family' : 'normal', 'weight' : 'bold', 'size' : 18}
matplotlib.rc('font', **font)
plt.figure()
plt.title('ROC Curve of Logistic Regression')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(fpr, tpr, LW=3, label='ROC curve (AUC = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', LW=3, linestyle='--') # reference line for random classifier
plt.legend(loc="lower right")
plt.show()


# Random Forest classifier
estimators = 10 # number of trees parameter
mss = 2 # mininum samples split parameter
print ('\n\n ***********************   Random Forest classifier  ********************** \n')
clf = RandomForestClassifier(n_estimators=estimators, min_samples_split=mss) # default parameters are fine
clf.fit(TrainSet.loc[:,["z-norm age", "z-norm protime", "labels"]], TrainSet.loc[:,"class"]) 
predictions = clf.predict(TestSet.loc[:,["z-norm age", "z-norm protime", "labels"]])
TestSet["prediction"] = predictions

print("\n\n*****************  All The Test Data w/ Predictions for Random Forest ****************************")
print(TestSet.to_string())


print ('\n****************  Confusion Matrix and Metrics for Random Forest **************************')
Threshold = 0.6 # Some number between 0 and 1
print ("\nTHRESHOLD :", Threshold)
print ("\nEXPLANATION: Perhaps we don't mind false positives since we are talking about possible death")

BothProbabilities = clf.predict_proba(TestSet.loc[:,["z-norm age", "z-norm protime", "labels"]])
probabilities = BothProbabilities[:,1]
predictions = (probabilities > Threshold).astype(int)
print(predictions)
CM = confusion_matrix(TestSet.loc[:,"class"], predictions)
tn, fp, fn, tp = CM.ravel()
print ("TP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR = accuracy_score(TestSet.loc[:,"class"], predictions)
print ("Accuracy rate:", np.round(AR, 2))
P = precision_score(TestSet.loc[:,"class"], predictions)
print ("Precision:", np.round(P, 2))
R = recall_score(TestSet.loc[:,"class"], predictions)
print ("Recall:", np.round(R, 2))

# False Positive Rate, True Positive Rate, probability thresholds
fpr, tpr, th = roc_curve(TestSet.loc[:,"class"], probabilities)
AUC = auc(fpr, tpr)

plt.rcParams["figure.figsize"] = [10, 10] # Square
font = {'family' : 'normal', 'weight' : 'bold', 'size' : 18}
matplotlib.rc('font', **font)
plt.figure()
plt.title('ROC Curve of Random Forest Classifier')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(fpr, tpr, LW=3, label='ROC curve (AUC = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', LW=3, linestyle='--') # reference line for random classifier
plt.legend(loc="lower right")
plt.show()



print(
'''
Conclusion: After running each classifer 25 times the random forest classifier
more consistently presented a better ROC curve. Unfortunately I don't know
that much about Hepatitis so it's hard for me to say whether there's really
a cause and effect relationship here. BUT FOR X & Y the patient would not have
died. It's been a lot of fun and I look forward to the next class.
'''      
)

