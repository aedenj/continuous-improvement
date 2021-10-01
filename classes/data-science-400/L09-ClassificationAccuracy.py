"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

# import package
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
import matplotlib.pyplot as plt
import matplotlib
from pandas.plotting import scatter_matrix

# Download the data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data"
Mamm = pd.read_csv(url, header=None)
Mamm.columns = ["BI-RADS", "Age", "Shape", "Margin", "Density", "Severity"]



# Remove rows with missing values
Mamm = Mamm.replace(to_replace="?", value=float("NaN"))
Mamm = Mamm.dropna(axis=0)

# Coerce to numeric
Mamm.loc[:, "BI-RADS"] = pd.to_numeric(Mamm.loc[:, "BI-RADS"], errors='coerce')
Mamm.loc[Mamm.loc[:, "BI-RADS"] > 6, "BI-RADS"] = 6
Mamm.loc[:, "Age"] = pd.to_numeric(Mamm.loc[:, "Age"], errors='coerce')
Mamm.loc[:, "Density"] = pd.to_numeric(Mamm.loc[:, "Density"], errors='coerce')

# The category columns are decoded, missing values are imputed, and categories
# are consolidated
Mamm.loc[ Mamm.loc[:, "Shape"] == "1", "Shape"] = "oval"
Mamm.loc[Mamm.loc[:, "Shape"] == "2", "Shape"] = "oval"
Mamm.loc[Mamm.loc[:, "Shape"] == "3", "Shape"] = "lobular"
Mamm.loc[Mamm.loc[:, "Shape"] == "4", "Shape"] = "irregular"
Mamm.loc[:, "oval"] = (Mamm.loc[:, "Shape"] == "oval").astype(int)
Mamm.loc[:, "lobul"] = (Mamm.loc[:, "Shape"] == "lobular").astype(int)
Mamm.loc[:, "irreg"] = (Mamm.loc[:, "Shape"] == "irregular").astype(int)
Mamm = Mamm.drop("Shape", axis=1)

Mamm.loc[Mamm.loc[:, "Margin"] == "1", "Margin"] = "circumscribed"
Mamm.loc[Mamm.loc[:, "Margin"] == "2", "Margin"] = "ill-defined"
Mamm.loc[Mamm.loc[:, "Margin"] == "3", "Margin"] = "ill-defined"
Mamm.loc[Mamm.loc[:, "Margin"] == "4", "Margin"] = "ill-defined"
Mamm.loc[Mamm.loc[:, "Margin"] == "5", "Margin"] = "spiculated"
Mamm.loc[:, "ill-d"] = (Mamm.loc[:, "Margin"] == "ill-defined").astype(int)
Mamm.loc[:, "circu"] = (Mamm.loc[:, "Margin"] == "circumscribed").astype(int)
Mamm.loc[:, "spicu"] = (Mamm.loc[:, "Margin"] == "spiculated").astype(int)
Mamm = Mamm.drop("Margin", axis=1)

##############
print ('\nVerify that all variables are numeric')
print(Mamm.dtypes)


##############
print ('\nDetermine Model Accuracy')

TestFraction = 0.3
print ("Test fraction is chosen to be:", TestFraction)

print ('\nSimple approximate split:')
isTest = np.random.rand(len(Mamm)) < TestFraction
TrainSet = Mamm[~isTest]
TestSet = Mamm[isTest] # should be 249 but usually is not
print ('Test size should have been ', 
       TestFraction*len(Mamm), "; and is: ", len(TestSet))

print ('\nsklearn accurate split:')
TrainSet, TestSet = train_test_split(Mamm, test_size=TestFraction)
print ('Test size should have been ', 
       TestFraction*len(Mamm), "; and is: ", len(TestSet))

print ('\n Use logistic regression to predict Severity from other variables in Mamm')
Target = "Severity"
Inputs = list(Mamm.columns)
Inputs.remove(Target)
clf = LogisticRegression()
clf.fit(TrainSet.loc[:,Inputs], TrainSet.loc[:,Target])
BothProbabilities = clf.predict_proba(TestSet.loc[:,Inputs])
probabilities = BothProbabilities[:,1]

print ('\nConfusion Matrix and Metrics')
Threshold = 0.5 # Some number between 0 and 1
print ("Probability Threshold is chosen to be:", Threshold)
predictions = (probabilities > Threshold).astype(int)
print(TestSet.to_string())
CM = confusion_matrix(TestSet.loc[:,Target], predictions)
tn, fp, fn, tp = CM.ravel()
print ("TP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR = accuracy_score(TestSet.loc[:,Target], predictions)
print ("Accuracy rate:", np.round(AR, 2))
P = precision_score(TestSet.loc[:,Target], predictions)
print ("Precision:", np.round(P, 2))
R = recall_score(TestSet.loc[:,Target], predictions)
print ("Recall:", np.round(R, 2))

 # False Positive Rate, True Posisive Rate, probability thresholds
fpr, tpr, th = roc_curve(TestSet.loc[:,Target], probabilities)
AUC = auc(fpr, tpr)

plt.rcParams["figure.figsize"] = [8, 8] # Square
font = {'family' : 'normal', 'weight' : 'bold', 'size' : 18}
matplotlib.rc('font', **font)
plt.figure()
plt.title('ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(fpr, tpr, LW=3, label='ROC curve (AUC = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', LW=3, linestyle='--') # reference line for random classifier
plt.legend(loc="lower right")
plt.show()

##############
