#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print(
'''
Author: Aeden Jameson

Project: Assignment 7

Data Description: Hepatitis Data Set 

  I. Source: https://archive.ics.uci.edu/ml/datasets/Heart+Disease
    
  
  II. Characteristics:
      
    Number of Attributes: 14
    Number of Observations: 303
    
    
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

     age                     float64   normal distribution
     sex                     float64   male: 206, female: 97
     chest pain type         float64   skewed left
     resting blood press     float64   skewed right
     serum cholestoral       float64   normal distribution
     fasting blood sugar     float64   true: 45, false 258
     electrocardiographic    float64   
     maximum heart rate      float64
     exercise induced angina float64
     oldpeak                 float64
     slope                   float64
     # major vessels         object
     thal                    object
     prediction              int64
     
  III. Missing & Outlier Values

    All of the following variables had both missing values, indicated by '?',
    and outliers,
    
        protime
        sgot  
        
  Values were considered outliers if they fell outside two standard deviations of 
  the mean. All missing values were replaced because it appeared that all columns
  had a sensible median. 
    

  IV. Removed Attributes

    The age attribute was removed in favor of one-hot encoding it.


  V. Removed Rows

    No rows were removed. After sifting throught the data I didn't see obvious
    candidates, but I'm probably wrong.
'''
)
    
from sklearn.preprocessing import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
patients = pd.read_csv(url, header=None)
patients.dtypes
patients.columns = ["age", "sex", "chest pain type", "resting blood pressure", 
  "serum cholestoral", "fasting blood sugar", "electrocardiographic",
  "maximum heart rate", "exercise induced angina", "oldpeak", "slope",
  "# major vessels", "thal", "prediction"]

plt.hist(patients.loc[:,"fasting blood sugar"])

patients["fasting blood sugar"].value_counts()

 


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
    "age",
    "alk phosphate",
    "bilirubin"
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



plt.scatter(patients.loc[:,"age"], patients.loc[:,"albumin"], alpha=a, s=s, c=color, edgecolors=ec)

print(
'''
QUESTION: Are hepatitis patients 50-60 years of age more likely to have longer      
prothrombin times?
'''      
)

print(
'''
QUESTION: In what age range are hepatitis patients most likely to have the longest prothrombin times?
'''      
)

print(f'********* Perform K-Means on Age vs Prothrombin Time - Not Normalized ************************')
X = patients.filter(["age", "protime"])
kmeans = KMeans(n_clusters=4, random_state=0).fit(X.values)
plt.scatter(X.loc[:, "age"], X.loc[:, "protime"], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.title('Age (yrs) vs Prothrombin Time in Hepatitis Patients (secs)')
plt.xlabel('Age (yrs)')
plt.ylabel('Prothrombin Time (secs)')
plt.show()


print(f'********* Perform K-Means on Age vs Prothrombin Time - Z-Normalized ************************')
xNorm = patients.filter(["z-norm age", "z-norm protime"])
normKmeans = KMeans(n_clusters=7, random_state=0).fit(xNorm.values)
plt.scatter(xNorm.loc[:, "z-norm age"], xNorm.loc[:, "z-norm protime"], c=normKmeans.labels_.astype(float), s=50, alpha=0.5)
normCentroids = normKmeans.cluster_centers_
plt.scatter(normCentroids[:, 0], normCentroids[:, 1], c='red', s=50)
plt.title('Age vs Prothrombin Time in Hepatitis Patients')
plt.xlabel('Age')
plt.ylabel('Prothrombin Time')
plt.show()
