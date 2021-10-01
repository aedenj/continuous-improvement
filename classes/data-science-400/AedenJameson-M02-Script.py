#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Aeden Jameson

Project: Milestone 2

Description:

  Re: Missing & Outlier Values

  All of the following variables had both missing values, indicated by '?',
  and outliers,
    
        protime
        sgot  
        
  Values were considered outliers if they fell outside two standard deviations of 
  the mean. All missing values were replaced because it appeared that all columns
  had a sensible median. 
    

  Re: Removing Attributes

  The age attribute was removed in favor of one-hot encoding it.


  Re: Removing Rows

  No rows were removed. After sifting throught the data I didn't see obvious
  candidates, but I'm probably wrong.
"""

from sklearn.preprocessing import *
import numpy as np
import pandas as pd

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
    "sgot"   
]
for c in columns:   
  print(f'*********** Replace Missing: {c} Column *******************')
  print(f'# Values Missing: {len(list(patients.loc[patients.loc[:, c] == "?", c]))}\n')
  print(f'Original Values: {list(patients.loc[:, c])}\n')
  patients = replace_missing(patients, c, int)
  print(f'# Values After: {len(list(patients.loc[patients.loc[:, c] == "?", c]))}\n')
  print(f'After Replace: {list(patients.loc[:, c])}')
  print('\n')

for c in columns: 
  flag_good = gaussian_dist_filter(patients, c)
  print(f'*********** Replace Outliers: {c} Column *******************')
  print(f'Original Values: {list(patients.loc[:, c])}\n')
  print(f'Outliers: {list(patients.loc[~flag_good, c])}\n')
  patients = replace_outliers(patients, c)
  print(f'After Replace: {list(patients.loc[:, c])}\n')
  print('\n')


for c in columns: 
  print(f'*********** Z-Normalize: {c} Column *******************')
  print(f'Values Before: {list(patients.loc[:,c])}\n')
  offset = np.mean(list(patients.loc[:, c]))
  spread = np.std(list(patients.loc[:, c]))
  patients[f'z-norm {c}'] = (patients.loc[:, c] - offset)/spread
  print(f'Z-Norm Values: {list(patients.loc[:, "z-norm " + c])}')
  print('\n')


print(f'*********** Bin: age Column *******************')
NB = 7
print(f'# Bins: {NB}\n')
bins  = np.linspace(np.min(patients.loc[:, "age"]), np.max(patients.loc[:, "age"]), NB + 1) 
labels = [1,2,3,4,5,6,7]
print(f'Bins: {bins}\n')
patients["binned age"] = pd.cut(patients["age"], bins=bins, labels=labels)
print(f'Binned Values:\n {patients.loc[:, "binned age"]}')
print('\n')

print(f'*********** Consolidate:  *******************')
print('We would not do this in practice, but this dataset is lacking fields for consoliation.\n')
print("Spread before consolidation:")
print(patients.loc[:, "class"].value_counts())
patients.loc[patients.loc[:, "class"] == 1, "class"] = 2
print("Spread after consolidation:")
print(patients.loc[:, "class"].value_counts())
print('\n')


print(f'*********** One-hot encode: age *******************') 
patients.loc[:, "male"] = (patients.loc[:, "sex"] == 1).astype(int)
patients.loc[:, "female"] = (patients.loc[:, "sex"] == 2).astype(int)
print("After Encoding:\n")
print(patients.loc[:, "male":"female"])
print('\n')
  
print(f'*********** Remove obsolete columns:  *******************\n')
print("Remove 'sex' column")
print("After Removal: ")
patients = patients.drop("sex", axis=1)
print(list(patients.columns))

