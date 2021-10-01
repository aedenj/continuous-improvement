#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Aeden Jameson

Project: Assignment 4

Description:
  
Re: Missing & Outlier Values

All of the following variables had both missing values, indicated by '?',
and outliers

    Number of sexual partners
    First sexual intercourse
    Num of pregnancies
    Hormonal Contraceptives
    Hormonal Contraceptives (years)
    IUD (years)
    STDs: Number of diagnosis
    STDs: Time since first diagnosis 
    STDs: Time since last diagnosis

Values were considered outliers if they fell outside two standard deviations of 
the mean. All missing values were replaced because it appeared that all columns
had a sensible median. 


Re: Histograms

All of the attributes above were histogram because they could all tell
you something meaningful about the population. 


Re: Removing Attributes

No attributes were removed because all the numeric attributes could tell you
something under analysis


Re: Removing Rows

No rows were removed. After sifting throught the data I didn't see obvious
candidates, but I'm probably wrong.

    
"""

# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv'
subjects = pd.read_csv(url)
subjects.dtypes


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

# a helper method that returns a boolean array against that
# data that flags whether said value falls within two standard
# deviations 
def gaussian_dist_filter(data, column):
    hi = np.mean(data.loc[:,column]) + 2*np.std(data.loc[:,column])
    low = np.mean(data.loc[:,column]) - 2*np.std(data.loc[:,column])
    flag_good = (data.loc[:,column] >= low) & (data.loc[:,column] <= hi)
    
    return flag_good

# replaces missing data and outliers
def cleanse(data, column, dtype): 
  missing_fixed = replace_missing(data, column, dtype)
  return replace_outliers(missing_fixed, column)


# clean and print out a plot for each of the columns in the array
# below.
columns = [
    "Number of sexual partners",
    "First sexual intercourse",
    "Num of pregnancies",
    "Hormonal Contraceptives",
    "Hormonal Contraceptives (years)",
    "IUD (years)",
    "STDs: Number of diagnosis",
    "STDs: Time since first diagnosis", 
    "STDs: Time since last diagnosis"
]
for c in columns: 
  subjects = cleanse(subjects, c, int)
  print(f'Title: {c}')
  plt.hist(subjects.loc[:, c])
  plt.show()
  #print(f'Mean: {}, Median: {}')


scatter_matrix(subjects[columns])
plt.show()


print("Table of Standard Deviations:")
for c in columns:
    print(f'{c}: {np.std(subjects.loc[:,c])}')






