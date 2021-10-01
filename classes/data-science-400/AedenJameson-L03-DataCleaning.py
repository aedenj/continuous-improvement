# -*- coding: utf-8 -*-
"""
Author: Aeden Jameson

Project: Assignment 3

Description:
    
This script demonstrates the removal and replacement of various kinds
of aberrant data with two sets of data that describle daily temperatures at
various times of the day in the months of September and October in Seattle.

Individual values are removed and replaced according to whether they are a 
valid type or fall within two standard deviations of the mean.

Outliers:
    The arr1 dataset describes temperatures in Seattle in the month of September.
    There are a few outlier values that are mostly due to faulty recording.
    
          -64, -54 -- It's very unlikely it was every this cold this drasitically
                      at in point in the month so it's safe to replace.
                      
          114 -- It's very unlikely it's this hot during the time the dataset
                 covers so it's safe to replace.
"""

import numpy as np

# An array of temperatures (F) of Seattle in the month of September
arr1 = np.array([64, 57, 47,47, -64, 59, 53, 47, 62, 55, -54, 66, 64, 61, 50, 59,
                 58, 49, 51, 59, 58, 67, 114, 60, 58, 55, 63, 49, 50, 54 ])

# An array of temperatures (F) of Seattle in the month of October
arr2 = np.array([64, 57, 47,47, "?", 59, 53, 47, 62, 55, "", 66, 64, 61, 50, 59,
                 58, 49, 51, 59, 58, 67, "nan", 60, 58, 55, 63, 49, 50, 54 ])

# Returns the data with the outliers removed. Values that lie two standard 
# deviations from the mean are removed.
def remove_outlier(data):    
    return data[gaussian_dist_filter(data)]


# Returns the data with the outliers replaced with the mean. Values that 
# lie two standard deviations from the mean are replaced.
def replace_outlier(data):
    flag_good = gaussian_dist_filter(data)
    data[~flag_good] = np.mean(data[flag_good])

    return data

# a helper method that returns a boolean array against that
# data that flags whether said value falls within two standard
# deviations 
def gaussian_dist_filter(data):
    hi = np.mean(data) + 2*np.std(data)
    low = np.mean(data) - 2*np.std(data)
    flag_good = (data >= low) & (data <= hi)
    
    return flag_good

# Returns the data with the missing values replaced with the median. 
def fill_median(data):
    flag_bad =  [not element.isdigit() for element in data]
    data[flag_bad] = np.median(remove_missing(data)).astype(int)    
    return data.astype(int)

# Returns an array with non-numeric elements of the list removed
def remove_missing(data):
    flag_good = [element.isdigit() for element in data]
    cleaned = data[flag_good].astype(int)
    return cleaned   

flag_good = gaussian_dist_filter(arr1)
print('*********** Removing outliers in arr1 *******************')
print(f'Initial arr1:\n {arr1}\n')
print(f'Arr1 outliers:\n {arr1[~flag_good]}\n')
print(f'Arr1 after removal:\n {remove_outlier(arr1)}\n')
print('\n')
print('*********** Replacing outliers in arr1 *******************')
print(f'Initial arr1:\n {arr1}\n')
print(f'Arr1 outliers:\n {arr1[~flag_good]}\n')
print(f'Mean used for replacement: {np.mean(arr1[flag_good])}\n')
print(f'Arr1 after replacement:\n {replace_outlier(arr1)}\n')

print('*********** Replacing missing values in arr2 *******************')
print(f'Initial arr2:\n {arr2}\n')
print(f'Median of arr2: {np.median(remove_missing(arr2)).astype(int)}\n')
print(f'Arr2 after replacing missing values:\n {fill_median(arr2)}')

