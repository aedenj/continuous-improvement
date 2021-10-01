#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Aeden Jameson

Project: Assignment 5

Description:
  
  The varialbles chosen for plotted were chosen because they had a wide range of
  data a and looked interesting. The variables that were decoded, imputed, consolidated, 
  and one-hot encoded are identified in the print out.

    
"""


# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import *

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
cars = pd.read_csv(url, header=None)

cars.columns = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration",
                "num-of-doors", "body-style", "drive-wheels", "engine-location",
                "wheel-base", "length", "width", "height", "curb-weight", "engine-type",
                "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke",
                "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg",
                "price"]

############
def bin(x, b): # x = data array, b = boundaries array
    nb = len(b)
    N = len(x)
    y = np.empty(N, int) # empty integer array to store the bin numbers (output)
    
    for i in range(1, nb): # repeat for each pair of bin boundaries
        y[(x >= b[i-1])&(x < b[i])] = i
    
    y[x == b[-1]] = nb - 1 # ensure that the borderline cases are also binned appropriately
    return y
#############

# Returns the data set with the missing values replaced with the median. 
def replace_missing(data, column, dtype):
    data.loc[:, column] = pd.to_numeric(data.loc[:, column], errors='coerce')
    HasNan = np.isnan(data.loc[:, column])
    data.loc[HasNan, column] = np.nanmedian(data.loc[:,column])
    data.loc[:,column] = data.loc[:,column].astype(int)
    return data




print("*****************  Z-Normalize thecompression-ratio Column **********************\n")
print("First 10 values of original compression-ratio data:")
print(cars[:10]["compression-ratio"].values)

offset = np.mean(cars.loc[:,"compression-ratio"])
spread = np.std(cars.loc[:,"compression-ratio"])
cars["compression-ratio"] = (cars["compression-ratio"] - offset)/spread

print("\n")
print("First 10 values of Number of sexual partners data after Z-Normalizing:")
print(cars[:10]["compression-ratio"].values)

print("\n***************** End Z-Normalization ****************************")

print("\n\n")

print("*****************  Equal Width Bin city-mpg Column **********************")
print(f'Min Age: {np.min(cars["city-mpg"].values)}')
print(f'Max Age: {np.max(cars["city-mpg"].values)}')

print("\n")
print("First 10 values of original Age data:")
print(cars[:10]["city-mpg"].values)
NB = 6
bounds = np.linspace(np.min(cars["city-mpg"].values), np.max(cars["city-mpg"].values), NB + 1) 
bx = bin(cars["city-mpg"].values, bounds)

print ("\n\nBin boundaries: ", bounds)

print("\n")
print("First 10 values of Age data after binning")
print(bx[:10])

print("\n***************** End Equal Width Binning **************************\n")


print("***************** Decode Column **********************")
print("\n All the columns are already decoded. Finding data sets that required all the steps in the assignment is very hard.")
print("***************** End Decode Column **********************\n\n")


print("***************** Impute Missing Categories **********************")
print("Frequency table for the column num-of doors BEFORE imputation:")
print(cars["num-of-doors"].value_counts())

cars.loc[cars.loc[:, "num-of-doors"] == "?", "num-of-doors"] = "four"

print("Frequency table for the column num-of doors AFTER imputation:")
print(cars["num-of-doors"].value_counts())


print("***************** End Impute Missing Categories **********************")


print("***************** Consolidate Categorical Data **********************")
print("Frequency table for the column body-style BEFORE consolidation:")
print(cars["body-style"].value_counts())

cars.loc[cars.loc[:, "body-style"] == "hardtop", "body-style"] = "sport"
cars.loc[cars.loc[:, "body-style"] == "wagon", "body-style"] = "non-sport"
cars.loc[cars.loc[:, "body-style"] == "sedan", "body-style"] = "non-sport"
cars.loc[cars.loc[:, "body-style"] == "hatchback", "body-style"] = "non-sport"
cars.loc[cars.loc[:, "body-style"] == "convertible", "body-style"] = "sport"

print("Frequency table for the column body-style AFTER consolidation:")
print(cars["body-style"].value_counts())

print("***************** End Consolidate Categorical Data **********************")




print("***************** One Hot Encode Drive Wheels Column **********************")
cars.loc[:, "4wd"] = (cars.loc[:, "drive-wheels"] == "4wd").astype(int)
cars.loc[:, "fwd"] = (cars.loc[:, "drive-wheels"] == "fwd").astype(int)
cars.loc[:, "rwd"] = (cars.loc[:, "drive-wheels"] == "rwd").astype(int)

print(cars[["drive-wheels", "4wd", "fwd", "rwd"]])
print("***************** End One Hot Encode Drive Wheels Column **********************")


print("***************** Remove Obsolete Drive Wheels Column **********************")
# Remove obsolete column
cars = cars.drop("drive-wheels", axis=1)
print("***************** End Remove Obsolete Drive Wheels Column **********************")



print("Title: Car Makes")
cars.loc[:,"make"].value_counts().plot(kind='bar')
plt.show()

print("Title: Engine Types")
cars.loc[:,"engine-type"].value_counts().plot(kind='bar')
plt.show()


