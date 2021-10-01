"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

# import package
import pandas as pd

# Download the data
# http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data"
Mamm = pd.read_csv(url, header=None)
Mamm.columns = ["BI-RADS", "Age", "Shape", "Margin", "Density", "Severity"] 
##################

# The category columns are decoded and missing values are imputed
Mamm.loc[ Mamm.loc[:, "Shape"] == "1", "Shape"] = "round"
Mamm.loc[Mamm.loc[:, "Shape"] == "2", "Shape"] = "oval"
Mamm.loc[Mamm.loc[:, "Shape"] == "3", "Shape"] = "lobular"
Mamm.loc[Mamm.loc[:, "Shape"] == "4", "Shape"] = "irregular"
Mamm.loc[Mamm.loc[:, "Shape"] == "?", "Shape"] = "irregular"
Mamm.loc[Mamm.loc[:, "Margin"] == "1", "Margin"] = "circumscribed"
Mamm.loc[Mamm.loc[:, "Margin"] == "2", "Margin"] = "microlobulated"
Mamm.loc[Mamm.loc[:, "Margin"] == "3", "Margin"] = "obscured"
Mamm.loc[Mamm.loc[:, "Margin"] == "4", "Margin"] = "ill-defined"
Mamm.loc[Mamm.loc[:, "Margin"] == "5", "Margin"] = "spiculated"
Mamm.loc[Mamm.loc[:, "Margin"] == "?", "Margin"] = "circumscribed"
####################

# Check the first rows of the data frame
Mamm.head()
####################

# Get the counts for each value
Mamm.loc[:,"Shape"].value_counts()
Mamm.loc[:,"Margin"].value_counts()
####################

# Plot the counts for each category
Mamm.loc[:,"Shape"].value_counts().plot(kind='bar')
####################

# Simplify Shape by consolidating oval and round
Mamm.loc[Mamm.loc[:, "Shape"] == "round", "Shape"] = "oval"

# Plot the counts for each category
Mamm.loc[:,"Shape"].value_counts().plot(kind='bar')
####################

# Plot the counts for each category
Mamm.loc[:,"Margin"].value_counts().plot(kind='bar')
####################

# Simplify Margin by consolidating ill-defined, microlobulated, and obscured
Mamm.loc[Mamm.loc[:, "Margin"] == "microlobulated", "Margin"] = "ill-defined"
Mamm.loc[Mamm.loc[:, "Margin"] == "obscured", "Margin"] = "ill-defined"

# Plot the counts for each category
Mamm.loc[:,"Margin"].value_counts().plot(kind='bar')

#####################